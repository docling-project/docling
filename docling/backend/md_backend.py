# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import html
import logging
import re
import warnings
from copy import deepcopy
from enum import Enum
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    GroupLabel,
    ImageRef,
    ListItem,
    NodeItem,
    RefItem,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)
from pydantic import AnyUrl, BaseModel, Field, TypeAdapter
from typing_extensions import Annotated, override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
)
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import (
    HTMLBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError
from docling.utils.code_language import detect_code_language
from docling.utils.text_decoding import decode_text

_MARKO_AVAILABLE: bool = False
_MARKO_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    import marko
    import marko.element
    import marko.inline
    from marko import Markdown
    from marko.ext.gfm import elements as _gfm_el

    _MARKO_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _MARKO_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'marko' package is required to process Markdown files. "
    "Install it with `pip install 'docling-slim[format-markdown]'`."
)

_MARKER_BODY = "DOCLING_DOC_MD_HTML_EXPORT"
_START_MARKER = f"#_#_{_MARKER_BODY}_START_#_#"
_STOP_MARKER = f"#_#_{_MARKER_BODY}_STOP_#_#"


class _PendingCreationType(str, Enum):
    """CoordOrigin."""

    HEADING = "heading"
    LIST_ITEM = "list_item"


class _HeadingCreationPayload(BaseModel):
    kind: Literal["heading"] = "heading"
    level: int


class _ListItemCreationPayload(BaseModel):
    kind: Literal["list_item"] = "list_item"
    enumerated: bool
    marker: str = ""


_CreationPayload = Annotated[
    _HeadingCreationPayload | _ListItemCreationPayload,
    Field(discriminator="kind"),
]


def _only_plain_line_breaks(children: list) -> bool:
    """Return True when children consist solely of RawText/Literal runs separated
    by LineBreak nodes (soft or hard), with at least one break present.

    Such content is fully handled by the pending-line-break merge paths and does
    not need an inline_group wrapper.  Multiple plain runs without any LineBreak
    (e.g. `RawText + Literal + RawText` from an escaped character) return False
    so they still use the regular inline_group path.

    Args:
        children: Inline child nodes of a marko Paragraph, Heading, or ListItem
            paragraph to inspect.

    Returns:
        True if all children are plain-text runs joined only by line breaks,
        False otherwise.
    """
    if not _MARKO_AVAILABLE:
        return False
    has_break = any(isinstance(c, marko.inline.LineBreak) for c in children)
    return has_break and all(
        isinstance(c, (marko.inline.RawText, marko.inline.Literal))
        or isinstance(c, marko.inline.LineBreak)
        for c in children
    )


class MarkdownDocumentBackend(DeclarativeDocumentBackend):
    _ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|\w+);")
    _DELIMITER_CELL_RE = re.compile(r":?-+:?")
    _PIPE_ENTITY = "&#124;"

    @staticmethod
    def _apply_formatting(
        current: Formatting | None,
        *,
        bold: bool = False,
        italic: bool = False,
        strikethrough: bool = False,
    ) -> Formatting:
        """Return a copy of `current` with the requested flag(s) set.

        If `current` is `None` a fresh `Formatting` instance is created.
        Always returns a new object so the caller's reference is never mutated
        in place.

        Args:
            current: The inherited `Formatting` state, or `None`.
            bold: Set the bold flag when `True`.
            italic: Set the italic flag when `True`.
            strikethrough: Set the strikethrough flag when `True`.

        Returns:
            A `Formatting` instance with the requested flag(s) applied.
        """
        fmt = deepcopy(current) if current else Formatting()
        if bold:
            fmt.bold = True
        if italic:
            fmt.italic = True
        if strikethrough:
            fmt.strikethrough = True
        return fmt

    @staticmethod
    def _resolve_link_dest(dest: str) -> AnyUrl | Path | None:
        """Parse a link destination string into an `AnyUrl` or `Path`.

        Shared between the paragraph inline-element walker and the table-cell
        inline walker so that link-destination resolution is not duplicated.

        Args:
            dest: The raw destination string from a marko `Link` or
                `AutoLink` element.

        Returns:
            An `AnyUrl` or `Path` instance, or `None` when the destination
            cannot be parsed.
        """
        return TypeAdapter(AnyUrl | Path | None).validate_python(dest)

    @staticmethod
    def _split_table_row(row: str) -> list[str]:
        """Split a table row into its cells.

        The leading and trailing pipes are optional in GFM, so an empty field is
        only dropped when it comes from a pipe at the very edge of the row.
        """
        cells = row.split("|")
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [cell.strip() for cell in cells]

    @staticmethod
    def _is_delimiter_row(row: str) -> bool:
        """Whether a row is a GFM table delimiter row, e.g. ``--- | :---:``."""
        cells = MarkdownDocumentBackend._split_table_row(row)
        return bool(cells) and all(
            MarkdownDocumentBackend._DELIMITER_CELL_RE.fullmatch(cell) for cell in cells
        )

    @staticmethod
    def _escape_pipes(text: str) -> str:
        """Carry a pipe that is cell content rather than a cell delimiter.

        An entity is how the row buffer already spells such a pipe: a source
        ``&#124;`` survives ``_unescape_except_pipe`` intact and ``_close_table``
        turns it back into ``|`` once the cells are split. A backslash-escaped
        pipe has to join it there, because Marko resolves ``\\|`` to a Literal
        node holding a bare ``|``, which the buffer cannot tell from markup.
        """
        return text.replace("|", MarkdownDocumentBackend._PIPE_ENTITY)

    @staticmethod
    def _inline_text(node) -> str:
        """The text of an inline node, its markers dropped.

        A pipe goes on delimiting cells inside emphasis, code spans and links,
        so the markers can go but the text they wrap has to stay.
        """
        children = getattr(node, "children", None)
        if isinstance(children, str):
            # A Literal is a backslash escape, so its pipe is content.
            if isinstance(node, marko.inline.Literal):
                return MarkdownDocumentBackend._escape_pipes(children)
            return children
        return "".join(
            MarkdownDocumentBackend._inline_text(child) for child in children or []
        )

    @staticmethod
    def _starts_pipeless_table(element: marko.block.Paragraph) -> bool:
        """Whether a paragraph is a GFM table whose header has no leading pipe.

        Tables that do start with a pipe are detected line by line and must not
        go through here, so that their existing behaviour is left untouched.
        Without a leading pipe the header is indistinguishable from prose, so
        the delimiter row on the second line is the only reliable signal - hence
        the lookahead at paragraph level, where all lines are visible at once.
        """
        # Rebuilt line by line rather than read off the RawText nodes: a header
        # cell in bold or a link is a node of its own, so reading those alone
        # would split one line into several and shift the delimiter row away.
        lines = [""]
        for child in element.children:
            if isinstance(child, marko.inline.LineBreak):
                if len(lines) == 2:
                    break
                lines.append("")
            else:
                lines[-1] += MarkdownDocumentBackend._inline_text(child)
        if len(lines) < 2 or lines[0].lstrip().startswith("|"):
            return False
        if "|" not in lines[0] or not MarkdownDocumentBackend._is_delimiter_row(
            lines[1]
        ):
            return False
        # GFM: "The delimiter row must match the header row in the number of
        # cells. If not, a table will not be recognized."
        return len(MarkdownDocumentBackend._split_table_row(lines[0])) == len(
            MarkdownDocumentBackend._split_table_row(lines[1])
        )

    # md_table_buffer holds the rows of the table being read, one string per
    # row, in a form _split_table_row can split on "|" and _close_table decodes
    # exactly once with unescape(): a RawText is decoded on the way in except
    # for the pipe entities, so a pipe that is cell content survives the split;
    # literal text, such as a code span, is entity-encoded on the way in and
    # its pipes become &#124;.
    @staticmethod
    def _unescape_except_pipe(text: str) -> str:
        def replace(match):
            entity = match.group(0)
            decoded = unescape(entity)

            # Any spelling of | (&#x7c;, &verbar;, ...) stays encoded so it is not
            # taken for a cell delimiter; _close_table unescapes it after the split.
            if decoded == "|":
                return entity

            return decoded

        return MarkdownDocumentBackend._ENTITY_RE.sub(replace, text)

    def _shorten_underscore_sequences(self, markdown_text: str, max_length: int = 10):
        pattern = r"_+"

        def replace_match(match):
            underscore_sequence = match.group(0)

            if len(underscore_sequence) > max_length:
                return "_" * max_length
            else:
                return underscore_sequence

        shortened_text = re.sub(pattern, replace_match, markdown_text)

        if len(shortened_text) != len(markdown_text):
            warnings.warn("Detected potentially incorrect Markdown, correcting...")

        return shortened_text

    def _shorten_leading_dash_sequences(
        self, markdown_text: str, max_length: int = 10
    ) -> str:
        pattern = re.compile(
            rf"^([ \t]*)(?:-\s+){{{max_length + 1},}}-?(?=\S)", re.MULTILINE
        )
        shortened_text, count = pattern.subn(r"\1- ", markdown_text)

        if count > 0:
            warnings.warn("Detected potentially incorrect Markdown, correcting...")

        return shortened_text

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: MarkdownBackendOptions | None = None,
    ):
        # Raised first so a missing optional dependency gives an actionable
        # message rather than a NameError when marko is dereferenced below.
        if not _MARKO_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _MARKO_IMPORT_ERROR
        if options is None:
            options = MarkdownBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        _log.debug("Starting MarkdownDocumentBackend...")

        self.path_or_stream = path_or_stream
        self.valid = True
        self.markdown = ""

        self.in_table = False
        self.in_pipeless_table = False
        self.md_table_buffer: list[str] = []
        self._pending_hard_line_break = False
        self._pending_soft_line_break = False
        self._html_blocks: int = 0
        self._image_loader: ImageResourceLoader | None = None

        # A leading BOM is dropped. Kept, it prefixes the first line, so a
        # leading "# Title" is parsed as paragraph text and the BOM reaches the
        # output.
        try:
            md_content = decode_text(self.path_or_stream, options.encoding)
            # remove invalid sequences
            # very long sequences of underscores will lead to unnecessary long processing times.
            # In any proper Markdown files, underscores have to be escaped,
            # otherwise they represent emphasis (bold or italic)
            self.markdown = self._shorten_underscore_sequences(md_content)
            self.markdown = self._shorten_leading_dash_sequences(self.markdown)
            self.valid = True

            _log.debug(self.markdown)
        except DocumentLoadError:
            # Already carries a message naming what could not be decoded.
            raise
        except Exception as e:
            raise DocumentLoadError(
                f"Could not initialize MD backend for file with hash {self.document_hash}."
            ) from e
        return

    @staticmethod
    def _encode_table_literal(text: str) -> str:
        """Encode literal text for md_table_buffer: entities are escaped so the
        decode in _close_table returns the text as written, and a pipe is not
        a column separator."""
        return MarkdownDocumentBackend._escape_pipes(html.escape(text, quote=False))

    def _append_table_text(self, text: str) -> None:
        """Add text to the current row of md_table_buffer."""
        if self.md_table_buffer:
            self.md_table_buffer[-1] += text
        else:
            self.md_table_buffer.append(text)

    def _close_table(self, doc: DoclingDocument):
        self.in_pipeless_table = False
        if self.in_table:
            _log.debug("=== TABLE START ===")
            for md_table_row in self.md_table_buffer:
                _log.debug(md_table_row)
            _log.debug("=== TABLE END ===")
            tcells: list[TableCell] = []
            result_table = []
            for n, md_table_row in enumerate(self.md_table_buffer):
                data = []
                if n == 0:
                    header = MarkdownDocumentBackend._split_table_row(md_table_row)
                    for value in header:
                        data.append(value)
                    result_table.append(data)
                if n > 1:
                    values = MarkdownDocumentBackend._split_table_row(md_table_row)
                    for value in values:
                        data.append(value)
                    result_table.append(data)

            # GFM: "The remainder of the table's rows may vary in the number of
            # cells. If a row has fewer cells than the header row, empty cells
            # are inserted. If it has greater, the excess is ignored."
            if result_table and result_table[0]:
                num_header_cells = len(result_table[0])
                result_table = [
                    row[:num_header_cells] + [""] * (num_header_cells - len(row))
                    for row in result_table
                ]

            for trow_ind, trow in enumerate(result_table):
                for tcol_ind, cellval in enumerate(trow):
                    row_span = (
                        1  # currently supporting just simple tables (without spans)
                    )
                    col_span = (
                        1  # currently supporting just simple tables (without spans)
                    )
                    icell = TableCell(
                        text=unescape(cellval.strip()),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=trow_ind,
                        end_row_offset_idx=trow_ind + row_span,
                        start_col_offset_idx=tcol_ind,
                        end_col_offset_idx=tcol_ind + col_span,
                        column_header=trow_ind == 0,
                        row_header=False,
                    )
                    tcells.append(icell)

            num_rows = len(result_table)
            num_cols = len(result_table[0])
            self.in_table = False
            self.md_table_buffer = []
            table_data = TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=tcells
            )
            if len(tcells) > 0:
                doc.add_table(data=table_data)
        return

    @staticmethod
    def _cell_plain_text(cell: _gfm_el.TableCell) -> str:
        """Return the plain-text content of a GFM `TableCell`.

        Unlike `_inline_text`, this method treats `CodeSpan` nodes as literal
        text (no HTML unescaping) and HTML-unescapes only `RawText` and
        `Literal` nodes.

        Args:
            cell: A parsed GFM `TableCell` element.

        Returns:
            The plain-text string for the cell, without markup markers.
        """

        def _node_text(node) -> str:
            if isinstance(node, marko.inline.CodeSpan):
                return str(node.children)
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return unescape(children)
            return "".join(_node_text(c) for c in children or [])

        return "".join(_node_text(child) for child in cell.children)

    @staticmethod
    def _cell_has_rich_content(cell: _gfm_el.TableCell) -> bool:
        """Return True when a GFM TableCell contains inline formatting.

        A cell is considered rich when it contains any inline element that
        carries formatting (bold, italic, code span, link, image, or
        strikethrough). Plain cells that hold only `RawText` or `Literal`
        nodes are not rich.

        Args:
            cell: A parsed GFM `TableCell` element whose `children` contain
                marko inline nodes.

        Returns:
            `True` when rich inline content is detected, `False` otherwise.
        """
        rich_inline_types = (
            marko.inline.StrongEmphasis,
            marko.inline.Emphasis,
            marko.inline.CodeSpan,
            marko.inline.Link,
            marko.inline.AutoLink,
            marko.inline.Image,
        )
        if hasattr(_gfm_el, "Strikethrough"):
            rich_inline_types = (*rich_inline_types, _gfm_el.Strikethrough)

        def _check(node) -> bool:
            if isinstance(node, rich_inline_types):
                return True
            children = getattr(node, "children", None)
            if isinstance(children, list):
                return any(_check(c) for c in children)
            return False

        return any(_check(child) for child in cell.children)

    def _parse_gfm_table(
        self,
        table: _gfm_el.Table,
        doc: DoclingDocument,
        parent_item: NodeItem | None,
    ) -> None:
        """Convert a parsed GFM `Table` AST node into a Docling table.

        Each cell whose content is plain text is stored as a `TableCell`.
        Cells that contain inline formatting (bold, italic, code spans, links,
        …) are stored as `RichTableCell` objects. The cell's inline children
        are collected in an `InlineGroup` so that the markdown serializer
        renders them correctly on a single line.

        Args:
            table: The GFM `Table` element produced by the marko GFM extension.
            doc: The `DoclingDocument` being populated.
            parent_item: The docling node that will be the parent of the
                resulting `TableItem`.
        """
        rows = table.children
        num_rows = len(rows)
        if num_rows == 0:
            return
        num_cols = table.num_of_cols

        table_data = TableData(num_rows=num_rows, num_cols=num_cols)
        docling_table: TableItem = doc.add_table(data=table_data, parent=parent_item)

        for row_idx, row in enumerate(rows):
            is_header_row = row_idx == 0  # first row is the header in GFM
            cells = row.children
            for col_idx, cell in enumerate(cells):
                cell_text = MarkdownDocumentBackend._cell_plain_text(cell).strip()

                if MarkdownDocumentBackend._cell_has_rich_content(cell):
                    inline_group = doc.add_inline_group(parent=docling_table)
                    inline_refs: list[RefItem] = []
                    for child in cell.children:
                        self._iterate_cell_inline(
                            element=child,
                            doc=doc,
                            inline_parent=inline_group,
                            collected_refs=inline_refs,
                            formatting=None,
                            hyperlink=None,
                        )

                    if inline_refs:
                        group_name = (
                            f"rich_cell_group_{len(doc.tables) - 1}_{col_idx}_{row_idx}"
                        )
                        # RichTableCell.ref must point to a group whose parent
                        # is the table; wrap the InlineGroup accordingly.
                        cell_group = doc.add_group(
                            label=GroupLabel.UNSPECIFIED,
                            name=group_name,
                            parent=docling_table,
                        )
                        cell_group.children.append(inline_group.get_ref())
                        docling_table.children.remove(inline_group.get_ref())
                        inline_group.parent = cell_group.get_ref()

                        doc.add_table_cell(
                            table_item=docling_table,
                            cell=RichTableCell(
                                text=cell_text,
                                row_span=1,
                                col_span=1,
                                start_row_offset_idx=row_idx,
                                end_row_offset_idx=row_idx + 1,
                                start_col_offset_idx=col_idx,
                                end_col_offset_idx=col_idx + 1,
                                column_header=is_header_row,
                                row_header=False,
                                ref=cell_group.get_ref(),
                            ),
                        )
                    else:
                        # No items created; discard the empty InlineGroup.
                        if inline_group.get_ref() in docling_table.children:
                            docling_table.children.remove(inline_group.get_ref())
                        doc.groups.remove(inline_group)
                        doc.add_table_cell(
                            table_item=docling_table,
                            cell=TableCell(
                                text=cell_text,
                                row_span=1,
                                col_span=1,
                                start_row_offset_idx=row_idx,
                                end_row_offset_idx=row_idx + 1,
                                start_col_offset_idx=col_idx,
                                end_col_offset_idx=col_idx + 1,
                                column_header=is_header_row,
                                row_header=False,
                            ),
                        )
                else:
                    doc.add_table_cell(
                        table_item=docling_table,
                        cell=TableCell(
                            text=cell_text,
                            row_span=1,
                            col_span=1,
                            start_row_offset_idx=row_idx,
                            end_row_offset_idx=row_idx + 1,
                            start_col_offset_idx=col_idx,
                            end_col_offset_idx=col_idx + 1,
                            column_header=is_header_row,
                            row_header=False,
                        ),
                    )

    def _iterate_cell_inline(
        self,
        *,
        element,
        doc: DoclingDocument,
        inline_parent: NodeItem,
        collected_refs: list[RefItem],
        formatting: Formatting | None,
        hyperlink: AnyUrl | Path | None,
    ) -> None:
        """Process a single inline element from a GFM table cell.

        Recursively walks the inline AST of a table cell and creates the
        appropriate Docling text or code items as children of `inline_parent`,
        collecting their references in `collected_refs`. Only inline content
        permitted by the GFM specification inside a table cell is processed;
        block-level nodes are ignored.

        Args:
            element: A marko inline AST node.
            doc: The `DoclingDocument` being populated.
            inline_parent: The `InlineGroup` (or other `NodeItem`) to use as
                the parent for newly created text and code items.
            collected_refs: Accumulator for the `RefItem` values of every
                top-level doc item created while processing this cell.
            formatting: Inherited `Formatting` from ancestor inline nodes
                (e.g. bold wrapping a link).
            hyperlink: Inherited hyperlink URL from an ancestor `Link` node.
        """
        if isinstance(element, marko.inline.StrongEmphasis):
            formatting = MarkdownDocumentBackend._apply_formatting(
                formatting, bold=True
            )
            for child in element.children:
                self._iterate_cell_inline(
                    element=child,
                    doc=doc,
                    inline_parent=inline_parent,
                    collected_refs=collected_refs,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

        elif isinstance(element, marko.inline.Emphasis):
            formatting = MarkdownDocumentBackend._apply_formatting(
                formatting, italic=True
            )
            for child in element.children:
                self._iterate_cell_inline(
                    element=child,
                    doc=doc,
                    inline_parent=inline_parent,
                    collected_refs=collected_refs,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

        elif isinstance(element, marko.inline.Link | marko.inline.AutoLink):
            # AutoLink covers <https://url> and the GFM bare-URL form (gfm_el.Url
            # subclasses AutoLink).
            link_hyperlink = MarkdownDocumentBackend._resolve_link_dest(element.dest)
            for child in element.children:
                self._iterate_cell_inline(
                    element=child,
                    doc=doc,
                    inline_parent=inline_parent,
                    collected_refs=collected_refs,
                    formatting=formatting,
                    hyperlink=link_hyperlink,
                )

        elif isinstance(element, marko.inline.CodeSpan):
            snippet = str(element.children).strip()
            if snippet:
                item = doc.add_code(
                    parent=inline_parent,
                    text=snippet,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                collected_refs.append(item.get_ref())

        elif isinstance(element, marko.inline.RawText | marko.inline.Literal):
            raw = element.children if isinstance(element.children, str) else ""
            text = unescape(raw.strip())
            if text:
                item = doc.add_text(
                    label=DocItemLabel.TEXT,
                    parent=inline_parent,
                    text=text,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                collected_refs.append(item.get_ref())

        elif hasattr(_gfm_el, "Strikethrough") and isinstance(
            element, _gfm_el.Strikethrough
        ):
            formatting = MarkdownDocumentBackend._apply_formatting(
                formatting, strikethrough=True
            )
            for child in element.children:
                self._iterate_cell_inline(
                    element=child,
                    doc=doc,
                    inline_parent=inline_parent,
                    collected_refs=collected_refs,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

        elif isinstance(element, marko.inline.Image):
            image_ref = self._load_image_ref(element.dest)
            fig_caption = None
            if element.title:
                title = unescape(element.title)
                fig_caption = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=title,
                )
            pic = doc.add_picture(
                parent=inline_parent,
                image=image_ref,
                caption=fig_caption,
            )
            collected_refs.append(pic.get_ref())

        else:
            for child in getattr(element, "children", []):
                if not isinstance(child, str):
                    self._iterate_cell_inline(
                        element=child,
                        doc=doc,
                        inline_parent=inline_parent,
                        collected_refs=collected_refs,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )

    def _create_list_item(
        self,
        doc: DoclingDocument,
        parent_item: NodeItem | None,
        text: str,
        enumerated: bool,
        marker: str = "",
        formatting: Formatting | None = None,
        hyperlink: AnyUrl | Path | None = None,
    ):
        item = doc.add_list_item(
            text=text,
            enumerated=enumerated,
            marker=marker,
            parent=parent_item,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return item

    def _create_heading_item(
        self,
        doc: DoclingDocument,
        parent_item: NodeItem | None,
        text: str,
        level: int,
        formatting: Formatting | None = None,
        hyperlink: AnyUrl | Path | None = None,
    ):
        if level == 1:
            item = doc.add_title(
                text=text,
                parent=parent_item,
                formatting=formatting,
                hyperlink=hyperlink,
            )
        else:
            item = doc.add_heading(
                text=text,
                level=level - 1,
                parent=parent_item,
                formatting=formatting,
                hyperlink=hyperlink,
            )
        return item

    def _flush_creation_stack(
        self,
        *,
        doc: DoclingDocument,
        creation_stack: list[_CreationPayload],
        snippet_text: str,
        parent_item: NodeItem | None,
        list_ordered_flag_by_ref: dict[str, bool],
        list_start_by_ref: dict[str, int],
        list_item_counter_by_ref: dict[str, int],
        list_last_item_by_ref: dict[str, ListItem],
        formatting: Formatting | None,
        hyperlink: AnyUrl | Path | None,
    ) -> NodeItem | None:
        """Lazily create list items / headings when we first see their inline content.

        Important: Marko list items/headings can contain inline nodes that are NOT RawText
        (e.g. CodeSpan, Link). If we only flush on RawText, pending payloads can leak to
        later nodes and attach to a wrong parent, producing a very deep tree.
        """
        while len(creation_stack) > 0:
            to_create = creation_stack.pop()
            if isinstance(to_create, _ListItemCreationPayload):
                parent_ref = parent_item.self_ref if parent_item else None
                parent_item = self._create_list_item(
                    doc=doc,
                    parent_item=parent_item,
                    text=snippet_text,
                    enumerated=to_create.enumerated,
                    marker=to_create.marker,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                if parent_ref:
                    list_last_item_by_ref[parent_ref] = cast(ListItem, parent_item)
                    list_item_counter_by_ref[parent_ref] = (
                        list_item_counter_by_ref.get(parent_ref, 0) + 1
                    )

            elif isinstance(to_create, _HeadingCreationPayload):
                # Not keeping as parent_item as logic for correctly tracking
                # that not implemented yet (section components not captured
                # as heading children in marko)
                self._create_heading_item(
                    doc=doc,
                    parent_item=parent_item,
                    text=snippet_text,
                    level=to_create.level,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

        return parent_item

    def _iterate_elements(  # noqa: C901
        self,
        *,
        element: marko.element.Element,
        depth: int,
        doc: DoclingDocument,
        visited: set[marko.element.Element],
        creation_stack: list[
            _CreationPayload
        ],  # stack for lazy item creation triggered deep in marko's AST (on RawText)
        list_ordered_flag_by_ref: dict[str, bool],
        list_start_by_ref: dict[str, int],
        list_item_counter_by_ref: dict[str, int],
        list_last_item_by_ref: dict[str, ListItem],
        parent_item: NodeItem | None = None,
        formatting: Formatting | None = None,
        hyperlink: AnyUrl | Path | None = None,
    ):
        if element in visited:
            return

        # A line break only joins runs of the same paragraph. When no text run
        # follows the break inside it (for example, the paragraph ends in inline
        # HTML or a code span, or the next lines are table rows), the pending
        # flag would otherwise join the first run of a later block onto the
        # last text item.
        if isinstance(element, marko.block.BlockElement):
            self._pending_hard_line_break = False
            self._pending_soft_line_break = False

        # Iterates over all elements in the AST
        # Check for different element types and process relevant details
        if (
            isinstance(element, marko.block.Heading)
            or isinstance(element, marko.block.SetextHeading)
        ) and len(element.children) > 0:
            self._close_table(doc)
            _log.debug(
                " - Heading level %s, content: %s",
                element.level,
                element.children[0].children,  # type: ignore
            )

            if len(element.children) > 1:  # inline group will be created further down
                parent_item = self._create_heading_item(
                    doc=doc,
                    parent_item=parent_item,
                    text="",
                    level=element.level,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
            else:
                creation_stack.append(_HeadingCreationPayload(level=element.level))

        elif isinstance(element, marko.block.List):
            has_non_empty_list_items = False
            for child in element.children:
                if isinstance(child, marko.block.ListItem) and len(child.children) > 0:
                    has_non_empty_list_items = True
                    break

            self._close_table(doc)
            _log.debug(" - List %s", "ordered" if element.ordered else "unordered")
            if has_non_empty_list_items:
                parent_item = doc.add_list_group(name="list", parent=parent_item)
                list_ordered_flag_by_ref[parent_item.self_ref] = element.ordered
                if element.ordered:
                    list_start_by_ref[parent_item.self_ref] = element.start

        elif (
            isinstance(element, marko.block.ListItem)
            and len(element.children) > 0
            and isinstance((child := element.children[0]), marko.block.Paragraph)
            and len(child.children) > 0
        ):
            self._close_table(doc)
            _log.debug(" - List item")

            enumerated = (
                list_ordered_flag_by_ref.get(parent_item.self_ref, False)
                if parent_item
                else False
            )
            parent_ref: str | None = parent_item.self_ref if parent_item else None
            marker = ""
            if enumerated and parent_ref is not None:
                start = list_start_by_ref.get(parent_ref, 1)
                count = list_item_counter_by_ref.get(parent_ref, 0)
                marker = f"{start + count}."
            non_list_children: list[marko.element.Element] = [
                item
                for item in child.children
                if not isinstance(item, marko.block.ListItem)
            ]
            # Skip the inline_group path for items whose children are plain text
            # runs separated by line breaks; the pending-line-break merge paths
            # will produce a single list_item with the correct joined text.
            if len(non_list_children) > 1 and not _only_plain_line_breaks(
                non_list_children
            ):  # inline group will be created further down
                parent_ref: str | None = parent_item.self_ref if parent_item else None
                parent_item = self._create_list_item(
                    doc=doc,
                    parent_item=parent_item,
                    text="",
                    enumerated=enumerated,
                    marker=marker,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                if parent_ref:
                    list_last_item_by_ref[parent_ref] = cast(ListItem, parent_item)
                    list_item_counter_by_ref[parent_ref] = (
                        list_item_counter_by_ref.get(parent_ref, 0) + 1
                    )
            else:
                creation_stack.append(
                    _ListItemCreationPayload(enumerated=enumerated, marker=marker)
                )

        elif isinstance(element, marko.inline.Image):
            self._close_table(doc)
            _log.debug(" - Image with alt: %s, url: %s", element.title, element.dest)

            fig_caption: TextItem | None = None
            if element.title is not None and element.title != "":
                title = unescape(element.title)
                fig_caption = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=title,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

            image_ref = self._load_image_ref(element.dest)
            doc.add_picture(parent=parent_item, image=image_ref, caption=fig_caption)

        elif isinstance(element, marko.inline.Emphasis):
            _log.debug(" - Emphasis: %s", element.children)
            formatting = MarkdownDocumentBackend._apply_formatting(
                formatting, italic=True
            )

        elif isinstance(element, marko.inline.StrongEmphasis):
            _log.debug(" - StrongEmphasis: %s", element.children)
            formatting = MarkdownDocumentBackend._apply_formatting(
                formatting, bold=True
            )

        elif isinstance(element, marko.inline.Link):
            _log.debug(" - Link: %s", element.children)
            hyperlink = MarkdownDocumentBackend._resolve_link_dest(element.dest)

        elif isinstance(element, marko.inline.RawText | marko.inline.Literal):
            _log.debug(" - RawText/Literal: %s", element.children)
            original_text = (
                element.children if isinstance(element.children, str) else ""
            )
            snippet_text = unescape(original_text.strip())
            # A Literal is a backslash escape, so a pipe it holds is content
            # and not markup: it cannot open a table of its own.
            is_escape = isinstance(element, marko.inline.Literal)
            is_table_row = bool(snippet_text) and (
                # A header cell in bold or a link arrives as its own node with
                # no pipe in it, so once the paragraph is known to be a table,
                # every piece of it belongs to that table, pipe or not.
                self.in_pipeless_table
                or (
                    not is_escape
                    and "|" in snippet_text
                    and (self.in_table or original_text.lstrip().startswith("|"))
                )
            )
            if is_table_row:
                self.in_table = True
            if self.in_table and original_text:
                # Whitespace is kept: a cell can be several nodes ("run ", a
                # code span, " now"), and _split_table_row strips the cell once.
                cell_text = self._unescape_except_pipe(original_text)
                if is_escape:
                    cell_text = self._escape_pipes(cell_text)
                self._append_table_text(cell_text)
            elif snippet_text:
                self._close_table(doc)

                if creation_stack:
                    parent_item = self._flush_creation_stack(
                        doc=doc,
                        creation_stack=creation_stack,
                        snippet_text=snippet_text,
                        parent_item=parent_item,
                        list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                        list_start_by_ref=list_start_by_ref,
                        list_item_counter_by_ref=list_item_counter_by_ref,
                        list_last_item_by_ref=list_last_item_by_ref,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )
                    self._pending_hard_line_break = False
                    self._pending_soft_line_break = False
                else:
                    # A code span is its own item: text after a break never
                    # joins it (that would type prose as code).
                    last_is_code = bool(doc.texts) and isinstance(
                        doc.texts[-1], CodeItem
                    )
                    if (
                        self._pending_hard_line_break
                        and doc.texts
                        and not last_is_code
                        and doc.texts[-1].formatting == formatting
                        and doc.texts[-1].hyperlink == hyperlink
                    ):
                        doc.texts[-1].text += "\n" + snippet_text
                        doc.texts[-1].orig += "\n" + snippet_text
                    elif (
                        self._pending_soft_line_break
                        and doc.texts
                        and not last_is_code
                        and doc.texts[-1].formatting == formatting
                        and doc.texts[-1].hyperlink == hyperlink
                    ):
                        doc.texts[-1].text += " " + snippet_text
                        doc.texts[-1].orig += " " + snippet_text
                    else:
                        prefix = "\n" if self._pending_hard_line_break else ""
                        doc.add_text(
                            label=DocItemLabel.TEXT,
                            parent=parent_item,
                            text=prefix + snippet_text,
                            formatting=formatting,
                            hyperlink=hyperlink,
                        )
                    self._pending_hard_line_break = False
                    self._pending_soft_line_break = False

        elif isinstance(element, marko.inline.CodeSpan):
            _log.debug(" - Code Span: %s", element.children)
            snippet_text = str(element.children)
            if self.in_table:
                # A CodeSpan does not delimit cells; keep its content in the
                # current buffer slot. Its text is literal, so it is encoded for
                # the one decode in _close_table, and a pipe in it is not a
                # column separator.
                self._append_table_text(self._encode_table_literal(snippet_text))
            else:
                self._close_table(doc)
                snippet_text = snippet_text.strip()
                # If this CodeSpan is the only content of a list item / heading, Marko won't
                # emit RawText. Flush pending creations here to avoid leaking payloads.
                if creation_stack and snippet_text:
                    parent_item = self._flush_creation_stack(
                        doc=doc,
                        creation_stack=creation_stack,
                        snippet_text=snippet_text,
                        parent_item=parent_item,
                        list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                        list_start_by_ref=list_start_by_ref,
                        list_item_counter_by_ref=list_item_counter_by_ref,
                        list_last_item_by_ref=list_last_item_by_ref,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )
                    # Represent CodeSpan as the container's text; avoid adding a duplicate CodeItem.
                    return
                doc.add_code(
                    parent=parent_item,
                    text=snippet_text,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                # The code span consumed the break that preceded it, like a
                # text run does.
                self._pending_hard_line_break = False
                self._pending_soft_line_break = False

        elif (
            isinstance(element, marko.block.CodeBlock | marko.block.FencedCode)
            and len(element.children) > 0
            and isinstance((child := element.children[0]), marko.inline.RawText)
            # Drop blank lines around the code but keep the first line's
            # indentation, which is part of the code.
            and len(
                snippet_text := re.sub(
                    r"\A(?:[ \t]*\r?\n)+", "", child.children
                ).rstrip()
            )
            > 0
        ):
            self._close_table(doc)
            _log.debug(" - Code Block: %s", element.children)
            doc.add_code(
                parent=parent_item,
                text=snippet_text,
                code_language=detect_code_language(snippet_text, hint=element.lang),
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif isinstance(element, marko.inline.LineBreak):
            if self.in_table:
                _log.debug("Line break in a table")
                self.md_table_buffer.append("")
            elif element.soft:
                _log.debug("Soft line break")
                self._pending_soft_line_break = True
            else:
                _log.debug("Hard line break")
                self._pending_hard_line_break = True

        elif isinstance(element, marko.block.HTMLBlock):
            self._html_blocks += 1
            self._close_table(doc)
            _log.debug("HTML Block: %s", element)
            if (
                len(element.body) > 0
            ):  # If Marko doesn't return any content for HTML block, skip it
                html_block = element.body.strip()

                # wrap in markers to enable post-processing in convert()
                text_to_add = f"{_START_MARKER}{html_block}{_STOP_MARKER}"
                doc.add_code(
                    parent=parent_item,
                    text=text_to_add,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

        elif isinstance(element, _gfm_el.Table):
            self._close_table(doc)
            _log.debug(" - GFM Table")
            self._parse_gfm_table(
                table=element,
                doc=doc,
                parent_item=parent_item,
            )
            # Mark visited to skip the default child-iteration loop below.
            visited.add(element)
            return

        else:
            if not isinstance(element, str):
                self._close_table(doc)
                _log.debug("Some other element: %s", type(element).__name__)

        if isinstance(element, marko.block.Paragraph):
            # Set before descending: the RawText branch below reads this to let a
            # header without a leading pipe open a table. _close_table clears it.
            self.in_pipeless_table = MarkdownDocumentBackend._starts_pipeless_table(
                element
            )

        element_children = getattr(element, "children", [])
        if (
            isinstance(element, marko.block.Paragraph | marko.block.Heading)
            and len(element_children) > 1
            and not _only_plain_line_breaks(element_children)
        ):
            parent_item = doc.add_inline_group(parent=parent_item)

        processed_block_types = (
            marko.block.CodeBlock,
            marko.block.FencedCode,
            marko.inline.RawText,
        )

        if hasattr(element, "children") and not isinstance(
            element, processed_block_types
        ):
            for child in element.children:
                if (
                    isinstance(element, marko.block.ListItem)
                    and isinstance(child, marko.block.List)
                    and parent_item
                    and list_last_item_by_ref.get(parent_item.self_ref, None)
                ):
                    _log.debug(
                        "walking into new List hanging from item of parent list %s",
                        parent_item.self_ref,
                    )
                    parent_item = list_last_item_by_ref[parent_item.self_ref]

                self._iterate_elements(
                    element=child,
                    depth=depth + 1,
                    doc=doc,
                    visited=visited,
                    creation_stack=creation_stack,
                    list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                    list_start_by_ref=list_start_by_ref,
                    list_item_counter_by_ref=list_item_counter_by_ref,
                    list_last_item_by_ref=list_last_item_by_ref,
                    parent_item=parent_item,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

    def _get_image_loader(self) -> ImageResourceLoader:
        """Lazily build the shared image-resource loader.

        Resolving and decoding image sources (``data:`` URIs, local files, and
        remote URLs) together with the relevant safety limits is shared with the
        HTML backend through :class:`ImageResourceLoader`, so that logic is not
        duplicated here.
        """
        if self._image_loader is None:
            md_options = cast(MarkdownBackendOptions, self.options)
            self._image_loader = ImageResourceLoader(
                enable_local_fetch=md_options.enable_local_fetch,
                enable_remote_fetch=md_options.enable_remote_fetch,
                max_image_data_base64_bytes=md_options.max_image_data_base64_bytes,
            )
        return self._image_loader

    def _load_image_ref(self, dest: str) -> ImageRef | None:
        """Resolve and decode a Markdown image source into an ``ImageRef``.

        Returns ``None`` when image loading is disabled, the source is empty, or
        the image cannot be loaded.
        """
        md_options = cast(MarkdownBackendOptions, self.options)
        if not md_options.fetch_images or not dest:
            return None
        base_path = (
            str(md_options.source_uri) if md_options.source_uri is not None else None
        )
        return self._get_image_loader().load_image_ref(dest, base_path)

    def is_valid(self) -> bool:
        return self.valid

    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.MD}

    def convert(self) -> DoclingDocument:
        _log.debug("converting Markdown...")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/markdown",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        if self.is_valid():
            # The GFM extension gives tables a structured AST with inline
            # children per cell, enabling RichTableCell for formatted content.
            marko_parser = Markdown(extensions=["gfm"])
            parsed_ast = marko_parser.parse(self.markdown)
            self._iterate_elements(
                element=parsed_ast,
                depth=0,
                doc=doc,
                parent_item=None,
                visited=set(),
                creation_stack=[],
                list_ordered_flag_by_ref={},
                list_start_by_ref={},
                list_item_counter_by_ref={},
                list_last_item_by_ref={},
            )
            self._close_table(doc=doc)  # flush any trailing text-buffer table

            if self._html_blocks > 0:
                html_backend_cls = HTMLDocumentBackend
                html_str = doc.export_to_html()

                def _restore_original_html(txt, regex):
                    _txt, count = re.subn(regex, "", txt)
                    if count != self._html_blocks:
                        raise RuntimeError(
                            "An internal error has occurred during Markdown conversion."
                        )
                    return _txt

                # restore original HTML by removing previously added markers
                for regex in [
                    rf"<pre>\s*<code>\s*{_START_MARKER}",
                    rf"{_STOP_MARKER}\s*</code>\s*</pre>",
                ]:
                    html_str = _restore_original_html(txt=html_str, regex=regex)
                self._html_blocks = 0
                # delegate to HTML backend
                stream = BytesIO(bytes(html_str, encoding="utf-8"))
                md_options = cast(MarkdownBackendOptions, self.options)
                html_options = HTMLBackendOptions(
                    enable_local_fetch=md_options.enable_local_fetch,
                    enable_remote_fetch=md_options.enable_remote_fetch,
                    fetch_images=md_options.fetch_images,
                    source_uri=md_options.source_uri,
                    infer_furniture=False,
                    add_title=False,
                )
                in_doc = InputDocument(
                    path_or_stream=stream,
                    format=InputFormat.HTML,
                    backend=html_backend_cls,
                    filename=self.file.name,
                    backend_options=html_options,
                )
                html_backend_obj = html_backend_cls(
                    in_doc=in_doc,
                    path_or_stream=stream,
                    options=html_options,
                )
                doc = html_backend_obj.convert()
        else:
            raise RuntimeError(
                f"Cannot convert md with {self.document_hash} because the backend failed to init."
            )
        return doc
