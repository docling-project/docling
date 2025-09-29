import logging
import re
import warnings
from copy import deepcopy
from enum import Enum
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union, cast

import marko
import marko.element
import marko.inline
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ListItem,
    NodeItem,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.document import Formatting
from marko import Markdown
from pydantic import AnyUrl, BaseModel, Field, TypeAdapter
from typing_extensions import Annotated

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

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


_CreationPayload = Annotated[
    Union[
        _HeadingCreationPayload,
        _ListItemCreationPayload,
    ],
    Field(discriminator="kind"),
]


class MarkdownDocumentBackend(DeclarativeDocumentBackend):
    def _shorten_underscore_sequences(self, markdown_text: str, max_length: int = 10):
        # This regex will match any sequence of underscores
        pattern = r"_+"

        def replace_match(match):
            underscore_sequence = match.group(
                0
            )  # Get the full match (sequence of underscores)

            # Shorten the sequence if it exceeds max_length
            if len(underscore_sequence) > max_length:
                return "_" * max_length
            else:
                return underscore_sequence  # Leave it unchanged if it is shorter or equal to max_length

        # Use re.sub to replace long underscore sequences
        shortened_text = re.sub(pattern, replace_match, markdown_text)

        if len(shortened_text) != len(markdown_text):
            warnings.warn("Detected potentially incorrect Markdown, correcting...")

        return shortened_text

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        _log.debug("Starting MarkdownDocumentBackend...")

        # Markdown file:
        self.path_or_stream = path_or_stream
        self.valid = True
        self.markdown = ""  # To store original Markdown string

        self.in_table = False
        self.md_table_buffer: list[str] = []
        self._html_blocks: int = 0

        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue().decode("utf-8")
                # remove invalid sequences
                # very long sequences of underscores will lead to unnecessary long processing times.
                # In any proper Markdown files, underscores have to be escaped,
                # otherwise they represent emphasis (bold or italic)
                self.markdown = self._shorten_underscore_sequences(text_stream)
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    md_content = f.read()
                    # remove invalid sequences
                    # very long sequences of underscores will lead to unnecessary long processing times.
                    # In any proper Markdown files, underscores have to be escaped,
                    # otherwise they represent emphasis (bold or italic)
                    self.markdown = self._shorten_underscore_sequences(md_content)
            self.valid = True

            _log.debug(self.markdown)
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize MD backend for file with hash {self.document_hash}."
            ) from e
        return

    def _close_table(self, doc: DoclingDocument):
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
                    header = [t.strip() for t in md_table_row.split("|")[1:-1]]
                    for value in header:
                        data.append(value)
                    result_table.append(data)
                if n > 1:
                    values = [t.strip() for t in md_table_row.split("|")[1:-1]]
                    for value in values:
                        data.append(value)
                    result_table.append(data)

            for trow_ind, trow in enumerate(result_table):
                for tcol_ind, cellval in enumerate(trow):
                    row_span = (
                        1  # currently supporting just simple tables (without spans)
                    )
                    col_span = (
                        1  # currently supporting just simple tables (without spans)
                    )
                    icell = TableCell(
                        text=cellval.strip(),
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
            self.md_table_buffer = []  # clean table markdown buffer
            # Initialize Docling TableData
            table_data = TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=tcells
            )
            # Populate
            for tcell in tcells:
                table_data.table_cells.append(tcell)
            if len(tcells) > 0:
                doc.add_table(data=table_data)
        return

    def _create_list_item(
        self,
        doc: DoclingDocument,
        parent_item: Optional[NodeItem],
        text: str,
        enumerated: bool,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        item = doc.add_list_item(
            text=text,
            enumerated=enumerated,
            parent=parent_item,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return item

    def _create_heading_item(
        self,
        doc: DoclingDocument,
        parent_item: Optional[NodeItem],
        text: str,
        level: int,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
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
        list_last_item_by_ref: dict[str, ListItem],
        parent_item: Optional[NodeItem] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
    ):
        if element in visited:
            return

        # Iterates over all elements in the AST
        # Check for different element types and process relevant details
        if isinstance(element, marko.block.Heading) and len(element.children) > 0:
            self._close_table(doc)
            _log.debug(
                f" - Heading level {element.level}, content: {element.children[0].children}"  # type: ignore
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
            _log.debug(f" - List {'ordered' if element.ordered else 'unordered'}")
            if has_non_empty_list_items:
                parent_item = doc.add_list_group(name="list", parent=parent_item)
                list_ordered_flag_by_ref[parent_item.self_ref] = element.ordered

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
            non_list_children: list[marko.element.Element] = [
                item
                for item in child.children
                if not isinstance(item, marko.block.ListItem)
            ]
            if len(non_list_children) > 1:  # inline group will be created further down
                parent_ref: Optional[str] = (
                    parent_item.self_ref if parent_item else None
                )
                parent_item = self._create_list_item(
                    doc=doc,
                    parent_item=parent_item,
                    text="",
                    enumerated=enumerated,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                if parent_ref:
                    list_last_item_by_ref[parent_ref] = cast(ListItem, parent_item)
            else:
                creation_stack.append(_ListItemCreationPayload(enumerated=enumerated))

        elif isinstance(element, marko.inline.Image):
            self._close_table(doc)
            _log.debug(f" - Image with alt: {element.title}, url: {element.dest}")

            fig_caption: Optional[TextItem] = None
            if element.title is not None and element.title != "":
                title = unescape(element.title)
                fig_caption = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=title,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

            doc.add_picture(parent=parent_item, caption=fig_caption)

        elif isinstance(element, marko.inline.Emphasis):
            _log.debug(f" - Emphasis: {element.children}")
            formatting = deepcopy(formatting) if formatting else Formatting()
            formatting.italic = True

        elif isinstance(element, marko.inline.StrongEmphasis):
            _log.debug(f" - StrongEmphasis: {element.children}")
            formatting = deepcopy(formatting) if formatting else Formatting()
            formatting.bold = True

        elif isinstance(element, marko.inline.Link):
            _log.debug(f" - Link: {element.children}")
            hyperlink = TypeAdapter(Optional[Union[AnyUrl, Path]]).validate_python(
                element.dest
            )

        elif isinstance(element, (marko.inline.RawText, marko.inline.Literal)):
            _log.debug(f" - RawText/Literal: {element.children}")
            snippet_text = (
                element.children.strip() if isinstance(element.children, str) else ""
            )
            snippet_text = unescape(snippet_text)
            # Detect start of the table:
            if "|" in snippet_text or self.in_table:
                # most likely part of the markdown table
                self.in_table = True
                if len(self.md_table_buffer) > 0:
                    self.md_table_buffer[len(self.md_table_buffer) - 1] += snippet_text
                else:
                    self.md_table_buffer.append(snippet_text)
            elif snippet_text:
                self._close_table(doc)

                if creation_stack:
                    while len(creation_stack) > 0:
                        to_create = creation_stack.pop()
                        if isinstance(to_create, _ListItemCreationPayload):
                            enumerated = (
                                list_ordered_flag_by_ref.get(
                                    parent_item.self_ref, False
                                )
                                if parent_item
                                else False
                            )
                            parent_ref = parent_item.self_ref if parent_item else None
                            parent_item = self._create_list_item(
                                doc=doc,
                                parent_item=parent_item,
                                text=snippet_text,
                                enumerated=enumerated,
                                formatting=formatting,
                                hyperlink=hyperlink,
                            )
                            if parent_ref:
                                list_last_item_by_ref[parent_ref] = cast(
                                    ListItem, parent_item
                                )

                        elif isinstance(to_create, _HeadingCreationPayload):
                            # not keeping as parent_item as logic for correctly tracking
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
                else:
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        parent=parent_item,
                        text=snippet_text,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )

        elif isinstance(element, marko.inline.CodeSpan):
            self._close_table(doc)
            _log.debug(f" - Code Span: {element.children}")
            snippet_text = str(element.children).strip()
            doc.add_code(
                parent=parent_item,
                text=snippet_text,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif (
            isinstance(element, (marko.block.CodeBlock, marko.block.FencedCode))
            and len(element.children) > 0
            and isinstance((child := element.children[0]), marko.inline.RawText)
            and len(snippet_text := (child.children.strip())) > 0
        ):
            self._close_table(doc)
            _log.debug(f" - Code Block: {element.children}")
            doc.add_code(
                parent=parent_item,
                text=snippet_text,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif isinstance(element, marko.inline.LineBreak):
            if self.in_table:
                _log.debug("Line break in a table")
                self.md_table_buffer.append("")

        elif isinstance(element, marko.block.HTMLBlock):
            self._html_blocks += 1
            self._close_table(doc)
            _log.debug(f"HTML Block: {element}")
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
        else:
            if not isinstance(element, str):
                self._close_table(doc)
                _log.debug(f"Some other element: {element}")

        if (
            isinstance(element, (marko.block.Paragraph, marko.block.Heading))
            and len(element.children) > 1
        ):
            parent_item = doc.add_inline_group(parent=parent_item)

        processed_block_types = (
            marko.block.CodeBlock,
            marko.block.FencedCode,
            marko.inline.RawText,
        )

        # Iterate through the element's children (if any)
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
                        f"walking into new List hanging from item of parent list {parent_item.self_ref}"
                    )
                    parent_item = list_last_item_by_ref[parent_item.self_ref]

                self._iterate_elements(
                    element=child,
                    depth=depth + 1,
                    doc=doc,
                    visited=visited,
                    creation_stack=creation_stack,
                    list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                    list_last_item_by_ref=list_last_item_by_ref,
                    parent_item=parent_item,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

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
            # Parse the markdown into an abstract syntax tree (AST)
            marko_parser = Markdown()
            parsed_ast = marko_parser.parse(self.markdown)
            # Start iterating from the root of the AST
            self._iterate_elements(
                element=parsed_ast,
                depth=0,
                doc=doc,
                parent_item=None,
                visited=set(),
                creation_stack=[],
                list_ordered_flag_by_ref={},
                list_last_item_by_ref={},
            )
            self._close_table(doc=doc)  # handle any last hanging table

            # if HTML blocks were detected, export to HTML and delegate to HTML backend
            if self._html_blocks > 0:
                # export to HTML
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
                in_doc = InputDocument(
                    path_or_stream=stream,
                    format=InputFormat.HTML,
                    backend=html_backend_cls,
                    filename=self.file.name,
                )
                html_backend_obj = html_backend_cls(
                    in_doc=in_doc, path_or_stream=stream
                )
                doc = html_backend_obj.convert()
        else:
            raise RuntimeError(
                f"Cannot convert md with {self.document_hash} because the backend failed to init."
            )
        return doc
