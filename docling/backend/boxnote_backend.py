import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    GroupLabel,
    NodeItem,
    RichTableCell,
    TableCell,
    TableData,
)
from pydantic import AnyUrl, TypeAdapter, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_HYPERLINK = TypeAdapter(AnyUrl)

# Box Note link marks carry web URLs, so only these schemes become hyperlinks. A
# stray "javascript:" or a bare string is dropped rather than mis-read as a link.
_SAFE_LINK_SCHEMES = frozenset({"http", "https", "mailto"})

# (text, inline formatting, hyperlink) for one styled span of a block.
_Run = tuple[str, Formatting | None, AnyUrl | None]


class BoxNoteDocumentBackend(DeclarativeDocumentBackend):
    """Declarative backend for Box Notes (.boxnote files).

    Box Notes are JSON. This reads the current (post-August 2022) schema, a
    ProseMirror-style "doc" node tree, and maps it onto a DoclingDocument.
    Notes saved before that release use the older "atext"/"pool" model and are
    reported as unsupported rather than silently mis-parsed.
    """

    @override
    def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path):
        super().__init__(in_doc, path_or_stream)

        self.data: dict[str, Any] = {}
        try:
            raw = ""
            if isinstance(self.path_or_stream, BytesIO):
                raw = self.path_or_stream.getvalue().decode("utf-8")
            elif isinstance(self.path_or_stream, Path):
                raw = self.path_or_stream.read_text(encoding="utf-8")
            if raw.strip():
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    self.data = loaded
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            raise DocumentLoadError(
                f"Could not load Box Note document with hash {self.document_hash}."
            ) from e

        if "atext" in self.data and not self.is_valid():
            raise DocumentLoadError(
                "Legacy Box Notes (the pre-August-2022 atext/pool format) are not "
                "supported yet; only the current Box Note format can be converted."
            )

    @override
    def is_valid(self) -> bool:
        return isinstance(self.data.get("doc"), dict)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.BOXNOTE}

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert Box Note with hash {self.document_hash}: "
                "no 'doc' node found."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file.boxnote",
            mimetype="application/vnd.box.boxnote",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        self._add_blocks(self.data["doc"].get("content", []), doc, None)
        return doc

    def _add_blocks(
        self, nodes: list[dict], doc: DoclingDocument, parent: NodeItem | None
    ) -> None:
        for node in nodes:
            self._add_block(node, doc, parent)

    def _add_block(
        self, node: dict[str, Any], doc: DoclingDocument, parent: NodeItem | None
    ) -> None:
        node_type = node.get("type")
        content = node.get("content", [])

        if node_type == "heading":
            text, formatting, hyperlink = self._collapse(content)
            if text:
                level = node.get("attrs", {}).get("level") or 1
                if level <= 1:
                    doc.add_title(
                        text=text,
                        parent=parent,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )
                else:
                    doc.add_heading(
                        text=text,
                        level=level - 1,
                        parent=parent,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )
        elif node_type == "paragraph":
            self._add_paragraph(content, doc, parent)
        elif node_type in ("bullet_list", "ordered_list", "check_list"):
            self._add_list(node_type, content, doc, parent)
        elif node_type == "code_block":
            code = self._plain_text(content)
            if code:
                doc.add_code(text=code, parent=parent)
        elif node_type == "table":
            self._add_table(content, doc, parent)
        elif node_type == "image":
            self._add_image(node.get("attrs", {}), doc, parent)
        elif content:
            # blockquote, call_out_box and other wrappers have no dedicated item;
            # keep their inner blocks rather than dropping the text.
            self._add_blocks(content, doc, parent)

    def _add_paragraph(
        self, content: list[dict], doc: DoclingDocument, parent: NodeItem | None
    ) -> None:
        runs = self._runs(content)
        if not runs:
            return
        if len(runs) == 1:
            text, formatting, hyperlink = runs[0]
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                parent=parent,
                formatting=formatting,
                hyperlink=hyperlink,
            )
            return
        group = doc.add_inline_group(parent=parent)
        for text, formatting, hyperlink in runs:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                parent=group,
                formatting=formatting,
                hyperlink=hyperlink,
            )

    def _add_list(
        self,
        list_type: str,
        items: list[dict],
        doc: DoclingDocument,
        parent: NodeItem | None,
    ) -> None:
        enumerated = list_type == "ordered_list"
        group = doc.add_group(
            label=GroupLabel.ORDERED_LIST if enumerated else GroupLabel.LIST,
            name="list",
            parent=parent,
        )
        for item in items:
            if item.get("type") == "check_list_item":
                self._add_check_item(item, doc, group)
            else:
                self._add_list_item(item, doc, group, enumerated)

    def _add_list_item(
        self,
        item: dict[str, Any],
        doc: DoclingDocument,
        group: NodeItem,
        enumerated: bool,
    ) -> None:
        text, formatting, hyperlink, nested = self._split_item(item)
        list_item = doc.add_list_item(
            text=text,
            enumerated=enumerated,
            parent=group,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if nested:
            self._add_blocks(nested, doc, list_item)

    def _add_check_item(
        self, item: dict[str, Any], doc: DoclingDocument, group: NodeItem
    ) -> None:
        text, formatting, hyperlink, nested = self._split_item(item)
        label = (
            DocItemLabel.CHECKBOX_SELECTED
            if item.get("attrs", {}).get("checked")
            else DocItemLabel.CHECKBOX_UNSELECTED
        )
        check_item = doc.add_text(
            label=label,
            text=text,
            parent=group,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        if nested:
            self._add_blocks(nested, doc, check_item)

    def _split_item(
        self, item: dict[str, Any]
    ) -> tuple[str, Formatting | None, AnyUrl | None, list[dict]]:
        """Split a list item into its own text and any nested blocks.

        The leading paragraph becomes the item text; everything else (nested
        lists, extra paragraphs) is returned to be added as children.
        """
        text, formatting, hyperlink = "", None, None
        nested: list[dict] = []
        for child in item.get("content", []):
            if not text and child.get("type") == "paragraph":
                text, formatting, hyperlink = self._collapse(child.get("content", []))
            else:
                nested.append(child)
        return text, formatting, hyperlink, nested

    def _add_table(
        self, rows: list[dict], doc: DoclingDocument, parent: NodeItem | None
    ) -> None:
        rows = [row for row in rows if row.get("type") == "table_row"]
        if not rows:
            return

        data = TableData(num_rows=len(rows), num_cols=0, table_cells=[])
        table = doc.add_table(data=data, parent=parent)

        occupied: set[tuple[int, int]] = set()
        num_cols = 0
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.get("content", []):
                cell_type = cell.get("type")
                if cell_type not in ("table_cell", "table_header"):
                    continue
                while (row_idx, col_idx) in occupied:
                    col_idx += 1
                attrs = cell.get("attrs", {})
                row_span = attrs.get("rowspan") or 1
                col_span = attrs.get("colspan") or 1
                end_row = row_idx + row_span
                end_col = col_idx + col_span
                blocks = cell.get("content", [])
                is_header = cell_type == "table_header"

                if self._cell_is_rich(blocks):
                    group = doc.add_group(
                        label=GroupLabel.UNSPECIFIED, name="table_cell", parent=table
                    )
                    self._add_blocks(blocks, doc, group)
                    table_cell: TableCell = RichTableCell(
                        text=self._cell_text(blocks),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=end_row,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=end_col,
                        column_header=is_header,
                        ref=group.get_ref(),
                    )
                else:
                    table_cell = TableCell(
                        text=self._cell_text(blocks),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=end_row,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=end_col,
                        column_header=is_header,
                    )
                doc.add_table_cell(table_item=table, cell=table_cell)

                for span_row in range(row_idx, row_idx + row_span):
                    for span_col in range(col_idx, col_idx + col_span):
                        occupied.add((span_row, span_col))
                col_idx += col_span
                num_cols = max(num_cols, col_idx)

        table.data.num_cols = num_cols

    def _cell_is_rich(self, blocks: list[dict]) -> bool:
        """A cell is rich when it cannot be one plain-text TableCell.

        That covers images, multiple blocks, any non-paragraph block (lists,
        code, nested tables), and a lone paragraph carrying a link or formatting:
        a plain TableCell has no hyperlink field, so a flattened link would be
        lost.
        """
        meaningful = [
            block
            for block in blocks
            if block.get("type") != "paragraph" or self._runs(block.get("content", []))
        ]
        if len(meaningful) > 1:
            return True
        if any(block.get("type") != "paragraph" for block in meaningful):
            return True
        return any(
            formatting or hyperlink
            for block in meaningful
            for _, formatting, hyperlink in self._runs(block.get("content", []))
        )

    def _add_image(
        self, attrs: dict[str, Any], doc: DoclingDocument, parent: NodeItem | None
    ) -> None:
        caption = None
        label = attrs.get("alt") or attrs.get("fileName")
        if label:
            caption = doc.add_text(label=DocItemLabel.CAPTION, text=label)
        doc.add_picture(caption=caption, parent=parent)

    def _collapse(self, content: list[dict]) -> _Run:
        """Reduce inline content to a single run.

        Headings and list items take one formatting and one hyperlink, so a
        single styled span keeps its style and a mixed span falls back to plain
        joined text.
        """
        runs = self._runs(content)
        if len(runs) == 1:
            return runs[0]
        return "".join(text for text, _, _ in runs), None, None

    def _runs(self, content: list[dict]) -> list[_Run]:
        runs: list[_Run] = []
        for node in content or []:
            node_type = node.get("type")
            if node_type == "text":
                text = node.get("text", "")
                if text:
                    formatting, hyperlink = self._marks(node.get("marks", []))
                    runs.append((text, formatting, hyperlink))
            elif node_type == "hard_break":
                runs.append((" ", None, None))
        return runs

    def _marks(self, marks: list[dict]) -> tuple[Formatting | None, AnyUrl | None]:
        formatting: Formatting | None = None
        hyperlink: AnyUrl | None = None
        for mark in marks or []:
            mark_type = mark.get("type")
            if mark_type == "strong":
                formatting = formatting or Formatting()
                formatting.bold = True
            elif mark_type == "em":
                formatting = formatting or Formatting()
                formatting.italic = True
            elif mark_type == "underline":
                formatting = formatting or Formatting()
                formatting.underline = True
            elif mark_type == "strikethrough":
                formatting = formatting or Formatting()
                formatting.strikethrough = True
            elif mark_type == "link":
                href = mark.get("attrs", {}).get("href")
                if isinstance(href, str) and href:
                    hyperlink = self._as_url(href)
        return formatting, hyperlink

    def _plain_text(self, nodes: list[dict]) -> str:
        parts: list[str] = []
        for node in nodes or []:
            node_type = node.get("type")
            if node_type == "text":
                parts.append(node.get("text", ""))
            elif node_type == "hard_break":
                parts.append("\n")
            elif node.get("content"):
                parts.append(self._plain_text(node["content"]))
        return "".join(parts)

    def _cell_text(self, blocks: list[dict]) -> str:
        texts = (self._plain_text(block.get("content", [])).strip() for block in blocks)
        return " ".join(text for text in texts if text)

    @staticmethod
    def _as_url(href: str) -> AnyUrl | None:
        try:
            if urlparse(href).scheme not in _SAFE_LINK_SCHEMES:
                return None
            return _HYPERLINK.validate_python(href)
        except (ValueError, ValidationError):
            # urlparse raises on malformed input (e.g. unbalanced IPv6 brackets);
            # drop the link rather than failing the whole conversion.
            return None
