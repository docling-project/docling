import logging
import re
import traceback
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union, cast

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.document import ContentLayer
from pydantic import BaseModel
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128

# Tags that initiate distinct Docling items
_BLOCK_TAGS: Final = {
    "address",
    "details",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "pre",
    "code",
    "ul",
    "ol",
    "summary",
    "table",
}


class _Context(BaseModel):
    list_ordered_flag_by_ref: dict[str, bool] = {}
    list_start_by_ref: dict[str, int] = {}


class HTMLDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream)
        self.soup: Optional[Tag] = None
        self.path_or_stream = path_or_stream

        # Initialize the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        self.ctx = _Context()
        for i in range(self.max_levels):
            self.parents[i] = None

        try:
            raw = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else Path(path_or_stream).read_bytes()
            )
            self.soup = BeautifulSoup(raw, "html.parser")
        except Exception as e:
            raise RuntimeError(
                "Could not initialize HTML backend for file with "
                f"hash {self.document_hash}."
            ) from e

    @override
    def is_valid(self) -> bool:
        return self.soup is not None

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.HTML}

    @override
    def convert(self) -> DoclingDocument:
        _log.debug("Starting HTML conversion...")
        if not self.is_valid():
            raise RuntimeError("Invalid HTML document.")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/html",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        assert self.soup is not None
        # set the title as furniture, since it is part of the document metadata
        title = self.soup.title
        if title:
            doc.add_title(
                text=title.get_text(separator=" ", strip=True),
                content_layer=ContentLayer.FURNITURE,
            )
        # remove scripts/styles
        for tag in self.soup(["script", "style"]):
            tag.decompose()
        content = self.soup.body or self.soup
        # normalize <br> tags
        for br in content("br"):
            br.replace_with(NavigableString("\n"))
        # set default content layer
        headers = content.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        self.content_layer = (
            ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
        )
        # reset context
        self.ctx = _Context()

        try:
            self._walk(content, doc)
        except Exception:
            print(traceback.format_exc())

        return doc

    def _walk(self, element: Tag, doc: DoclingDocument) -> None:
        """Parse an XML tag by recursively walking its content.

        While walking, the method buffers inline text across tags like <b> or <span>,
        emitting text nodes only at block boundaries.

        Args:
            element: The XML tag to parse.
            doc: The Docling document to be updated with the parsed content.
        """
        buffer: list[str] = []

        def flush_buffer():
            if not buffer:
                return
            text = "".join(buffer).strip()
            buffer.clear()
            if not text:
                return
            for part in text.split("\n"):
                seg = part.strip()
                if seg:
                    doc.add_text(
                        DocItemLabel.TEXT,
                        seg,
                        parent=self.parents[self.level],
                        content_layer=self.content_layer,
                    )

        for node in element.contents:
            if isinstance(node, Tag):
                name = node.name.lower()
                if name == "img":
                    flush_buffer()
                    self._emit_image(node, doc)
                elif name in _BLOCK_TAGS:
                    flush_buffer()
                    self._handle_block(node, doc)
                elif node.find(_BLOCK_TAGS):
                    flush_buffer()
                    self._walk(node, doc)
                else:
                    buffer.append(node.text)
            elif isinstance(node, NavigableString) and not isinstance(
                node, PreformattedString
            ):
                buffer.append(str(node))

        flush_buffer()

    def _handle_heading(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()
        # set default content layer to BODY as soon as we encounter a heading
        self.content_layer = ContentLayer.BODY
        level = int(tag_name[1])
        text = tag.get_text(strip=True, separator=" ")
        # the first level is for the title item
        if level == 1:
            for key in self.parents.keys():
                self.parents[key] = None
            self.level = 0
            self.parents[self.level + 1] = doc.add_title(
                text, content_layer=self.content_layer
            )
        # the other levels need to be lowered by 1 if a title was set
        else:
            level -= 1
            if level > self.level:
                # add invisible group
                for i in range(self.level, level):
                    _log.debug(f"Adding invisible group to level {i}")
                    self.parents[i + 1] = doc.add_group(
                        name=f"header-{i + 1}",
                        label=GroupLabel.SECTION,
                        parent=self.parents[i],
                        content_layer=self.content_layer,
                    )
                self.level = level
            elif level < self.level:
                # remove the tail
                for key in self.parents.keys():
                    if key > level + 1:
                        _log.debug(f"Remove the tail of level {key}")
                        self.parents[key] = None
                self.level = level
            self.parents[self.level + 1] = doc.add_heading(
                parent=self.parents[self.level],
                text=text,
                level=self.level,
                content_layer=self.content_layer,
            )
        self.level += 1
        for img_tag in tag("img"):
            if isinstance(img_tag, Tag):
                self._emit_image(img_tag, doc)

    def _handle_list(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()
        start: Optional[int] = None
        name: str = ""
        is_ordered = tag_name == "ol"
        if is_ordered:
            start_attr = tag.get("start")
            if isinstance(start_attr, str) and start_attr.isnumeric():
                start = int(start_attr)
            name = "ordered list" + (f" start {start}" if start is not None else "")
        else:
            name = "list"
        # Create the list container
        list_group = doc.add_list_group(
            name=name,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )
        self.parents[self.level + 1] = list_group
        self.ctx.list_ordered_flag_by_ref[list_group.self_ref] = is_ordered
        if is_ordered and start is not None:
            self.ctx.list_start_by_ref[list_group.self_ref] = start
        self.level += 1

        # For each top-level <li> in this list
        for li in tag.find_all({"li", "ul", "ol"}, recursive=False):
            if not isinstance(li, Tag):
                continue

            # sub-list items should be indented under main list items, but temporarily
            # addressing invalid HTML (docling-core/issues/357)
            if li.name in {"ul", "ol"}:
                self._handle_block(li, doc)

            else:
                # 1) determine the marker
                if is_ordered and start is not None:
                    marker = f"{start + len(list_group.children)}."
                else:
                    marker = ""

                # 2) extract only the "direct" text from this <li>
                parts: list[str] = []
                for child in li.contents:
                    if isinstance(child, NavigableString) and not isinstance(
                        child, PreformattedString
                    ):
                        parts.append(child)
                    elif isinstance(child, Tag) and child.name not in ("ul", "ol"):
                        text_part = child.get_text()
                        if text_part:
                            parts.append(text_part)
                li_text = re.sub(r"\s+|\n+", " ", "".join(parts)).strip()

                # 3) add the list item
                if li_text:
                    self.parents[self.level + 1] = doc.add_list_item(
                        text=li_text,
                        enumerated=is_ordered,
                        marker=marker,
                        parent=list_group,
                        content_layer=self.content_layer,
                    )

                    # 4) recurse into any nested lists, attaching them to this <li> item
                    for sublist in li({"ul", "ol"}, recursive=False):
                        if isinstance(sublist, Tag):
                            self.level += 1
                            self._handle_block(sublist, doc)
                            self.parents[self.level + 1] = None
                            self.level -= 1
                else:
                    for sublist in li({"ul", "ol"}, recursive=False):
                        if isinstance(sublist, Tag):
                            self._handle_block(sublist, doc)

                # 5) extract any images under this <li>
                for img_tag in li("img"):
                    if isinstance(img_tag, Tag):
                        self._emit_image(img_tag, doc)

        self.parents[self.level + 1] = None
        self.level -= 1

    def _handle_block(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()

        if tag_name == "figure":
            img_tag = tag.find("img")
            if isinstance(img_tag, Tag):
                self._emit_image(img_tag, doc)

        elif tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._handle_heading(tag, doc)

        elif tag_name in {"ul", "ol"}:
            self._handle_list(tag, doc)

        elif tag_name in {"p", "address", "summary"}:
            for part in tag.text.split("\n"):
                seg = part.strip()
                if seg:
                    doc.add_text(
                        parent=self.parents[self.level],
                        label=DocItemLabel.TEXT,
                        text=seg,
                        content_layer=self.content_layer,
                    )
            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(img_tag, doc)

        elif tag_name == "table":
            data = HTMLDocumentBackend.parse_table_data(tag)
            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(tag, doc)
            if data is not None:
                doc.add_table(
                    data=data,
                    parent=self.parents[self.level],
                    content_layer=self.content_layer,
                )

        elif tag_name in {"pre", "code"}:
            # handle monospace code snippets (pre).
            text = tag.get_text(strip=True)
            if text:
                doc.add_code(
                    parent=self.parents[self.level],
                    text=text,
                    content_layer=self.content_layer,
                )

        elif tag_name == "details":
            # handle details and its content.
            self.parents[self.level + 1] = doc.add_group(
                name="details",
                label=GroupLabel.SECTION,
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )
            self.level += 1
            self._walk(tag, doc)
            self.parents[self.level + 1] = None
            self.level -= 1

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument) -> None:
        figure = img_tag.find_parent("figure")
        caption: str = ""
        if isinstance(figure, Tag):
            caption_tag = figure.find("figcaption", recursive=False)
            if isinstance(caption_tag, Tag):
                caption = caption_tag.get_text()
        if not caption:
            caption = str(img_tag.get("alt", "")).strip()

        caption_item: Optional[TextItem] = None
        if caption:
            caption_item = doc.add_text(
                DocItemLabel.CAPTION, text=caption, content_layer=self.content_layer
            )

        doc.add_picture(
            caption=caption_item,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )

    @staticmethod
    def _get_cell_spans(cell: Tag) -> tuple[int, int]:
        """Extract colspan and rowspan values from a table cell tag.

        This function retrieves the 'colspan' and 'rowspan' attributes from a given
        table cell tag.
        If the attribute does not exist or it is not numeric, it defaults to 1.
        """
        raw_spans: tuple[str, str] = (
            str(cell.get("colspan", "1")),
            str(cell.get("rowspan", "1")),
        )
        int_spans: tuple[int, int] = (
            int(raw_spans[0]) if raw_spans[0].isnumeric() else 1,
            int(raw_spans[1]) if raw_spans[0].isnumeric() else 1,
        )

        return int_spans

    @staticmethod
    def parse_table_data(element: Tag) -> Optional[TableData]:  # noqa: C901
        nested_tables = element.find("table")
        if nested_tables is not None:
            _log.debug("Skipping nested table.")
            return None

        # Find the number of rows and columns (taking into account spans)
        num_rows = 0
        num_cols = 0
        for row in element("tr"):
            col_count = 0
            is_row_header = True
            if not isinstance(row, Tag):
                continue
            for cell in row(["td", "th"]):
                if not isinstance(row, Tag):
                    continue
                cell_tag = cast(Tag, cell)
                col_span, row_span = HTMLDocumentBackend._get_cell_spans(cell_tag)
                col_count += col_span
                if cell_tag.name == "td" or row_span == 1:
                    is_row_header = False
            num_cols = max(num_cols, col_count)
            if not is_row_header:
                num_rows += 1

        _log.debug(f"The table has {num_rows} rows and {num_cols} cols.")

        grid: list = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Iterate over the rows in the table
        start_row_span = 0
        row_idx = -1
        for row in element("tr"):
            if not isinstance(row, Tag):
                continue

            # For each row, find all the column cells (both <td> and <th>)
            cells = row(["td", "th"])

            # Check if cell is in a column header or row header
            col_header = True
            row_header = True
            for html_cell in cells:
                if isinstance(html_cell, Tag):
                    _, row_span = HTMLDocumentBackend._get_cell_spans(html_cell)
                    if html_cell.name == "td":
                        col_header = False
                        row_header = False
                    elif row_span == 1:
                        row_header = False
            if not row_header:
                row_idx += 1
                start_row_span = 0
            else:
                start_row_span += 1

            # Extract the text content of each cell
            col_idx = 0
            for html_cell in cells:
                if not isinstance(html_cell, Tag):
                    continue

                # extract inline formulas
                for formula in html_cell("inline-formula"):
                    math_parts = formula.text.split("$$")
                    if len(math_parts) == 3:
                        math_formula = f"$${math_parts[1]}$$"
                        formula.replace_with(NavigableString(math_formula))

                # TODO: extract content correctly from table-cells with lists
                text = html_cell.text

                # label = html_cell.name
                col_span, row_span = HTMLDocumentBackend._get_cell_spans(html_cell)
                if row_header:
                    row_span -= 1
                while (
                    col_idx < num_cols
                    and grid[row_idx + start_row_span][col_idx] is not None
                ):
                    col_idx += 1
                for r in range(start_row_span, start_row_span + row_span):
                    for c in range(col_span):
                        if row_idx + r < num_rows and col_idx + c < num_cols:
                            grid[row_idx + r][col_idx + c] = text

                table_cell = TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=start_row_span + row_idx,
                    end_row_offset_idx=start_row_span + row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    column_header=col_header,
                    row_header=((not col_header) and html_cell.name == "th"),
                )
                data.table_cells.append(table_cell)

        return data
