"""Utilities for parsing chandra-ocr-2 HTML-with-bbox format.

chandra-ocr-2 produces HTML where each layout element is a top-level
``<div data-bbox="x0 y0 x1 y1" data-label="Label">content</div>``.
Bboxes are in 0-1000 normalized coordinate space.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from PIL import Image as PILImage

_log = logging.getLogger(__name__)

# Mapping from chandra-ocr-2 layout labels to DocItemLabel.
# Labels not present in DocItemLabel fall back to TEXT.
_LABEL_MAP: dict[str, DocItemLabel] = {
    "Text": DocItemLabel.TEXT,
    "Title": DocItemLabel.TITLE,
    "Section-Header": DocItemLabel.SECTION_HEADER,
    "Table": DocItemLabel.TABLE,
    "Figure": DocItemLabel.PICTURE,
    "Image": DocItemLabel.PICTURE,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-Header": DocItemLabel.PAGE_HEADER,
    "Page-Footer": DocItemLabel.PAGE_FOOTER,
    "List-Group": DocItemLabel.LIST_ITEM,
    "Equation-Block": DocItemLabel.FORMULA,
    "Code-Block": DocItemLabel.CODE,
    "Form": DocItemLabel.FORM,
    "Table-Of-Contents": DocItemLabel.TEXT,
    "Complex-Block": DocItemLabel.TEXT,
    "Chemical-Block": DocItemLabel.FORMULA,
    "Diagram": DocItemLabel.PICTURE,
    "Bibliography": DocItemLabel.REFERENCE,
    "Blank-Page": DocItemLabel.TEXT,
}

# Regex to match top-level divs with data-bbox and data-label (either order).
# Captures: (all attributes string), (inner content).
_DIV_PATTERN = re.compile(
    r"<div\s+([^>]*?)>(.*?)</div>",
    re.DOTALL,
)

_BBOX_ATTR = re.compile(r'data-bbox="(\d+\s+\d+\s+\d+\s+\d+)"')
_LABEL_ATTR = re.compile(r'data-label="([^"]+)"')

# Strip HTML tags from inner content.
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _TAG_RE.sub("", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _TableHTMLParser(HTMLParser):
    """Lightweight HTML table parser using stdlib html.parser."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict]] = []  # list of rows, each row is list of cell dicts
        self._in_row = False
        self._in_cell = False
        self._current_cell: dict = {}
        self._current_row: list[dict] = []
        self._cell_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "tr":
            self._in_row = True
            self._current_row = []
        elif tag_lower in ("td", "th"):
            self._in_cell = True
            attr_dict = dict(attrs)
            self._current_cell = {
                "tag": tag_lower,
                "colspan": int(attr_dict.get("colspan") or "1"),
                "rowspan": int(attr_dict.get("rowspan") or "1"),
            }
            self._cell_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("td", "th") and self._in_cell:
            self._current_cell["text"] = " ".join(
                "".join(self._cell_text_parts).split()
            ).strip()
            self._current_row.append(self._current_cell)
            self._in_cell = False
        elif tag_lower == "tr" and self._in_row:
            self.rows.append(self._current_row)
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text_parts.append(data)


def _parse_table_html(html_content: str) -> TableData:
    """Parse HTML table content and create TableData structure."""
    table_match = re.search(
        r"<table[^>]*>.*?</table>", html_content, re.DOTALL | re.IGNORECASE
    )
    if not table_match:
        return TableData(num_rows=0, num_cols=0, table_cells=[])

    try:
        parser = _TableHTMLParser()
        parser.feed(table_match.group(0))

        rows = parser.rows
        if not rows:
            return TableData(num_rows=0, num_cols=0, table_cells=[])

        num_rows = len(rows)
        num_cols = 0
        for row in rows:
            col_count = sum(cell["colspan"] for cell in row)
            num_cols = max(num_cols, col_count)

        grid: list[list[Union[None, str]]] = [
            [None for _ in range(num_cols)] for _ in range(num_rows)
        ]
        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row:
                while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1
                if col_idx >= num_cols:
                    break

                text = cell["text"]
                colspan = cell["colspan"]
                rowspan = cell["rowspan"]
                is_header = cell["tag"] == "th"

                for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                    for c in range(col_idx, min(col_idx + colspan, num_cols)):
                        grid[r][c] = text

                table_data.table_cells.append(
                    TableCell(
                        text=text,
                        row_span=rowspan,
                        col_span=colspan,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + rowspan,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + colspan,
                        column_header=is_header and row_idx == 0,
                        row_header=is_header and col_idx == 0,
                    )
                )
                col_idx += colspan

        return table_data

    except Exception as e:
        _log.warning(f"Failed to parse table HTML: {e}")
        return TableData(num_rows=0, num_cols=0, table_cells=[])


def parse_chandra_html(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
) -> DoclingDocument:
    """Parse chandra-ocr-2 HTML output into a DoclingDocument.

    Args:
        content: Raw HTML string from chandra-ocr-2.
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.

    Returns:
        DoclingDocument populated with parsed elements.
    """
    origin = DocumentOrigin(
        filename=filename,
        mimetype="text/html",
        binary_hash=0,
    )
    doc = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)

    pg_width = original_page_size.width
    pg_height = original_page_size.height

    scale_x = pg_width / 1000
    scale_y = pg_height / 1000

    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / pg_width)

    doc.add_page(
        page_no=page_no,
        size=Size(width=pg_width, height=pg_height),
        image=ImageRef.from_pil(image=page_image, dpi=image_dpi)
        if page_image
        else None,
    )

    if not content or not content.strip():
        return doc

    for m in _DIV_PATTERN.finditer(content):
        attrs_str = m.group(1)
        inner_html = m.group(2)

        bbox_m = _BBOX_ATTR.search(attrs_str)
        label_m = _LABEL_ATTR.search(attrs_str)
        if not bbox_m or not label_m:
            continue

        coords = bbox_m.group(1).split()
        if len(coords) != 4:
            continue

        try:
            x0, y0, x1, y1 = (int(c) for c in coords)
        except ValueError:
            continue

        label_str = label_m.group(1)

        bbox = BoundingBox(
            l=x0 * scale_x,
            t=y0 * scale_y,
            r=x1 * scale_x,
            b=y1 * scale_y,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=[0, 0])

        doc_label = _LABEL_MAP.get(label_str, DocItemLabel.TEXT)

        if label_str == "Table":
            table_data = _parse_table_html(inner_html)
            doc.add_table(data=table_data, prov=prov)
        elif label_str in ("Figure", "Image", "Diagram"):
            doc.add_picture(prov=prov)
        elif label_str == "Title":
            doc.add_title(text=_strip_tags(inner_html), prov=prov)
        elif label_str == "Section-Header":
            doc.add_heading(text=_strip_tags(inner_html), prov=prov)
        else:
            doc.add_text(label=doc_label, text=_strip_tags(inner_html), prov=prov)

    return doc
