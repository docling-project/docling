"""Utilities for parsing DeepSeek OCR annotated markdown format."""

import logging
import re
from typing import Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    RefItem,
    Size,
    TableCell,
    TableData,
    TextItem,
)
from lxml import etree
from PIL import Image as PILImage

_log = logging.getLogger(__name__)


def _parse_table_html(html_content: str) -> TableData:
    """Parse HTML table content and create TableData structure.

    Args:
        html_content: HTML string containing <table> element

    Returns:
        TableData with parsed table structure
    """
    # Extract table HTML if wrapped in other content
    table_match = re.search(
        r"<table[^>]*>.*?</table>", html_content, re.DOTALL | re.IGNORECASE
    )
    if not table_match:
        # No table found, return empty table
        return TableData(num_rows=0, num_cols=0, table_cells=[])

    table_html = table_match.group(0)

    try:
        # Parse HTML with lxml
        parser = etree.HTMLParser()
        tree = etree.fromstring(table_html, parser)

        # Find all rows
        rows = tree.xpath(".//tr")
        if not rows:
            return TableData(num_rows=0, num_cols=0, table_cells=[])

        # Calculate grid dimensions
        num_rows = len(rows)
        num_cols = 0

        # First pass: determine number of columns
        for row in rows:
            cells = row.xpath("./td | ./th")
            col_count = 0
            for cell in cells:
                colspan = int(cell.get("colspan", "1"))
                col_count += colspan
            num_cols = max(num_cols, col_count)

        # Create grid to track cell positions
        grid: list[list[Union[None, str]]] = [
            [None for _ in range(num_cols)] for _ in range(num_rows)
        ]
        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Second pass: populate cells
        for row_idx, row in enumerate(rows):
            cells = row.xpath("./td | ./th")
            col_idx = 0

            for cell in cells:
                # Find next available column
                while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                if col_idx >= num_cols:
                    break

                # Get cell properties
                text = "".join(cell.itertext()).strip()
                colspan = int(cell.get("colspan", "1"))
                rowspan = int(cell.get("rowspan", "1"))
                is_header = cell.tag.lower() == "th"

                # Mark grid cells as occupied
                for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                    for c in range(col_idx, min(col_idx + colspan, num_cols)):
                        grid[r][c] = text

                # Create table cell
                table_cell = TableCell(
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
                table_data.table_cells.append(table_cell)

                col_idx += colspan

        return table_data

    except Exception as e:
        _log.warning(f"Failed to parse table HTML: {e}")
        return TableData(num_rows=0, num_cols=0, table_cells=[])


def _collect_annotation_content(
    lines: list[str],
    i: int,
    label_str: str,
    annotation_pattern: str,
    visited_lines: set[int],
) -> tuple[str, int]:
    """Collect content for an annotation.

    Args:
        lines: All lines from the document
        i: Current line index (after annotation line)
        label_str: The annotation label (e.g., 'table', 'text')
        annotation_pattern: Regex pattern to match annotations
        visited_lines: Set of already visited line indices

    Returns:
        Tuple of (content string, next line index)
    """
    content_lines = []

    # Special handling for table: extract only <table>...</table>
    if label_str == "table":
        table_started = False
        ii = i
        while ii < len(lines):
            line = lines[ii]
            if "<table" in line.lower():
                table_started = True
            if table_started:
                visited_lines.add(ii)
                content_lines.append(line.rstrip())
            if table_started and "</table>" in line.lower():
                break
            ii += 1
    else:
        # Original logic for other labels
        while i < len(lines):
            content_line = lines[i].strip()
            if content_line:
                if re.match(annotation_pattern, content_line):
                    break
                visited_lines.add(i)
                content_lines.append(lines[i].rstrip())
                i += 1
                if label_str not in ["figure", "image"]:
                    break
            else:
                i += 1
                if content_lines:
                    break

    return "\n".join(content_lines), i


def _process_annotation_item(
    label_str: str,
    content: str,
    prov: ProvenanceItem,
    caption_item: Optional[Union[TextItem, RefItem]],
    page_doc: DoclingDocument,
    label_map: dict[str, DocItemLabel],
) -> None:
    """Process and add a single annotation item to the document.

    Args:
        label_str: The annotation label
        content: The content text
        prov: Provenance information
        caption_item: Optional caption item to link
        page_doc: Document to add item to
        label_map: Mapping of label strings to DocItemLabel
    """
    doc_label = label_map.get(label_str, DocItemLabel.TEXT)

    if label_str in ["figure", "image"]:
        page_doc.add_picture(caption=caption_item, prov=prov)
    elif label_str == "table":
        table_data = _parse_table_html(content)
        page_doc.add_table(data=table_data, caption=caption_item, prov=prov)
    elif label_str == "title":
        clean_content = content
        if content.startswith("#"):
            hash_count = 0
            for char in content:
                if char == "#":
                    hash_count += 1
                else:
                    break
            clean_content = content[hash_count:].strip()
        page_doc.add_title(text=clean_content, prov=prov)
    elif label_str == "sub_title":
        heading_level = 1
        clean_content = content
        if content.startswith("#"):
            hash_count = 0
            for char in content:
                if char == "#":
                    hash_count += 1
                else:
                    break
            if hash_count > 1:
                heading_level = hash_count - 1
            clean_content = content[hash_count:].strip()
        page_doc.add_heading(text=clean_content, level=heading_level, prov=prov)
    else:
        page_doc.add_text(label=doc_label, text=content, prov=prov)


def parse_deepseekocr_markdown(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: Optional[PILImage.Image] = None,
) -> DoclingDocument:
    """Parse DeepSeek OCR markdown with label[[x1, y1, x2, y2]] format.

    This function parses markdown content that has been annotated with bounding box
    coordinates for different document elements.

    Labels supported:
    - text: Standard body text
    - title: Main document or section titles
    - sub_title: Secondary headings or sub-headers
    - table: Tabular data
    - table_caption: Descriptive text for tables
    - figure: Image-based elements or diagrams
    - figure_caption: Titles or descriptions for figures/images
    - header / footer: Content at top or bottom margins of pages

    Args:
        content: The annotated markdown content string
        page_image: Optional PIL Image of the page
        page_no: Page number (default: 1)
        filename: Source filename (default: "file")

    Returns:
        DoclingDocument with parsed content
    """
    # Label mapping
    label_map = {
        "text": DocItemLabel.TEXT,
        "title": DocItemLabel.TITLE,
        "sub_title": DocItemLabel.SECTION_HEADER,
        "table": DocItemLabel.TABLE,
        "table_caption": DocItemLabel.CAPTION,
        "figure": DocItemLabel.PICTURE,
        "figure_caption": DocItemLabel.CAPTION,
        "image": DocItemLabel.PICTURE,
        "image_caption": DocItemLabel.CAPTION,
        "header": DocItemLabel.PAGE_HEADER,
        "footer": DocItemLabel.PAGE_FOOTER,
    }

    # Pattern to match: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|> or label[[x1, y1, x2, y2]]
    annotation_pattern = r"^(?:<\|ref\|>)?(\w+)(?:<\|/ref\|>)?(?:<\|det\|>)?\[\[([0-9., ]+)\]\](?:<\|/det\|>)?\s*$"

    # Create a new document
    origin = DocumentOrigin(
        filename=filename,
        mimetype="text/markdown",
        binary_hash=0,
    )
    page_doc = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)

    # Get page dimensions - use original page size if provided, otherwise image size
    pg_width = original_page_size.width
    pg_height = original_page_size.height

    # Calculate scale factor for bbox conversion
    # VLM produces bboxes in unit of 1000
    scale_x = pg_width / 1000
    scale_y = pg_height / 1000

    # Calculate DPI for the image
    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / pg_width)

    # Add page metadata
    page_doc.add_page(
        page_no=page_no,
        size=Size(width=pg_width, height=pg_height),
        image=ImageRef.from_pil(image=page_image, dpi=image_dpi)
        if page_image
        else None,
    )

    # Split into lines and parse - collect all annotations first
    lines = content.split("\n")
    annotations = []
    i = 0
    visited_lines: set[int] = set()

    while i < len(lines):
        if i in visited_lines:
            i += 1
            continue

        line = lines[i].strip()
        match = re.match(annotation_pattern, line)
        if match:
            label_str = match.group(1)
            coords_str = match.group(2)

            try:
                coords = [float(x.strip()) for x in coords_str.split(",")]
                if len(coords) == 4:
                    # Scale bounding box from image coordinates to original page coordinates
                    bbox = BoundingBox(
                        l=coords[0] * scale_x,
                        t=coords[1] * scale_y,
                        r=coords[2] * scale_x,
                        b=coords[3] * scale_y,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )
                    prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=[0, 0])

                    # Get the content (next non-empty line)
                    i += 1
                    content_text, i = _collect_annotation_content(
                        lines, i, label_str, annotation_pattern, visited_lines
                    )
                    annotations.append((label_str, content_text, prov))
                    continue
            except (ValueError, IndexError):
                pass
        i += 1

    # Process annotations and link captions that appear AFTER tables/figures
    for idx, (label_str, content_text, prov) in enumerate(annotations):
        # Check if NEXT annotation is a caption for this table/figure/image
        # (caption appears AFTER table in the file: table[[...]] then table_caption[[...]])
        caption_item = None
        if label_str in ["table", "figure", "image"] and idx + 1 < len(annotations):
            next_label, next_content, next_prov = annotations[idx + 1]
            if (
                (label_str == "table" and next_label == "table_caption")
                or (label_str == "figure" and next_label == "figure_caption")
                or (label_str == "image" and next_label == "image_caption")
            ):
                # Create caption item
                caption_label = label_map.get(next_label, DocItemLabel.CAPTION)
                caption_item = page_doc.add_text(
                    label=caption_label,
                    text=next_content,
                    prov=next_prov,
                )

        # Skip if this is a caption that was already processed
        if label_str in ["figure_caption", "table_caption", "image_caption"]:
            if idx > 0:
                prev_label = annotations[idx - 1][0]
                if (
                    (label_str == "table_caption" and prev_label == "table")
                    or (label_str == "figure_caption" and prev_label == "figure")
                    or (label_str == "image_caption" and prev_label == "image")
                ):
                    continue

        # Add the item
        _process_annotation_item(
            label_str, content_text, prov, caption_item, page_doc, label_map
        )

    return page_doc
