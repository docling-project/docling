"""Utilities for parsing dots.ocr / dots.mocr JSON layout format.

dots.ocr (3B, Qwen2.5-VL based) and dots.mocr produce a JSON array of
layout elements::

    [{"bbox": [x1, y1, x2, y2], "category": "Label", "text": "content"}, ...]

Bboxes are pixel coordinates relative to the model input resolution.
If ``model_image_size`` is provided the coords are rescaled to the
original page coordinate space.

11 categories: Caption, Footnote, Formula, List-item, Page-footer,
Page-header, Picture, Section-header, Table, Text, Title.

Tables arrive as HTML ``<table>``; formulas as LaTeX; Pictures have no
``text`` field.  The model sometimes truncates output, so the parser
is tolerant of malformed JSON.
"""

import json
import logging
from typing import Optional

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _parse_table_html

_log = logging.getLogger(__name__)

# Mapping from dots.ocr/dots.mocr category strings to DocItemLabel.
_LABEL_MAP: dict[str, DocItemLabel] = {
    "Text": DocItemLabel.TEXT,
    "Title": DocItemLabel.TITLE,
    "Section-header": DocItemLabel.SECTION_HEADER,
    "Table": DocItemLabel.TABLE,
    "Picture": DocItemLabel.PICTURE,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
    "List-item": DocItemLabel.LIST_ITEM,
    "Formula": DocItemLabel.FORMULA,
}


def _clean_json(raw: str) -> str:
    """Best-effort cleanup of potentially truncated JSON arrays.

    1. Strip leading text before the first ``[``.
    2. If the array does not end with ``]``, find the last ``}`` and append ``]``.
    3. Return ``"[]"`` if no valid JSON structure found.
    """
    idx = raw.find("[")
    if idx == -1:
        return "[]"
    raw = raw[idx:]

    stripped = raw.rstrip()
    if not stripped.endswith("]"):
        last_brace = stripped.rfind("}")
        if last_brace == -1:
            return "[]"
        raw = stripped[: last_brace + 1] + "]"

    return raw


def parse_dots_json(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    model_image_size: Size | None = None,
) -> DoclingDocument:
    """Parse dots.ocr / dots.mocr JSON output into a DoclingDocument.

    Args:
        content: Raw JSON string (array of element dicts).
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.
        model_image_size: If provided, bbox pixel coords are rescaled from
            this resolution to *original_page_size*.

    Returns:
        DoclingDocument populated with parsed elements.
    """
    origin = DocumentOrigin(
        filename=filename,
        mimetype="application/json",
        binary_hash=0,
    )
    doc = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)

    pg_width = original_page_size.width
    pg_height = original_page_size.height

    # Compute rescaling factors
    if model_image_size is not None:
        scale_x = pg_width / model_image_size.width
        scale_y = pg_height / model_image_size.height
    else:
        # No rescaling — assume pixel coords already match page coords
        scale_x = 1.0
        scale_y = 1.0

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

    cleaned = _clean_json(content)
    try:
        elements = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        _log.warning("Failed to parse dots JSON after cleanup: %s", exc)
        return doc

    if not isinstance(elements, list):
        _log.warning("Expected JSON array, got %s", type(elements).__name__)
        return doc

    for elem in elements:
        if not isinstance(elem, dict):
            continue

        category = elem.get("category", "")
        raw_bbox = elem.get("bbox")
        text = elem.get("text", "")

        if not raw_bbox or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in raw_bbox)
        except (ValueError, TypeError):
            continue

        bbox = BoundingBox(
            l=x1 * scale_x,
            t=y1 * scale_y,
            r=x2 * scale_x,
            b=y2 * scale_y,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=[0, 0])

        doc_label = _LABEL_MAP.get(category, DocItemLabel.TEXT)

        if category == "Table":
            table_data = _parse_table_html(text)
            doc.add_table(data=table_data, prov=prov)
        elif category == "Picture":
            doc.add_picture(prov=prov)
        elif category == "Title":
            doc.add_title(text=text, prov=prov)
        elif category == "Section-header":
            doc.add_heading(text=text, prov=prov)
        else:
            doc.add_text(label=doc_label, text=text, prov=prov)

    return doc
