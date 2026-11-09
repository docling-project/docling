# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Typhoon OCR V2 by the Typhoon team: https://huggingface.co/typhoon-ai/typhoon-ocr2-2b

"""Parse Typhoon OCR V2 layout JSON into a :class:`DoclingDocument`.

The model replies with a JSON list of layout elements in reading order, usually
inside a ``json`` code fence::

    [{"bbox": [62, 48, 938, 96], "category": "title", "text": "Annual Report"}]

Boxes are ``[x1, y1, x2, y2]`` on a 0-1000 grid relative to the page width and
height, with the origin at the top left. The text is Markdown, HTML for a table,
LaTeX for a formula, and empty for a figure. The model uses the ``title``
category for every heading and writes no heading markup, so each title becomes a
section header. A reply that the token limit cuts keeps its complete elements.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

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

from docling.utils.chandra_utils import _add_html_table
from docling.utils.deepseekocr_utils import _parse_table_html
from docling.utils.dots_utils import _add_inline_text, _unescape_markdown_literals
from docling.utils.lightonocr_utils import _THINK
from docling.utils.nemotron_parse_v3_5_utils import _add_rich_text
from docling.utils.unlimited_ocr2_utils import _strip_table_tags
from docling.utils.vlm_utils import (
    link_adjacent_captions,
    normalize_section_header_levels,
    parse_markdown_heading,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

_GRID = 1000
# `_add_rich_text` moves the items of a page header or footer to the furniture layer.
_TEXT_LABELS = {
    "text": DocItemLabel.TEXT,
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
    "figure_caption": DocItemLabel.CAPTION,
    "table_caption": DocItemLabel.CAPTION,
    "formula_caption": DocItemLabel.CAPTION,
    "figure_footnote": DocItemLabel.FOOTNOTE,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "page_footnote": DocItemLabel.FOOTNOTE,
}
_CATEGORIES = {*_TEXT_LABELS, "figure", "formula", "table", "title"}
# A formula caption stays standalone: a formula cannot own a caption.
_CAPTION_OWNER_LABELS = {
    "figure_caption": {DocItemLabel.PICTURE},
    "table_caption": {DocItemLabel.TABLE},
}
# A JSON string holds no raw line break, so the fence cannot match inside an element.
# The closing fence is optional: the token limit can cut it off.
_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)(?:\r?\n[ \t]*```|\Z)", re.DOTALL)
_LIST_START = re.compile(r"\[\s*[{\]]")
_DECODER = json.JSONDecoder()


def _decode_list(text: str, start: int) -> list[Any]:
    """Decode the list at ``start`` one value at a time and keep the complete values.

    Unlike a cut at the last ``}``, this is safe with braces in strings, such as the
    LaTeX ``\\frac{a}{b}``.

    Raises:
        ValueError: If the list breaks before its first complete value.
    """
    values: list[Any] = []
    pos = start + 1
    while True:
        while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        if pos < len(text) and text[pos] == "]":
            return values
        try:
            value, pos = _DECODER.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        values.append(value)
    if not values:
        raise ValueError("the reply ends before its first complete layout element")
    _log.warning(
        "Typhoon OCR V2 reply is cut or holds invalid JSON; keeping %d complete "
        "elements",
        len(values),
    )
    return values


def _elements(content: str) -> list[Any]:
    """Return the layout elements of one reply. A blank reply has none.

    Raises:
        ValueError: If a reply that is not blank holds no layout elements.
    """
    text = _THINK.sub("", content)
    if (fence := _FENCE.search(text)) is not None:
        text = fence.group(1)
    text = text.strip()
    if not text:
        return []
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        start = _LIST_START.search(text)
        if start is None:
            raise ValueError(
                "the reply holds no JSON list of layout elements"
            ) from None
        data = _decode_list(text, start.start())
    if isinstance(data, dict):
        # The prompt asks for "a single JSON object": accept one element or a
        # wrapper around the list. Check the element keys first: a box is a list.
        if {"bbox", "category", "text"} & data.keys():
            data = [data]
        else:
            lists = [value for value in data.values() if isinstance(value, list)]
            if len(lists) != 1:
                raise ValueError("the reply is a JSON object without one element list")
            data = lists[0]
    if not isinstance(data, list) or (
        data and not any(isinstance(value, dict) for value in data)
    ):
        raise ValueError("the reply is JSON but not a list of layout elements")
    return data


def _category(element: dict[str, Any]) -> str:
    raw = element.get("category")
    return raw.strip().lower() if isinstance(raw, str) else ""


def _is_coordinate(value: Any) -> bool:
    # json accepts NaN and Infinity, and an int can be too large for a float.
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


def _provenance(
    raw: Any, text: str, page_size: Size, page_no: int
) -> ProvenanceItem | None:
    """Map a box on the 0-1000 page grid to page points, or None for a bad box."""
    if not isinstance(raw, list) or len(raw) != 4 or not all(map(_is_coordinate, raw)):
        return None
    x1, y1, x2, y2 = (min(max(value, 0), _GRID) / _GRID for value in raw)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x1 == x2 or y1 == y2:
        return None
    return ProvenanceItem(
        page_no=page_no,
        charspan=(0, len(text)),
        bbox=BoundingBox(
            l=x1 * page_size.width,
            t=y1 * page_size.height,
            r=x2 * page_size.width,
            b=y2 * page_size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )


def _add_table(doc: DoclingDocument, html: str, prov: ProvenanceItem | None) -> bool:
    """Add a table and return False if the HTML gives no cells."""
    if prov is not None:
        try:
            return _add_html_table(doc, html, prov) is not None
        except ValueError as exc:
            # A dense page can pass the grid limit of the shared table mapper.
            _log.warning("Reading a large Typhoon OCR V2 table without spans: %s", exc)
    data = _parse_table_html(html)
    if not data.table_cells:
        return False
    doc.add_table(data=data, prov=prov)
    return True


def _add_element(
    doc: DoclingDocument, category: str, text: str, prov: ProvenanceItem | None
) -> None:
    if category == "figure":
        doc.add_picture(prov=prov)
        if text:  # The prompt asks for no figure text; keep text that the model gives.
            _add_rich_text(
                doc,
                text,
                label=DocItemLabel.TEXT,
                prov=prov.model_copy(deep=True) if prov is not None else None,
            )
        return
    if not text:
        return
    if category == "title":
        # Without a `#` marker, the raw depth is 2, as for `##`.
        clean, depth = parse_markdown_heading(text)
        heading = doc.add_heading(text="", orig=text, level=depth or 2, prov=prov)
        if not _add_inline_text(doc, clean, label=DocItemLabel.TEXT, parent=heading):
            heading.text = _unescape_markdown_literals(clean)
        return
    if category == "formula":
        doc.add_formula(text=strip_math_delimiters(text), orig=text, prov=prov)
        return
    if category == "table":
        if _add_table(doc, text, prov):
            return
        _log.warning("Keeping a Typhoon OCR V2 table without cells as text")
        text = _strip_table_tags(text)
        if not text:
            return
    _add_rich_text(
        doc, text, label=_TEXT_LABELS.get(category, DocItemLabel.TEXT), prov=prov
    )


def parse_typhoon_ocr2_json(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    *,
    normalize_heading_levels: bool = True,
) -> DoclingDocument:
    """Parse the reply of one Typhoon OCR V2 page request.

    Args:
        content: Raw model output for one page.
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.
        normalize_heading_levels: If False, keep raw heading depths for a later
            document-wide :func:`normalize_section_header_levels`.

    Returns:
        A document for this page. A blank reply or an empty list gives an empty page.

    Raises:
        ValueError: If a reply that is not blank holds no layout elements.
    """
    doc = DoclingDocument(
        name=Path(filename).stem,
        origin=DocumentOrigin(
            filename=filename, mimetype="application/json", binary_hash=0
        ),
    )
    doc.add_page(
        page_no=page_no,
        size=original_page_size,
        image=ImageRef.from_pil(
            image=page_image,
            dpi=max(1, round(72 * page_image.width / original_page_size.width)),
        )
        if page_image is not None
        else None,
    )
    elements = _elements(content)
    unknown = {_category(e) for e in elements if isinstance(e, dict)} - _CATEGORIES
    if unknown:
        _log.warning(
            "Reading unknown Typhoon OCR V2 categories as text: %s", sorted(unknown)
        )

    block_starts: list[int] = []
    caption_owners: dict[int, set[DocItemLabel]] = {}
    without_box = 0
    for index, element in enumerate(elements):
        # A skipped element still separates its neighbors for caption links.
        block_starts.append(len(doc.body.children))
        if not isinstance(element, dict):
            continue
        category = _category(element)
        raw_text = element.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        prov = _provenance(element.get("bbox"), text, original_page_size, page_no)
        if prov is None and (text or category == "figure"):
            without_box += 1
        if category in _CAPTION_OWNER_LABELS:
            caption_owners[index] = _CAPTION_OWNER_LABELS[category]
        _add_element(doc, category, text, prov)
    if without_box:
        _log.warning(
            "Keeping %d Typhoon OCR V2 elements without a valid box", without_box
        )

    link_adjacent_captions(doc, block_starts, caption_owners)
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
