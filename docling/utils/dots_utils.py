# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    NodeItem,
    ProvenanceItem,
    Size,
    TableData,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _add_html_fragment, _add_html_table
from docling.utils.code_language import detect_code_language
from docling.utils.markdown_math import _MATH, create_markdown_parser
from docling.utils.vlm_utils import (
    link_adjacent_captions,
    normalize_section_header_levels,
    parse_markdown_heading,
)

_log = logging.getLogger(__name__)

_FENCED_CODE_PATTERN = re.compile(
    r"^```(?P<hint>[^`\r\n]*)\r?\n(?P<text>.*?)(?:\r?\n)?```[ \t]*$",
    re.DOTALL,
)
_HTML_TAG_PATTERN = re.compile(
    r"</?(?:a|b|br|code|del|div|em|i|math|p|s|span|strong|sub|sup|table|tbody|td|tfoot|th|thead|tr|u)(?=[\s/>])",
    re.IGNORECASE,
)
_LITERAL_INLINE_PATTERN = re.compile(
    rf"{_MATH.pattern}|<code\b[^>]*>[\s\S]*?</code>",
    re.IGNORECASE,
)
# Literal spans and code spans pass through; `\\` keeps `\\*` from unescaping.
# A code span opens only at a full backtick run, read atomically in a lookahead:
# a backtracking `(`+)` takes cubic time on a long unclosed run.
_MARKDOWN_ESCAPE_PATTERN = re.compile(
    rf"{_LITERAL_INLINE_PATTERN.pattern}"
    r"|(?<!`)(?=(?P<ticks>`+))(?P=ticks)[\s\S]*?(?<!`)(?P=ticks)(?!`)"
    r"|\\(?P<escape>[*_])|\\\\",
    re.IGNORECASE,
)


def _unescape_markdown_literals(text: str) -> str:
    r"""Turn the Markdown escapes `\*` and `\_` into literal characters."""
    return _MARKDOWN_ESCAPE_PATTERN.sub(
        lambda match: match.group("escape") or match.group(0), text
    )


def _markdown_inline_html(text: str) -> str:
    """Render Markdown emphasis as inline HTML; unescape other text."""
    html_text = text
    if any(token in text for token in ("*", "_", "$", r"\(", r"\[", "`")):
        rendered = create_markdown_parser(inline_only=True).convert(text)
        if re.search(r"<(?:strong|em|math)\b", rendered):
            # Keep soft and hard line breaks as <br>.
            html_text = re.sub(r"(?:<br />)?\n", "<br>", rendered.strip())
        else:
            html_text = _unescape_markdown_literals(text)
    return html_text


def _add_inline_text(
    doc: DoclingDocument,
    text: str,
    *,
    label: DocItemLabel,
    prov: ProvenanceItem | None = None,
    parent: NodeItem | None = None,
) -> bool:
    """Map Markdown emphasis and inline HTML through the rich HTML builder."""
    html_text = _markdown_inline_html(text)
    if not _HTML_TAG_PATTERN.search(html_text):
        return False
    if parent is None:
        parent = doc.add_text(label=label, text="", orig=text, prov=prov)
        label = DocItemLabel.TEXT
        prov = None
    return _add_html_fragment(doc, html_text, label=label, prov=prov, parent=parent)


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
    *,
    normalize_heading_levels: bool = True,
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
        normalize_heading_levels: If False, keep raw heading depths for a later
            document-wide :func:`normalize_section_header_levels`.

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

    current_list_group = None
    block_starts: list[int] = []
    caption_owners: dict[int, set[DocItemLabel]] = {}
    for index, elem in enumerate(elements):
        block_starts.append(len(doc.body.children))
        if not isinstance(elem, dict):
            continue

        category = elem.get("category", "")
        if category == "Caption":
            caption_owners[index] = {
                DocItemLabel.TABLE,
                DocItemLabel.PICTURE,
                DocItemLabel.CODE,
            }
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
            current_list_group = None
            if _add_html_table(doc, text, prov) is None:
                _log.warning("Invalid dots HTML table; preserving an empty table")
                doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=prov)
        elif category == "Picture":
            current_list_group = None
            doc.add_picture(prov=prov)
        elif category == "Title":
            current_list_group = None
            clean_text, _ = parse_markdown_heading(text)
            item = doc.add_title(text="", orig=text, prov=prov)
            if not _add_inline_text(
                doc, clean_text, label=DocItemLabel.TEXT, parent=item
            ):
                item.text = _unescape_markdown_literals(clean_text)
        elif category == "Section-header":
            current_list_group = None
            clean_text, markdown_level = parse_markdown_heading(text)
            item = doc.add_heading(
                text="",
                orig=text,
                # Raw markup depth, normalized per document later.
                level=markdown_level if markdown_level is not None else 2,
                prov=prov,
            )
            if not _add_inline_text(
                doc, clean_text, label=DocItemLabel.TEXT, parent=item
            ):
                item.text = _unescape_markdown_literals(clean_text)
        elif category == "List-item":
            if current_list_group is None:
                current_list_group = doc.add_list_group()
            item = doc.add_list_item(
                text="", orig=text, parent=current_list_group, prov=prov
            )
            if not _add_inline_text(doc, text, label=DocItemLabel.TEXT, parent=item):
                item.text = _unescape_markdown_literals(text)
        else:
            current_list_group = None
            code = _FENCED_CODE_PATTERN.fullmatch(text)
            if code is not None:
                code_text = code.group("text")
                doc.add_code(
                    text=code_text,
                    orig=text,
                    prov=prov,
                    code_language=detect_code_language(
                        code_text, hint=code.group("hint").strip() or None
                    ),
                )
            elif category == "Formula":
                doc.add_text(label=doc_label, text=text, prov=prov)
            elif not _add_inline_text(doc, text, label=doc_label, prov=prov):
                doc.add_text(
                    label=doc_label,
                    text=_unescape_markdown_literals(text),
                    orig=text,
                    prov=prov,
                )

    link_adjacent_captions(doc, block_starts, caption_owners)
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
