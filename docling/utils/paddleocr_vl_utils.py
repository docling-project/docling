"""Convert saved PaddleOCR-VL page results into ``DoclingDocument`` objects.

The adapter targets the single-page result schema emitted by PaddleOCR-VL 1.6
through PaddleX 3.7.2. It consumes an already-produced result and deliberately
does not import PaddleOCR, run a model, or reproduce Paddle's multi-page
``restructure_pages()`` workflow.

Coordinates are preserved in Paddle's processed-page pixel canvas with a
top-left origin. When polygon points are present, the rectangular
``block_bbox`` supplied by Paddle is used for Docling provenance because the
canonical provenance model does not currently store arbitrary polygons.
Picture-like ``block_content`` is kept as adjacent text rather than being
treated as an image description, because its meaning depends on provider
recognition and formatting switches.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ProvenanceItem,
    Size,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from docling.utils.chandra_utils import _parse_table_html

_log = logging.getLogger(__name__)

__all__ = ["parse_paddleocr_vl_result"]


class _PaddleOCRVLBlock(BaseModel):
    """Validated subset of a serialized PaddleOCR-VL page block."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    block_label: str
    block_content: str = ""
    block_bbox: tuple[float, float, float, float]
    block_id: int | None = None
    block_order: int | None = None
    group_id: int | None = None
    global_block_id: int | None = None
    global_group_id: int | None = None
    block_polygon_points: list[tuple[float, float]] | None = None

    @field_validator("block_label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("block_label must not be empty")
        return value

    @field_validator("block_bbox")
    @classmethod
    def _validate_finite_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not all(math.isfinite(coord) for coord in value):
            raise ValueError("block_bbox coordinates must be finite")
        return value

    @field_validator("block_polygon_points")
    @classmethod
    def _validate_finite_polygon(
        cls, value: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        if value is not None and not all(
            math.isfinite(coord) for point in value for coord in point
        ):
            raise ValueError("block_polygon_points coordinates must be finite")
        return value


class _PaddleOCRVLPage(BaseModel):
    """Validated subset of a serialized PaddleOCR-VL single-page result."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    input_path: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    model_settings: dict[str, Any]
    parsing_res_list: list[_PaddleOCRVLBlock]


_TEXT_LABEL_MAP: dict[str, DocItemLabel] = {
    "abstract": DocItemLabel.TEXT,
    "algorithm": DocItemLabel.TEXT,
    "aside_text": DocItemLabel.TEXT,
    "content": DocItemLabel.TEXT,
    "figure_title": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    "footnote": DocItemLabel.FOOTNOTE,
    "formula_number": DocItemLabel.TEXT,
    "header": DocItemLabel.PAGE_HEADER,
    "number": DocItemLabel.TEXT,
    "ocr": DocItemLabel.TEXT,
    "reference": DocItemLabel.TEXT,
    "reference_content": DocItemLabel.REFERENCE,
    "spotting": DocItemLabel.TEXT,
    "text": DocItemLabel.TEXT,
    "vertical_text": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.FOOTNOTE,
}

_FORMULA_LABELS = {"display_formula", "formula", "inline_formula"}
_PICTURE_LABELS = {
    "chart",
    "footer_image",
    "header_image",
    "image",
    "seal",
}

_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.*?)\s*$", re.DOTALL)
_CENTERED_DIV_PATTERN = re.compile(
    r'^\s*<div style="text-align: center;">(.*)</div>\s*$',
    re.DOTALL,
)


def _parse_formatted_heading(text: str) -> tuple[str, int | None]:
    """Undo Paddle's optional Markdown heading wrapper."""
    match = _MARKDOWN_HEADING_PATTERN.match(text)
    if match is None:
        return text, None
    return match.group(2), len(match.group(1))


def _strip_formatted_centering(text: str) -> str:
    """Undo Paddle's optional centered ``div`` around textual captions."""
    match = _CENTERED_DIV_PATTERN.match(text)
    return match.group(1) if match is not None else text


def _load_page_payload(
    content: Mapping[str, Any] | str | bytes,
) -> _PaddleOCRVLPage:
    if isinstance(content, Mapping):
        payload: Any = dict(content)
    elif isinstance(content, (str, bytes)):
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid PaddleOCR-VL JSON") from exc
    else:
        raise TypeError(
            "content must be a JSON object mapping, JSON string, or JSON bytes"
        )

    if not isinstance(payload, Mapping):
        raise ValueError("PaddleOCR-VL JSON must contain an object")

    # ``result.json`` returns {"res": payload}; ``save_to_json()`` writes the
    # bare payload. Both forms represent the same official result contract.
    if "res" in payload:
        payload = payload["res"]
        if not isinstance(payload, Mapping):
            raise ValueError("PaddleOCR-VL 'res' must contain an object")

    return _PaddleOCRVLPage.model_validate(payload)


def _resolve_filename(page: _PaddleOCRVLPage, filename: str | None) -> str:
    if filename:
        return filename
    if page.input_path:
        # Provider results can be produced on a different operating system
        # than the one running this adapter. Normalize both path separators
        # before deriving the source filename.
        return Path(page.input_path.replace("\\", "/")).name
    return "paddleocr_vl_result.json"


def _resolve_page_no(page: _PaddleOCRVLPage, page_no: int | None) -> int:
    if page_no is not None:
        if page_no < 1:
            raise ValueError("page_no must be 1-based and greater than zero")
    if (
        page.page_count is not None
        and page.page_index is not None
        and page.page_index >= page.page_count
    ):
        raise ValueError(
            f"page_index {page.page_index} is outside page_count {page.page_count}"
        )
    if page.page_count is not None and page.page_count > 1 and page.page_index is None:
        raise ValueError(
            "A multi-page PaddleOCR-VL result without page_index is ambiguous; "
            "concatenated results are unsupported, so parse the original per-page "
            "results instead"
        )
    if page_no is not None:
        return page_no
    if page.page_index is not None:
        return page.page_index + 1
    return 1


def _make_provenance(
    block: _PaddleOCRVLBlock,
    *,
    page: _PaddleOCRVLPage,
    page_no: int,
) -> ProvenanceItem:
    left, top, right, bottom = block.block_bbox
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(
            f"Invalid block_bbox for block {block.block_id}: {block.block_bbox}"
        )
    if right > page.width or bottom > page.height:
        raise ValueError(
            f"block_bbox for block {block.block_id} exceeds the page canvas: "
            f"{block.block_bbox} not within {page.width}x{page.height}"
        )

    bbox = BoundingBox(
        l=left,
        t=top,
        r=right,
        b=bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    return ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))


def _add_block(
    doc: DoclingDocument,
    block: _PaddleOCRVLBlock,
    *,
    page: _PaddleOCRVLPage,
    page_no: int,
) -> None:
    prov = _make_provenance(block, page=page, page_no=page_no)
    label = block.block_label
    text = block.block_content
    formatted_content = page.model_settings.get("format_block_content", False) is True

    if label == "doc_title":
        if formatted_content:
            text, _ = _parse_formatted_heading(text)
        doc.add_title(text=text, prov=prov)
    elif label == "paragraph_title":
        heading_level = 1
        if formatted_content:
            text, markdown_level = _parse_formatted_heading(text)
            if markdown_level is not None:
                # Paddle reserves Markdown H1 for ``doc_title``. Docling's
                # section level 1 is exported as H2, so remove that offset.
                heading_level = max(1, markdown_level - 1)
        doc.add_heading(text=text, level=heading_level, prov=prov)
    elif label == "table":
        table_data = _parse_table_html(text)
        if table_data.num_rows == 0 or table_data.num_cols == 0:
            raise ValueError(
                f"Invalid table HTML for PaddleOCR-VL block {block.block_id}"
            )
        doc.add_table(data=table_data, prov=prov)
    elif label in _FORMULA_LABELS:
        doc.add_formula(text=text, prov=prov)
    elif label in _PICTURE_LABELS:
        doc.add_picture(prov=prov)
        if text:
            # Depending on provider switches, picture-like block content can
            # be OCR text, chart rows, or markup. It is not guaranteed to be
            # a natural-language picture description, so preserve it as a
            # separate text item rather than strengthening its semantics.
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
    else:
        doc_label = _TEXT_LABEL_MAP.get(label)
        if doc_label is None:
            _log.warning(
                "Unsupported PaddleOCR-VL block label %r; mapping it to text",
                label,
            )
            doc_label = DocItemLabel.TEXT
        if formatted_content and doc_label == DocItemLabel.CAPTION:
            text = _strip_formatted_centering(text)
        doc.add_text(label=doc_label, text=text, prov=prov)


def parse_paddleocr_vl_result(
    content: Mapping[str, Any] | str | bytes,
    *,
    filename: str | None = None,
    page_no: int | None = None,
) -> DoclingDocument:
    """Convert one saved PaddleOCR-VL page result into a ``DoclingDocument``.

    Args:
        content: A bare page payload written by ``save_to_json()``, the
            ``{"res": payload}`` mapping returned by ``result.json``, or either
            form encoded as JSON text/bytes.
        filename: Optional source filename override. Otherwise ``input_path``
            is used when present.
        page_no: Optional 1-based page number override. Otherwise Paddle's
            0-based ``page_index`` is converted to Docling's 1-based numbering.
            This cannot make a concatenated multi-page payload valid; those
            results remain unsupported and must be parsed per page.

    Returns:
        A single-page ``DoclingDocument`` with typed items and provenance in
        Paddle's processed-page pixel coordinate space.

    Raises:
        TypeError: If ``content`` is not a supported input type.
        ValueError: If JSON, page scope, geometry, or a table payload is invalid.
        pydantic.ValidationError: If the provider result does not match the
            required PaddleOCR-VL page schema.
    """
    page = _load_page_payload(content)
    resolved_filename = _resolve_filename(page, filename)
    resolved_page_no = _resolve_page_no(page, page_no)

    origin = DocumentOrigin(
        filename=resolved_filename,
        mimetype="application/json",
        binary_hash=0,
    )
    doc = DoclingDocument(name=Path(resolved_filename).stem, origin=origin)
    doc.add_page(
        page_no=resolved_page_no,
        size=Size(width=page.width, height=page.height),
    )

    for block in page.parsing_res_list:
        _add_block(doc, block, page=page, page_no=resolved_page_no)

    return doc
