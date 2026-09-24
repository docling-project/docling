# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Utilities for MinerU2 two-step document parsing."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Literal

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
    TableData,
    TextItem,
)
from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, ValidationError

from docling.utils.otsl import parse_otsl_output

_log = logging.getLogger(__name__)

MINERU2_LAYOUT_PROMPT = "\nLayout Detection:"
MINERU2_LAYOUT_IMAGE_SIZE = (1036, 1036)

_DEFAULT_RECOGNITION_PROMPT = "\nText Recognition:"
_RECOGNITION_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
}
_SKIP_RECOGNITION_TYPES = {
    "chart",
    "equation_block",
    "image",
    "image_block",
    "list",
}
_LAYOUT_PATTERN = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)",
    re.DOTALL,
)
_ROTATIONS = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}
_BLOCK_TYPES = {
    "algorithm",
    "aside_text",
    "caption",
    "chart",
    "code",
    "code_caption",
    "doc_title",
    "equation",
    "equation_block",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "image",
    "image_block",
    "image_caption",
    "image_footnote",
    "index",
    "list",
    "list_item",
    "page_footnote",
    "page_number",
    "paragraph_title",
    "phonetic",
    "ref_text",
    "table",
    "table_caption",
    "table_footnote",
    "text",
    "title",
}
_TEXT_LABELS = {
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "caption": DocItemLabel.CAPTION,
    "code": DocItemLabel.CODE,
    "code_caption": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    "footnote": DocItemLabel.FOOTNOTE,
    "formula_number": DocItemLabel.FORMULA,
    "header": DocItemLabel.PAGE_HEADER,
    "image_caption": DocItemLabel.CAPTION,
    "image_footnote": DocItemLabel.FOOTNOTE,
    "index": DocItemLabel.TEXT,
    "page_footnote": DocItemLabel.FOOTNOTE,
    "page_number": DocItemLabel.PAGE_FOOTER,
    "phonetic": DocItemLabel.TEXT,
    "ref_text": DocItemLabel.REFERENCE,
    "table_caption": DocItemLabel.CAPTION,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "text": DocItemLabel.TEXT,
}
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass
class MinerU2Region:
    """One MinerU2 layout region and its optional recognized content."""

    type: str
    bbox: tuple[float, float, float, float]
    angle: int | None = None
    content: str | None = None
    merge_prev: bool = False


@dataclass(frozen=True)
class MinerU2Crop:
    """A recognition crop linked to its source region."""

    region_index: int
    image: PILImage.Image
    prompt: str


class _MinerU2Recognition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    region_index: int
    text: str


class _MinerU2Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    layout: str
    recognition: list[_MinerU2Recognition]


def _normalize_bbox(
    values: tuple[str, str, str, str],
) -> tuple[float, float, float, float] | None:
    coords = tuple(int(value) for value in values)
    if any(coord < 0 or coord > 1000 for coord in coords):
        return None
    x1, y1, x2, y2 = coords
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000)


def _coverage_ratio(inner: MinerU2Region, outer: MinerU2Region) -> float:
    ix1, iy1, ix2, iy2 = inner.bbox
    ox1, oy1, ox2, oy2 = outer.bbox
    intersection_width = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    intersection_height = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    if inner_area == 0:
        return 0.0
    return intersection_width * intersection_height / inner_area


def _filter_table_internal_regions(
    regions: list[MinerU2Region],
) -> list[MinerU2Region]:
    tables = [region for region in regions if region.type == "table"]
    if not tables:
        return regions
    return [
        region
        for region in regions
        if region.type not in {"equation", "equation_block", "text"}
        or not any(_coverage_ratio(region, table) >= 0.9 for table in tables)
    ]


def parse_mineru2_layout(content: str) -> list[MinerU2Region]:
    """Parse MinerU2 native layout output into normalized regions."""
    regions: list[MinerU2Region] = []
    for match in _LAYOUT_PATTERN.finditer(content):
        x1, y1, x2, y2, region_type, rotation_token, tail = match.groups()
        bbox = _normalize_bbox((x1, y1, x2, y2))
        if bbox is None:
            _log.warning(
                "Ignoring MinerU2 region with invalid bbox: %s", match.group(0)
            )
            continue

        region_type = region_type.lower()
        if region_type == "inline_formula":
            continue
        if region_type == "unknown":
            region_type = "image"
        if region_type not in _BLOCK_TYPES:
            _log.warning("Ignoring unknown MinerU2 region type %r", region_type)
            continue

        regions.append(
            MinerU2Region(
                type=region_type,
                bbox=bbox,
                angle=_ROTATIONS.get(rotation_token),
                merge_prev=(region_type == "text" and "txt_contd_tgt" in tail),
            )
        )

    if not regions and content.strip():
        _log.warning("MinerU2 layout output did not contain valid regions")
    return _filter_table_internal_regions(regions)


def prepare_mineru2_layout_image(image: PILImage.Image) -> PILImage.Image:
    """Prepare the square image required by MinerU2 layout detection."""
    return image.convert("RGB").resize(
        MINERU2_LAYOUT_IMAGE_SIZE, PILImage.Resampling.BICUBIC
    )


def _resize_recognition_crop(image: PILImage.Image) -> PILImage.Image:
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > 50:
        width, height = image.size
        if width > height:
            new_size = (width, math.ceil(width / 50))
        else:
            new_size = (math.ceil(height / 50), height)
        padded = PILImage.new(image.mode, new_size, "white")
        padded.paste(
            image,
            ((new_size[0] - width) // 2, (new_size[1] - height) // 2),
        )
        image = padded
    if min(image.size) < 28:
        scale = 28 / min(image.size)
        image = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            PILImage.Resampling.BICUBIC,
        )
    return image


def prepare_mineru2_crops(
    image: PILImage.Image, regions: list[MinerU2Region]
) -> list[MinerU2Crop]:
    """Crop all regions that require the second recognition pass."""
    image = image.convert("RGB")
    crops: list[MinerU2Crop] = []
    for region_index, region in enumerate(regions):
        if region.type in _SKIP_RECOGNITION_TYPES:
            continue
        x1, y1, x2, y2 = region.bbox
        crop = image.crop(
            (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        )
        if crop.width < 1 or crop.height < 1:
            _log.warning("Ignoring empty MinerU2 crop for region %s", region_index)
            continue
        if region.angle in {90, 180, 270}:
            crop = crop.rotate(region.angle, expand=True)
        crops.append(
            MinerU2Crop(
                region_index=region_index,
                image=_resize_recognition_crop(crop),
                prompt=_RECOGNITION_PROMPTS.get(
                    region.type, _DEFAULT_RECOGNITION_PROMPT
                ),
            )
        )
    return crops


def serialize_mineru2_transcript(
    layout: str, recognition: list[tuple[int, str]]
) -> str:
    """Serialize the exact native outputs for one MinerU page."""
    return _MinerU2Transcript(
        layout=layout,
        recognition=[
            _MinerU2Recognition(region_index=region_index, text=text)
            for region_index, text in recognition
        ],
    ).model_dump_json()


def _deserialize_mineru2_transcript(content: str) -> _MinerU2Transcript:
    try:
        return _MinerU2Transcript.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"malformed transcript envelope: {exc}") from exc


def _provenance(
    region: MinerU2Region, original_page_size: Size, page_no: int
) -> ProvenanceItem:
    x1, y1, x2, y2 = region.bbox
    return ProvenanceItem(
        page_no=page_no,
        charspan=(0, len(region.content or "")),
        bbox=BoundingBox(
            l=x1 * original_page_size.width,
            t=y1 * original_page_size.height,
            r=x2 * original_page_size.width,
            b=y2 * original_page_size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )


def parse_mineru2(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
) -> DoclingDocument:
    """Parse a MinerU native-output transcript into a page document."""
    transcript = _deserialize_mineru2_transcript(content)
    regions = parse_mineru2_layout(transcript.layout)
    if transcript.layout.strip() and not regions:
        raise ValueError("non-empty layout output contained no valid regions")

    seen_region_indices: set[int] = set()
    for recognition in transcript.recognition:
        region_index = recognition.region_index
        if region_index < 0 or region_index >= len(regions):
            raise ValueError(f"recognition region index {region_index} is out of range")
        if region_index in seen_region_indices:
            raise ValueError(f"duplicate recognition region index {region_index}")
        seen_region_indices.add(region_index)
        regions[region_index].content = recognition.text

    origin = DocumentOrigin(
        filename=filename, mimetype="application/json", binary_hash=0
    )
    document = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)
    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / original_page_size.width)
    document.add_page(
        page_no=page_no,
        size=original_page_size,
        image=(
            ImageRef.from_pil(image=page_image, dpi=image_dpi)
            if page_image is not None
            else None
        ),
    )

    current_list_group = None
    previous_text_item: TextItem | None = None
    for region in regions:
        provenance = _provenance(region, original_page_size, page_no)
        text = (region.content or "").strip()
        if text == "[Non-Text]" and region.type not in {"footer", "header"}:
            continue

        if region.type == "list_item":
            if current_list_group is None:
                current_list_group = document.add_list_group()
            document.add_list_item(
                text=text,
                orig=region.content or "",
                parent=current_list_group,
                prov=provenance,
            )
            continue

        current_list_group = None
        if region.type == "doc_title":
            document.add_title(text=text, orig=region.content or "", prov=provenance)
        elif region.type in {"paragraph_title", "title"}:
            document.add_heading(
                text=text,
                orig=region.content or "",
                level=1,
                prov=provenance,
            )
        elif region.type == "table":
            _otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)
            document.add_table(
                data=TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells),
                prov=provenance,
            )
        elif region.type in {"chart", "image"}:
            document.add_picture(prov=provenance)
        elif region.type == "equation":
            document.add_text(
                label=DocItemLabel.FORMULA,
                text=text,
                orig=region.content or "",
                prov=provenance,
            )
        elif region.type in {"equation_block", "image_block", "list"}:
            continue
        else:
            if (
                region.type == "text"
                and region.merge_prev
                and previous_text_item is not None
            ):
                separator = "" if _CJK_PATTERN.search(text) else " "
                start = len(previous_text_item.text) + len(separator)
                previous_text_item.text += separator + text
                previous_text_item.orig += separator + (region.content or "")
                provenance.charspan = (start, start + len(text))
                previous_text_item.prov.append(provenance)
                continue

            text_item = document.add_text(
                label=_TEXT_LABELS.get(region.type, DocItemLabel.TEXT),
                text="" if text == "[Non-Text]" else text,
                orig=region.content or "",
                prov=provenance,
            )
            if region.type == "text":
                previous_text_item = text_item
    return document
