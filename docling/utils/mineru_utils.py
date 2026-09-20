# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Utilities for MinerU2 two-step document parsing."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from itertools import groupby

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
    "index": DocItemLabel.DOCUMENT_INDEX,
    "page_footnote": DocItemLabel.FOOTNOTE,
    "page_number": DocItemLabel.PAGE_FOOTER,
    "phonetic": DocItemLabel.TEXT,
    "ref_text": DocItemLabel.REFERENCE,
    "table_caption": DocItemLabel.CAPTION,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "text": DocItemLabel.TEXT,
}
_CONTENT_TOKENS = {"ched", "ecel", "fcel", "rhed", "srow"}
_OTSL_TAG_PATTERN = re.compile(
    r"<(?P<tag>[a-z]+)>(?P<text>.*?)</(?P=tag)>"
    r"|<(?P<stag>[a-z]+)\s*/>"
    r"|<(?P<otag>[a-z]+)>(?P<otext>[^<]*)",
    re.DOTALL,
)


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


def serialize_mineru2_regions(regions: list[MinerU2Region]) -> str:
    """Serialize completed MinerU2 regions for storage in ``VlmPrediction``."""
    return json.dumps([asdict(region) for region in regions], ensure_ascii=False)


def _regions_from_json(content: str) -> list[MinerU2Region]:
    try:
        values = json.loads(content)
    except json.JSONDecodeError as exc:
        _log.warning("Failed to parse MinerU2 JSON: %s", exc)
        return []
    if not isinstance(values, list):
        _log.warning("Expected MinerU2 JSON array, got %s", type(values).__name__)
        return []

    regions = []
    for value in values:
        if not isinstance(value, dict):
            continue
        region_type = value.get("type")
        bbox = value.get("bbox")
        if region_type not in _BLOCK_TYPES or not isinstance(bbox, (list, tuple)):
            continue
        if len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(coord) for coord in bbox)
        except (TypeError, ValueError):
            continue
        normalized_bbox = (x1, y1, x2, y2)
        if not (
            0 <= normalized_bbox[0] < normalized_bbox[2] <= 1
            and 0 <= normalized_bbox[1] < normalized_bbox[3] <= 1
        ):
            continue
        angle = value.get("angle")
        if angle not in {None, 0, 90, 180, 270}:
            angle = None
        raw_content = value.get("content")
        regions.append(
            MinerU2Region(
                type=region_type,
                bbox=normalized_bbox,
                angle=angle,
                content=raw_content if isinstance(raw_content, str) else None,
                merge_prev=value.get("merge_prev") is True,
            )
        )
    return regions


def _parse_otsl_table(content: str) -> TableData:
    token_pairs: list[tuple[str, str]] = []
    for match in _OTSL_TAG_PATTERN.finditer(content):
        if match.group("tag"):
            token_pairs.append((match.group("tag"), match.group("text") or ""))
        elif match.group("stag"):
            token_pairs.append((match.group("stag"), ""))
        elif match.group("otag"):
            token_pairs.append((match.group("otag"), match.group("otext") or ""))
    rows = [
        list(group)
        for is_newline, group in groupby(
            token_pairs, key=lambda token: token[0] == "nl"
        )
        if not is_newline
    ]
    if not rows:
        return TableData(num_rows=0, num_cols=0, table_cells=[])

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)
    grid = [row + [("", "")] * (num_cols - len(row)) for row in rows]
    cells: list[TableCell] = []
    for row_index, row in enumerate(grid):
        for col_index, (tag, text) in enumerate(row):
            if tag not in _CONTENT_TOKENS:
                continue
            col_span = 1
            for following_col in range(col_index + 1, num_cols):
                if grid[row_index][following_col][0] not in {"lcel", "xcel"}:
                    break
                col_span += 1
            row_span = 1
            for following_row in range(row_index + 1, num_rows):
                if grid[following_row][col_index][0] not in {"ucel", "xcel"}:
                    break
                row_span += 1
            cells.append(
                TableCell(
                    text=text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                    column_header=tag == "ched",
                    row_header=tag == "rhed",
                    row_section=tag == "srow",
                )
            )
    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


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
    """Parse serialized MinerU2 regions into a page ``DoclingDocument``."""
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
    for region in _regions_from_json(content):
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
            document.add_table(data=_parse_otsl_table(text), prov=provenance)
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
            document.add_text(
                label=_TEXT_LABELS.get(region.type, DocItemLabel.TEXT),
                text="" if text == "[Non-Text]" else text,
                orig=region.content or "",
                prov=provenance,
            )
    return document
