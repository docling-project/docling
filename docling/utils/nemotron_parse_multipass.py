# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Crop geometry and conservative refinement for Nemotron 3.5 Parse.

Uses NVIDIA's documented content-normalized coordinates, crop margins and
union thresholds. Only valid, compatible structural blocks can replace a
page prediction. The page block keeps its position and bounding box.
"""

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO

from PIL import Image

from docling.utils.nemotron_parse_v3_5_utils import (
    NemotronParseBlock,
    _parse_latex_cell,
    _split_top_level,
    extract_nemotron_parse_v3_5_blocks,
)

NEMOTRON_DETECT_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>"
)
_STRUCTURAL_LABELS = {"Table", "Chart"}
_TABULAR = re.compile(r"\\begin\{tabular\}\{([clrx|\s]+)\}(.*?)\\end\{tabular\}", re.S)


@dataclass(frozen=True)
class NemotronParseCrop:
    image: Image.Image
    region: NemotronParseBlock
    # Normalized crop box, including the margin, in the original page frame.
    bbox: tuple[float, float, float, float]


def _intersection(a: NemotronParseBlock, b: NemotronParseBlock) -> float:
    return max(0.0, min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])) * max(
        0.0, min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    )


def _iou(a: NemotronParseBlock, b: NemotronParseBlock) -> float:
    intersection = _intersection(a, b)
    area = sum((r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) for r in (a, b))
    return intersection / (area - intersection) if area > intersection else 0.0


def structural_regions(content: str) -> list[NemotronParseBlock]:
    """Read parser and no-text detector blocks, rejecting invalid rectangles."""
    regions = []
    for block in extract_nemotron_parse_v3_5_blocks(content):
        x1, y1, x2, y2 = block.bbox
        if (
            block.label in _STRUCTURAL_LABELS
            and 0 <= x1 < x2 <= 1
            and 0 <= y1 < y2 <= 1
        ):
            regions.append(block)
    return regions


def prepare_nemotron_crops(
    image: Image.Image,
    parser_content: str,
    detection_content: str | None = None,
    *,
    union: bool = True,
) -> list[NemotronParseCrop]:
    """Select at most 80 Table/Chart regions; pad and JPEG round-trip each crop."""
    primary = structural_regions(
        detection_content if detection_content is not None else parser_content
    )
    regions: list[NemotronParseBlock] = []
    for region in primary:
        if not any(
            region.label == other.label and _iou(region, other) >= 0.85
            for other in regions
        ):
            regions.append(region)
    if detection_content is not None and union:
        for region in structural_regions(parser_content):
            if not any(
                region.label == other.label and _iou(region, other) >= 0.5
                for other in regions
            ):
                regions.append(region)
    crops = []
    for region in regions[:80]:
        x1, y1, x2, y2 = region.bbox
        mx = 12 + (x2 - x1) * image.width * 0.04
        my = 12 + (y2 - y1) * image.height * 0.04
        pixels = (
            max(0, math.floor(x1 * image.width - mx)),
            max(0, math.floor(y1 * image.height - my)),
            min(image.width, math.ceil(x2 * image.width + mx)),
            min(image.height, math.ceil(y2 * image.height + my)),
        )
        left, top, right, bottom = pixels
        if right - left < 8 or bottom - top < 8:
            continue
        stream = BytesIO()
        image.crop(pixels).convert("RGB").save(stream, format="JPEG", quality=95)
        stream.seek(0)
        with Image.open(stream) as decoded:
            crops.append(
                NemotronParseCrop(
                    image=decoded.convert("RGB"),
                    region=region,
                    bbox=(
                        left / image.width,
                        top / image.height,
                        right / image.width,
                        bottom / image.height,
                    ),
                )
            )
    return crops


def _balanced_braces(text: str) -> bool:
    depth = 0
    for token in re.findall(r"\\.|[{}]", text):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _table_columns(text: str) -> int | None:
    """Validate the native grid, including multirow placeholders and spans."""
    matches = list(_TABULAR.finditer(text))
    if (
        not _balanced_braces(text)
        or len(matches) != 1
        or text.count(r"\begin{tabular}") != text.count(r"\end{tabular}")
        or text.count(r"\begin{tabular}") != 1
    ):
        return None
    match = matches[0]
    columns = sum(char in "clrx" for char in match[1])
    rows = [row for row in _split_top_level(match[2], r"\\") if row.strip()]
    if not rows or len(rows) > 4096 or not 0 < columns <= 512:
        return None
    occupied: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        column = 0
        for raw in _split_top_level(row, "&"):
            # The renderer repairs bad spans. A crop must pass a stricter guard.
            spans = re.findall(r"\\(?:multirow|multicolumn)\{([^{}]*)\}", raw)
            if any(not count.strip().isdigit() or int(count) <= 0 for count in spans):
                return None
            cell = _parse_latex_cell(raw)
            if r"\multirow" in cell.text or r"\multicolumn" in cell.text:
                return None
            if (
                row_index + cell.row_span > len(rows)
                or column + cell.col_span > columns
            ):
                return None
            positions = {
                (r, c)
                for r in range(row_index, row_index + cell.row_span)
                for c in range(column, column + cell.col_span)
            }
            if positions & occupied:
                if (
                    cell.text not in ("", r"\mbox{}")
                    or cell.header
                    or cell.row_span != 1
                    or not positions <= occupied
                ):
                    return None
            else:
                occupied.update(positions)
            column += cell.col_span
        if column != columns:
            return None
    return columns


def _valid_structure(block: NemotronParseBlock) -> bool:
    if block.label == "Table":
        return _table_columns(block.text) is not None
    # A chart needs a closed header and at least one rectangular data table.
    if (
        len(re.findall(r"<chart\b[^>]*>", block.text)) != 1
        or block.text.count("</chart>") != 1
    ):
        return False
    rows = [
        line.strip() for line in block.text.splitlines() if line.strip().startswith("|")
    ]
    widths = [len(row.strip("|").split("|")) for row in rows]
    return len(rows) >= 3 and len(set(widths)) == 1 and widths[0] >= 2


def remap_nemotron_crop(
    content: str, crop: NemotronParseCrop
) -> list[NemotronParseBlock]:
    left, top, right, bottom = crop.bbox
    candidates = []
    for block in structural_regions(content):
        x1, y1, x2, y2 = block.bbox
        mapped = NemotronParseBlock(
            label=block.label,
            bbox=(
                left + x1 * (right - left),
                top + y1 * (bottom - top),
                left + x2 * (right - left),
                top + y2 * (bottom - top),
            ),
            text=block.text,
        )
        if (
            mapped.label == crop.region.label
            and _iou(mapped, crop.region) >= 0.5
            and _valid_structure(mapped)
        ):
            candidates.append(mapped)
    return candidates


def merge_nemotron_crops(content: str, candidates: list[NemotronParseBlock]) -> str:
    """Guard replacement and additions; retain the original reading order."""
    blocks = extract_nemotron_parse_v3_5_blocks(content)
    if not blocks or not candidates:
        return content
    used: set[int] = set()
    changed = False
    for index, original in enumerate(blocks):
        matches = []
        for candidate_index, candidate in enumerate(candidates):
            if (
                candidate_index in used
                or candidate.label != original.label
                or not _valid_structure(candidate)
            ):
                continue
            overlap = _iou(candidate, original)
            if overlap < 0.5:
                continue
            old_columns = (
                _table_columns(original.text) if original.label == "Table" else None
            )
            if old_columns is not None and old_columns != _table_columns(
                candidate.text
            ):
                continue
            if (
                SequenceMatcher(
                    None, original.text, candidate.text, autojunk=False
                ).ratio()
                >= 0.65
            ):
                matches.append((overlap, candidate_index, candidate))
        if matches:
            _, candidate_index, candidate = max(matches, key=lambda item: item[:2])
            blocks[index] = NemotronParseBlock(
                original.label, original.bbox, candidate.text
            )
            used.add(candidate_index)
            changed = True
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in used or not _valid_structure(candidate):
            continue
        if any(
            candidate.label == block.label and _intersection(candidate, block) > 0
            for block in blocks
        ):
            continue
        # Insert new detector objects without sorting or moving the page blocks.
        position = next(
            (
                i
                for i, block in enumerate(blocks)
                if (block.bbox[1], block.bbox[0])
                > (candidate.bbox[1], candidate.bbox[0])
            ),
            len(blocks),
        )
        blocks.insert(position, candidate)
        changed = True
    if not changed:
        return content
    return "\n".join(
        f"<x_{b.bbox[0]:.8f}><y_{b.bbox[1]:.8f}>{b.text}<x_{b.bbox[2]:.8f}><y_{b.bbox[3]:.8f}><class_{b.label}>"
        for b in blocks
    )
