# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Shared VLM utility functions for output post-processing and image sizing."""

from __future__ import annotations

import math
import re

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    FloatingItem,
    SectionHeaderItem,
    Size,
    TextItem,
)

_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.*?)\s*$", re.DOTALL)


def link_adjacent_captions(
    doc: DoclingDocument,
    block_starts: list[int],
    owner_labels: dict[int, set[DocItemLabel]],
) -> None:
    """Link native caption blocks with exactly one compatible adjacent owner.

    Block boundaries include skipped output, so missing content cannot create
    adjacency. References preserve the original tree order and caption provenance.
    """
    ends = [*block_starts[1:], len(doc.body.children)]
    blocks = [
        [ref.resolve(doc) for ref in doc.body.children[start:end]]
        for start, end in zip(block_starts, ends)
    ]
    for index, labels in owner_labels.items():
        captions = blocks[index]
        # Core's caption head holds one text item and does not read child runs.
        if (
            len(captions) != 1
            or not isinstance(caption := captions[0], TextItem)
            or caption.label != DocItemLabel.CAPTION
            or not caption.text.strip()
            or caption.children
        ):
            continue
        owners = [
            block[0]
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(blocks)
            and len(block := blocks[neighbor]) == 1
            and isinstance(block[0], FloatingItem)
            and block[0].label in labels
        ]
        # ponytail: immediate neighbors only; ambiguous owners stay unlinked.
        # Our minimum Core cannot serialize code caption heads faithfully.
        if (
            len(owners) == 1
            and owners[0].label != DocItemLabel.CODE
            and not owners[0].captions
        ):
            owners[0].captions.append(caption.get_ref())


def parse_markdown_heading(text: str) -> tuple[str, int | None]:
    """Strip an optional Markdown heading marker and return its depth."""
    match = _MARKDOWN_HEADING_PATTERN.match(text)
    if match is None:
        return text, None
    return match.group(2), len(match.group(1))


def normalize_section_header_levels(doc: DoclingDocument) -> None:
    """Turn the raw heading depths of a whole document into section levels.

    VLM mappers store a section header with N Markdown ``#`` or HTML ``<hN>``
    at level N, and one without markup at level 2 (like ``##``). Docling exports
    section level N one step deeper, below the title. If no section header has
    level 1, all levels go up one step, so exports keep the source depth.
    Otherwise nothing changes, so two source depths never share one level.
    The layout label alone decides title versus section header. Call this once
    per document, after pages are joined, so a depth maps to one level on all
    pages. ``doc.texts`` holds every text item, in every layer and container.
    """
    headers = [item for item in doc.texts if isinstance(item, SectionHeaderItem)]
    if all(item.level > 1 for item in headers):
        for item in headers:
            item.level -= 1


# `\(` and `\)` are not math symbols, so in a formula they are delimiters; `\\` stays.
_INLINE_MATH_DELIMITER = re.compile(r"\\\\|\\[()]")
_OUTER_MATH = re.compile(
    r"\$\$((?:(?!\$\$).)*)\$\$|\\\[((?:(?!\\[\[\]]).)*)\\\]|\$([^$]*)\$", re.DOTALL
)


def strip_math_delimiters(text: str) -> str:
    """Formula LaTeX without math delimiters; the serializer adds its own.

    VLMs can wrap each segment in ``\\(...\\)``, often with text between them
    (``(6) \\(x_1=a\\), \\(x_2=b\\)``), and leave single ones in truncated output.
    """
    text = _INLINE_MATH_DELIMITER.sub(
        lambda match: match.group() if match.group() == "\\\\" else "", text
    ).strip()
    while (outer := _OUTER_MATH.fullmatch(text)) is not None:
        text = next(group for group in outer.groups() if group is not None).strip()
    return text


def strip_stop_strings(texts: list[str], stop_strings: list[str]) -> list[str]:
    """Strip stop strings from decoded texts.

    For each text, removes the first occurrence of any full stop string and
    everything after it.
    """
    cleaned = []
    for text in texts:
        for ss in stop_strings:
            idx = text.find(ss)
            if idx != -1:
                text = text[:idx]
        cleaned.append(text)
    return cleaned


def strip_trailing_token(texts: list[str], token: str) -> list[str]:
    """Remove all trailing occurrences of ``token`` from each text.

    Unlike ``str.rstrip``, which strips any trailing character contained in its
    argument, this only removes the token as a whole string.
    """
    cleaned = []
    for text in texts:
        while text.endswith(token):
            text = text[: -len(token)]
        cleaned.append(text)
    return cleaned


def compute_qwen2vl_image_size(
    width: int,
    height: int,
    scale: float = 1.0,
    max_size: int | None = None,
    factor: int = 28,
    min_pixels: int = 200704,
    max_pixels: int = 2_500_000,
) -> Size:
    """Compute the actual image resolution after Qwen2.5-VL smart_resize.

    Replicates the Qwen2VL preprocessor logic: scale the image, optionally
    clamp to max_size, then round dimensions to factor and clamp total pixels
    to [min_pixels, max_pixels].

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        scale: Scale factor applied before resize.
        max_size: Optional max dimension (longest side) clamp.
        factor: Patch size * merge size (default 28 for Qwen2.5-VL).
        min_pixels: Minimum total pixel budget.
        max_pixels: Maximum total pixel budget.

    Returns:
        Size with the computed width and height.
    """
    mw = int(width * scale)
    mh = int(height * scale)

    if max_size is not None:
        max_dim = max(mw, mh)
        if max_dim > max_size:
            sf = max_size / max_dim
            mw = int(mw * sf)
            mh = int(mh * sf)

    h_bar = round(mh / factor) * factor
    w_bar = round(mw / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((mh * mw) / max_pixels)
        h_bar = max(factor, math.floor(mh / beta / factor) * factor)
        w_bar = max(factor, math.floor(mw / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (mh * mw))
        h_bar = math.ceil(mh * beta / factor) * factor
        w_bar = math.ceil(mw * beta / factor) * factor

    return Size(width=w_bar, height=h_bar)
