"""Shared VLM utility functions for output post-processing and image sizing."""

from __future__ import annotations

import math

from docling_core.types.doc import Size


def strip_stop_strings(texts: list[str], stop_strings: list[str]) -> list[str]:
    """Strip stop strings and their partial trailing prefixes from decoded texts.

    For each text, removes the first occurrence of any stop string and everything
    after it. If no full stop string is found, removes any trailing partial prefix
    of a stop string (e.g. "<|im_" from an incomplete "<|im_end|>").
    """
    cleaned = []
    for text in texts:
        for ss in stop_strings:
            idx = text.find(ss)
            if idx != -1:
                text = text[:idx]
            else:
                for k in range(len(ss) - 1, 0, -1):
                    if text.endswith(ss[:k]):
                        text = text[:-k]
                        break
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
