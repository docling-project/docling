# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from PIL import Image


def _to_rgb(frame: Image.Image) -> Image.Image:
    """Convert a frame to RGB, flattening any transparency onto white.

    A plain ``convert("RGB")`` drops the alpha channel and keeps the colour
    stored under it, which is usually black for fully transparent pixels. Dark
    content on a transparent background would then become a black page.
    """
    if frame.mode in ("RGBA", "LA", "PA") or "transparency" in frame.info:
        rgba = frame.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return frame.convert("RGB")
