# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from PIL import Image


def check_image_pixels(size: tuple[int, int], max_pixels: int) -> None:
    """Raise ``ValueError`` if an image of ``size`` exceeds ``max_pixels``.

    Pillow reads the dimensions from the image header, so call this after
    ``Image.open`` (or ``seek``) and before the pixel data is decoded.
    """
    width, height = size
    if width * height > max_pixels:
        raise ValueError(
            f"Image of {width}x{height} pixels exceeds the limit of "
            f"{max_pixels} pixels."
        )


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to RGB, flattening any transparency onto white.

    A plain ``convert("RGB")`` drops the alpha channel and keeps the colour
    stored under it, which is usually black for fully transparent pixels. Dark
    content on a transparent background would then become a black image.
    Palette images with ``tRNS`` transparency also go through RGBA, which
    avoids the Pillow warning for transparency expressed in bytes.
    Integer images hold 16-bit samples, which ``convert("RGB")`` clips to white
    instead of scaling, so they are scaled to 8 bits first.
    """
    if image.mode.startswith("I"):
        image = image.convert("I").point(lambda value: value / 256)
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")
