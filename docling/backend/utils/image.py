# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

from PIL import Image, ImageCms

_PNG_MODES = frozenset({"1", "L", "LA", "I", "I;16", "I;16B", "P", "RGB", "RGBA"})


def normalize_image_for_png(image: Image.Image) -> Image.Image:
    """Convert unsupported modes to RGB or RGBA for PNG serialization.

    Keep PNG-compatible images unchanged, including their transparency. Use
    RGBA for PA images to preserve per-pixel alpha. For other unsupported modes,
    use the embedded ICC profile or Pillow's default RGB conversion if the
    profile is absent or unusable.
    Image decoding and default conversion errors are left to the caller.
    """
    if image.mode in _PNG_MODES:
        return image

    if image.mode == "PA":
        return image.convert("RGBA")

    profile = image.info.get("icc_profile")
    if profile:
        try:
            return ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except (ImageCms.PyCMSError, OSError, ValueError):
            # A bad or incompatible profile must not cause a valid image to be lost.
            pass

    return image.convert("RGB")
