# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import IO, Final, cast

from PIL import Image

IMAGE_INPUT_FORMATS: Final[tuple[str, ...]] = (
    "BMP",
    "DIB",
    "GIF",
    "JPEG",
    "PNG",
    "TIFF",
    "WEBP",
)
"""Pillow decoders for the image input formats (``InputFormat.IMAGE``).

``DIB`` is the headerless variant of BMP. The ``JPEG`` decoder also opens
multi-picture (MPO) JPEG files.
"""

RASTER_IMAGE_FORMATS: Final[tuple[str, ...]] = (
    *IMAGE_INPUT_FORMATS,
    "ICO",
    "JPEG2000",
)
"""Pillow decoders for the raster pictures that a document can embed."""

PICTURE_IMAGE_FORMATS: Final[tuple[str, ...]] = (*RASTER_IMAGE_FORMATS, "WMF")
"""Pillow decoders for the pictures of Office documents.

Pillow's ``WMF`` plugin identifies both WMF and EMF metafiles. It draws them on
Windows only and raises ``OSError`` on load elsewhere, so the backends can
rasterize the metafile with LibreOffice instead.
"""

# Windows metafile signatures. A placeable WMF opens with the Aldus magic and an
# EMF carries " EMF" in the dSignature field of its EMR_HEADER, 40 bytes in.
# These are the two Pillow itself identifies.
WMF_PLACEABLE_MAGIC: Final = b"\xd7\xcd\xc6\x9a"
_EMF_SIGNATURE: Final = b" EMF"
_EMF_SIGNATURE_OFFSET: Final = 40


def open_image(
    source: bytes | Path | IO[bytes],
    formats: Sequence[str] = RASTER_IMAGE_FORMATS,
) -> Image.Image:
    """Open an image from a document with an explicit list of Pillow decoders.

    Without ``formats``, Pillow selects the decoder from the first bytes of the
    data, so a picture could select a decoder that starts an external program
    (``EPS`` runs Ghostscript). Use this function for all image data that comes
    from an input document.

    Like ``Image.open``, this reads only the header. The pixels are decoded on
    ``load()``.

    Raises:
        PIL.UnidentifiedImageError: The data is not an image in ``formats``.
        OSError, ValueError: Pillow cannot read the image header.
    """
    fp: Path | IO[bytes] = BytesIO(source) if isinstance(source, bytes) else source
    return Image.open(fp, formats=list(formats))


def is_metafile(image_bytes: bytes) -> bool:
    """Return True when the bytes are a Windows metafile (WMF or EMF).

    Pillow renders metafiles only on Windows, where it delegates to the GDI
    ``PlayEnhMetaFile`` API; the wheels for every other platform carry a stub
    that raises on load. Telling a metafile apart from a genuinely broken
    image decides whether a picture is worth rasterizing externally.

    Args:
        image_bytes: Raw bytes of an embedded picture.

    Returns:
        True when the bytes carry a placeable WMF or an EMF signature.
    """
    return image_bytes[:4] == WMF_PLACEABLE_MAGIC or (
        image_bytes[_EMF_SIGNATURE_OFFSET : _EMF_SIGNATURE_OFFSET + 4] == _EMF_SIGNATURE
    )


# Leading signatures of the formats in RASTER_IMAGE_FORMATS. Each entry is an
# offset and the bytes expected there.
_RASTER_SIGNATURES: Final[tuple[tuple[int, bytes], ...]] = (
    (0, b"\x89PNG\r\n\x1a\n"),
    (0, b"\xff\xd8\xff"),  # JPEG
    (0, b"GIF87a"),
    (0, b"GIF89a"),
    (0, b"BM"),  # BMP
    (0, b"II*\x00"),  # TIFF, little-endian
    (0, b"MM\x00*"),  # TIFF, big-endian
    (8, b"WEBP"),  # after the RIFF header
    (0, b"\x00\x00\x01\x00"),  # ICO
    (0, b"\xff\x4f\xff\x51"),  # JPEG 2000 codestream
    (4, b"jP  "),  # JPEG 2000 file
)


def is_known_picture(image_bytes: bytes) -> bool:
    """Return True when the bytes start like a picture format that Docling decodes.

    Pillow raises ``UnidentifiedImageError`` also for a supported picture whose
    header it cannot parse, for example a PNG with a truncated chunk in Pillow 12.
    Such a picture can still be worth rasterizing externally, while data with no
    known signature (for example PostScript) cannot.

    Args:
        image_bytes: Raw bytes of an embedded picture.

    Returns:
        True for a metafile or for bytes with a raster signature.
    """
    return is_metafile(image_bytes) or any(
        image_bytes[offset : offset + len(magic)] == magic
        for offset, magic in _RASTER_SIGNATURES
    )


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


def _max_value(image: Image.Image) -> float:
    """Return the largest sample of a single-band image."""
    _, maximum = cast("tuple[float, float]", image.getextrema())
    return maximum


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to RGB, flattening any transparency onto white.

    A plain ``convert("RGB")`` drops the alpha channel and keeps the colour
    stored under it, which is usually black for fully transparent pixels. Dark
    content on a transparent background would then become a black image.
    Palette images with ``tRNS`` transparency also go through RGBA, which
    avoids the Pillow warning for transparency expressed in bytes.
    Integer images are scaled from the 16-bit range to 8 bits before the
    conversion (the value is divided by 256 and truncated, so 65535 becomes
    255). The ``I;16*`` modes are always scaled. Mode ``I`` is a 32-bit
    container, so it is scaled only when it holds a value above 255. This keeps
    data that is already in the 8-bit range unchanged.
    """
    if image.mode.startswith("I;16") or (image.mode == "I" and _max_value(image) > 255):
        image = image.convert("I").point(lambda value: value / 256)
        transparency = image.info.get("transparency")
        if isinstance(transparency, int):
            image.info["transparency"] = transparency // 256
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")
