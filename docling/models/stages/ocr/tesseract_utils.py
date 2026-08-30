# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tessdata names, orientation and box geometry, shared by both Tesseract models.

`tesseract_ocr_model.py` (the tesserocr bindings) and `tesseract_ocr_cli_model.py`
drive the same engine through different front-ends, so everything that is about
Tesseract itself rather than about either front-end lives here -- the same reason
`ppocr_languages.py` sits beside the two engines that speak PP-OCR.
"""

from collections.abc import Sequence
from typing import Optional, Tuple

import langcodes
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)
from docling.utils.orientation import CLIPPED_ORIENTATIONS, rotate_bounding_box


def map_tesseract_script(script: str) -> str:
    r"""Map an OSD-reported script name onto its tessdata `script/` file name."""
    if script == "Katakana" or script == "Hiragana":
        script = "Japanese"
    elif script == "Han":
        script = "HanS"
    elif script == "Korean":
        script = "Hangul"
    return script


# Canonical tag -> tessdata language file, where Tesseract deviates from
# ISO 639-2/T. Everything else is handled by `.to_alpha3(variant="T")`.
_TESSERACT_LANGUAGE_NAMES: dict[str, str] = {
    "zh-Hans": "chi_sim",
    "zh-Hant": "chi_tra",
    "sr-Cyrl": "srp",
    "sr-Latn": "srp_latn",
    "az-Cyrl": "aze_cyrl",
    "az-Latn": "aze",
    "uz-Cyrl": "uzb_cyrl",
    "uz-Latn": "uzb",
    "ku-Latn": "kmr",
    "nb-Latn": "nor",
    "nn-Latn": "nor",
    "no-Latn": "nor",
    # Fraktur has its own traineddata; `to_alpha3()` would flatten it to `deu`.
    "de-Latf": "deu_latf",
}

_TESSERACT_TO_CANONICAL: dict[str, str] = {
    name: tag for tag, name in _TESSERACT_LANGUAGE_NAMES.items()
}


def map_tesseract_language(language: OcrLanguage, script_prefix: str) -> str | None:
    """Map a canonical tag onto a tessdata language file name.

    A passthrough names a traineddata file directly: `script/Cyrillic`. It is
    always written with the `script/` prefix, but older tessdata installs list
    the script files unprefixed, so the prefix is re-applied from what this
    install actually reports. `mul` has no file of its own.
    """
    if language.is_passthrough:
        assert language.native is not None
        name = language.native.removeprefix(
            OcrLanguageResolver.TESSERACT_SCRIPT_FILE_PREFIX
        )
        return f"{script_prefix}{name}"
    if language.is_multilingual:
        return None
    if language.tag in _TESSERACT_LANGUAGE_NAMES:
        return _TESSERACT_LANGUAGE_NAMES[language.tag]
    # Tesseract's vocabulary *is* ISO 639-2/T: deu, fra, ell, ces, kat.
    assert language.bcp47_language is not None
    return langcodes.Language.get(language.bcp47_language).to_alpha3(variant="T")


def installed_language_tags(
    names: Sequence[str], script_prefix: str, kind: str
) -> list[str]:
    """The canonical tags this install can actually serve.

    `kind` is the calling engine's `OcrOptions.kind`: the two Tesseract bindings
    read the same tessdata names, and either one selects that vocabulary.

    A name is reported only if the tag it renders as maps back to a file that is
    installed. Without that round trip the list can offer a tag this install
    cannot load: `frk` and `deu_latf` are both `de-Latf`, but only `deu_latf` is
    what the tag maps to, so an install carrying just `frk` would advertise a
    language it then refuses.
    """
    tags = set()
    for name in names:
        if name in _TESSERACT_TO_CANONICAL:
            token = _TESSERACT_TO_CANONICAL[name]
        elif script_prefix and name.startswith(script_prefix):
            # A script traineddata file is named back as itself: that is what the
            # user has to type to select it -- except the vertical-text ones, which
            # `canonicalize_ocr_language` refuses, so naming them back would
            # advertise a value that cannot be asked for.
            if name.lower().endswith("_vert"):
                continue
            token = name
        elif not script_prefix and name in OcrLanguageResolver.TESSERACT_SCRIPT_FILES:
            # This install lists its script packs unprefixed, but `script/<Name>`
            # is still the spelling that selects one. Left bare, the name is either
            # dropped as unparseable (`Latin`) or read as a language (`Lao`).
            token = f"{OcrLanguageResolver.TESSERACT_SCRIPT_FILE_PREFIX}{name}"
        else:
            token = name
        language = OcrLanguageResolver.canonicalize_ocr_language(
            token, kind, raise_exception=False
        )
        if language is None:
            continue
        if map_tesseract_language(language, script_prefix) in names:
            tags.add(language.tag)
    return sorted(tags)


def parse_tesseract_orientation(orientation: str) -> int:
    # Tesseract orientation is [0, 90, 180, 270] clockwise, bounding rectangle angles
    # are [0, 360[ counterclockwise
    parsed = int(orientation)
    if parsed not in CLIPPED_ORIENTATIONS:
        msg = (
            f"invalid tesseract document orientation {orientation}, "
            f"expected orientation: {sorted(CLIPPED_ORIENTATIONS)}"
        )
        raise ValueError(msg)
    parsed = -parsed
    parsed %= 360
    return parsed


def tesseract_box_to_bounding_rectangle(
    bbox: BoundingBox,
    *,
    original_offset: Optional[BoundingBox] = None,
    scale: float,
    orientation: int,
    im_size: Tuple[int, int],
) -> BoundingRectangle:
    # box is in the top, left, height, width format, top left coordinates
    rect = rotate_bounding_box(bbox, angle=orientation, im_size=im_size)
    rect = BoundingRectangle(
        r_x0=rect.r_x0 / scale,
        r_y0=rect.r_y0 / scale,
        r_x1=rect.r_x1 / scale,
        r_y1=rect.r_y1 / scale,
        r_x2=rect.r_x2 / scale,
        r_y2=rect.r_y2 / scale,
        r_x3=rect.r_x3 / scale,
        r_y3=rect.r_y3 / scale,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    if original_offset is not None:
        if original_offset.coord_origin is not CoordOrigin.TOPLEFT:
            msg = f"expected coordinate origin to be {CoordOrigin.TOPLEFT.value}"
            raise ValueError(msg)
        if original_offset is not None:
            rect.r_x0 += original_offset.l
            rect.r_x1 += original_offset.l
            rect.r_x2 += original_offset.l
            rect.r_x3 += original_offset.l
            rect.r_y0 += original_offset.t
            rect.r_y1 += original_offset.t
            rect.r_y2 += original_offset.t
            rect.r_y3 += original_offset.t
    return rect
