# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tessdata names, orientation and box geometry, shared by both Tesseract models.

`tesseract_ocr_model.py` (the tesserocr bindings) and `tesseract_ocr_cli_model.py`
drive the same engine through different front-ends, so everything that is about
Tesseract itself rather than about either front-end lives here.
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

# Tessdata files that are not recognizers: `osd` is the orientation-and-script
# detector, `equ` the equation model. Neither is a language anyone can ask for.
_NON_LANGUAGE_TRAINEDDATA = frozenset({"osd", "equ"})


# Canonical tag -> tessdata file, where Tesseract deviates from ISO 639-2/T
# Everything else is handled by `.to_alpha3(variant="T")`.
_TESSERACT_DEVIATIONAL_CODES: dict[str, str] = {
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

_DEVIATIONAL_CODE_TO_CANONICAL: dict[str, str] = {
    name: tag for tag, name in _TESSERACT_DEVIATIONAL_CODES.items()
}


# Prefix of the tessdata script-family files, e.g. `script/Cyrillic`. The bare
# script name is deliberately *not* accepted as a language: `Lao` is a tessdata
# script file and also a valid BCP-47 primary subtag.
_TESSERACT_SCRIPT_FILE_PREFIX = "script/"


def tesseract_vocabulary(codes: Sequence[str]) -> list[str]:
    r"""The traineddata names an install reports, normalized.

    Tesseract spells script packs with the OS path separator, so on Windows it
    reports `script\Latin`. Both front-ends see it: `tesseract --list-langs`
    prints `GetAvailableLanguagesAsVector()` and `tesserocr.get_languages()`
    calls the same API. The forward-slash form is what `tesseract -l` expects on
    every platform, and the only one `_sanitize_lang` accepts.
    """
    return [str(code).replace("\\", "/") for code in codes]


def osd_script_to_tesseract_code(script: str) -> str:
    """The tessdata file an OSD-detected script selects: `Katakana` -> `script/Japanese`.

    OSD reports a script, never a language, out of the fixed set its traineddata
    was built on, and every one of those names has a `script/` file once the few
    that are not spelled like their file are folded onto it.
    """
    if script == "Katakana" or script == "Hiragana":
        script = "Japanese"
    elif script == "Han":
        script = "HanS"
    elif script == "Korean":
        script = "Hangul"
    return f"{_TESSERACT_SCRIPT_FILE_PREFIX}{script}"


def language_to_tesseract_code(language: OcrLanguage) -> str | None:
    """Map an OcrLanguage object to a tesseract code"""
    if language.native is not None:
        return language.native
    if language.is_multilingual:
        return None
    if language.bcp47 in _TESSERACT_DEVIATIONAL_CODES:
        return _TESSERACT_DEVIATIONAL_CODES[language.bcp47]
    # Tesseract's vocabulary *is* ISO 639-2/T: deu, fra, ell, ces, kat.
    assert language.bcp47_language is not None
    return langcodes.Language.get(language.bcp47_language).to_alpha3(variant="T")


def installed_tesseract_tags(codes: Sequence[str]) -> list[str]:
    """
    The tags this install can serve. This can be either a canonical BCP47 tag or a native tesseract
    """
    tags = set()
    for code in codes:
        if code in _NON_LANGUAGE_TRAINEDDATA:
            continue
        # First resolve against the deviational codes
        tag = _DEVIATIONAL_CODE_TO_CANONICAL.get(code, code)

        # Try to canonicalize or receive a None language
        language = OcrLanguageResolver.canonicalize_ocr_language(
            tag, raise_exception=False
        )
        # Check if it is a native code
        if language is None or language_to_tesseract_code(language) != code:
            language = OcrLanguage(native=code)
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
