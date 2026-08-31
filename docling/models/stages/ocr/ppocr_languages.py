# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Canonical BCP-47 to PP-OCR recognizer codes.

RapidOCR and the KServe v2 OCR client both address PP-OCR recognizers by the
same codes, so the mapping lives here rather than in either engine. RapidOCR
consults the installed `rapidocr` package for the authoritative PP-OCRv6 set and
falls back to the static copy below; the KServe client uses the static copy only,
so it never has to import `rapidocr`.

The static code sets mirror the PP-OCR release notes summarised in
`docs/concepts/OCR.md`. They can drift from a newer `rapidocr`; this module is
their single owner.
"""

from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)

# Recognition languages served by the PP-OCRv4 backbone (the torch fallback).
PPOCRV4_CODES = frozenset(
    {"arabic", "cyrillic", "devanagari", "ka", "korean", "latin", "ta", "te"}
)

# Recognition languages served by the PP-OCRv5 backbone.
PPOCRV5_CODES = frozenset(
    {
        "arabic",
        "ch",
        "cyrillic",
        "devanagari",
        "el",
        "en",
        "eslav",
        "korean",
        "latin",
        "ta",
        "te",
        "th",
    }
)

# Static copy of the PP-OCRv6 recognition languages. RapidOCR prefers the set
# exported by the installed package; this is the offline/KServe fallback.
PPOCRV6_CODES = frozenset(
    {
        "af", "az", "bs", "ca", "ch", "chinese_cht", "cs", "cy", "da", "de",
        "en", "es", "et", "eu", "fi", "fr", "french", "ga", "german", "gl",
        "hr", "hu", "id", "is", "it", "japan", "ku", "la", "lb", "lt", "lv",
        "mi", "ms", "mt", "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro",
        "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi",
    }
)  # fmt: skip

# PP-OCR code used when `lang` is left empty: the engine's own default
# recognizer, which is Simplified Chinese.
PPOCR_DEFAULT_CODE = "ch"

# Canonical tag -> PP-OCR code, for the languages whose code is not simply the
# primary subtag. `None` marks a tag that must *not* fall through to the generic
# rules below, because the code that looks right means something else.
_CANONICAL_TO_CODE: dict[str, str | None] = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "ja-Jpan": "japan",
    "ko-Kore": "korean",
    "sr-Latn": "rs_latin",
    # `tl` is PP-OCR's code; BCP-47 canonicalizes Tagalog to `fil`.
    "fil-Latn": "tl",
    # PP-OCR serves East Slavic with a narrower recognizer than `cyrillic`.
    "ru-Cyrl": "eslav",
    "uk-Cyrl": "eslav",
    "be-Cyrl": "eslav",
    # PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.
    "kn-Knda": "ka",
    "ka-Geor": None,
}

# ISO 15924 script -> PP-OCR script-family code. Internal routing only: users
# name a language and this finds the script-wide recognizer that covers it, for
# the many languages PP-OCR serves no other way.
_SCRIPT_TO_CODE: dict[str, str] = {
    "Latn": "latin",
    "Cyrl": "cyrillic",
    "Arab": "arabic",
    "Deva": "devanagari",
}

# Reverse of the language table, for rendering a vocabulary back as tags. The
# script recognizers are not reversed: they are named back as themselves, which
# is what the user types to select one.
_CODE_TO_CANONICAL: dict[str, list[str]] = {}
for _tag, _token in _CANONICAL_TO_CODE.items():
    if _token is not None:
        _CODE_TO_CANONICAL.setdefault(_token, []).append(_tag)

# PP-OCRv6 codes that duplicate a language already reachable by its subtag.
_REDUNDANT_CODES = frozenset({"french", "german"})


def ppocr_code(language: OcrLanguage, vocabulary: frozenset[str]) -> str | None:
    """Map a canonical tag onto a PP-OCR code, or `None` if there is no model.

    `vocabulary` is the union of code sets the caller can actually reach, so
    the resolution never returns a code the backend cannot serve.
    """
    if language.is_passthrough:
        # `arabic`, `cyrillic`: a recognizer named after a script, handed over as
        # the user wrote it.
        return language.native if language.native in vocabulary else None
    if language.is_multilingual:
        return None

    if language.bcp47 in _CANONICAL_TO_CODE:
        code = _CANONICAL_TO_CODE[language.bcp47]
        return code if code is not None and code in vocabulary else None

    # The primary subtag identifies the recognizer only when the language is
    # written in its usual script: PP-OCR's `az` and `uz` are the Latin ones.
    if language.has_default_script and language.bcp47_language in vocabulary:
        return language.bcp47_language

    # PP-OCR serves many languages only through a script-wide recognizer: there
    # is no `ar` or `hi` model, and on the PP-OCRv4 backbone most of the
    # vocabulary is script models. This routing is internal -- users name a
    # language and docling finds the recognizer that covers it.
    family = _SCRIPT_TO_CODE.get(language.bcp47_script or "")
    if family is not None and family in vocabulary:
        return family
    return None


def ppocr_supported_tags(vocabulary: frozenset[str]) -> list[str]:
    """Render a PP-OCR code vocabulary back as the canonical tags it serves."""
    tags: set[str] = set()
    for code in vocabulary:
        if code in _REDUNDANT_CODES:
            continue
        if code in _CODE_TO_CANONICAL:
            tags.update(_CODE_TO_CANONICAL[code])
            continue
        language = OcrLanguageResolver.canonicalize_ocr_language(
            code, raise_exception=False
        )
        # `None` is a code that is not a language code and has no reverse
        # entry; it is unreachable from a canonical tag anyway.
        if language is not None:
            tags.add(language.tag)
    return sorted(tags)
