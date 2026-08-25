# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Canonical BCP-47 to PP-OCR recognizer tokens.

RapidOCR and the KServe v2 OCR client both address PP-OCR recognizers by the
same tokens, so the mapping lives here rather than in either engine. RapidOCR
consults the installed `rapidocr` package for the authoritative PP-OCRv6 set and
falls back to the static copy below; the KServe client uses the static copy only,
so it never has to import `rapidocr`.

The static token sets mirror the PP-OCR release notes summarised in
`docs/concepts/OCR.md`. They can drift from a newer `rapidocr`; this module is
their single owner.
"""

from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)

# Recognition languages served by the PP-OCRv4 backbone (the torch fallback).
PPOCRV4_LANGS = frozenset(
    {"arabic", "cyrillic", "devanagari", "ka", "korean", "latin", "ta", "te"}
)

# Recognition languages served by the PP-OCRv5 backbone.
PPOCRV5_LANGS = frozenset(
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
PPOCRV6_LANGS = frozenset(
    {
        "af", "az", "bs", "ca", "ch", "chinese_cht", "cs", "cy", "da", "de",
        "en", "es", "et", "eu", "fi", "fr", "french", "ga", "german", "gl",
        "hr", "hu", "id", "is", "it", "japan", "ku", "la", "lb", "lt", "lv",
        "mi", "ms", "mt", "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro",
        "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi",
    }
)  # fmt: skip

# PP-OCR token used when `lang` is left empty: the engine's own default
# recognizer, which is Simplified Chinese.
PPOCR_DEFAULT_TOKEN = "ch"

# Canonical tag -> PP-OCR token, for the languages whose token is not simply the
# primary subtag. `None` marks a tag that must *not* fall through to the generic
# rules below, because the token that looks right means something else.
_CANONICAL_TO_TOKEN: dict[str, str | None] = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "ja-Jpan": "japan",
    "ko-Kore": "korean",
    "sr-Latn": "rs_latin",
    # `tl` is PP-OCR's token; BCP-47 canonicalizes Tagalog to `fil`.
    "fil-Latn": "tl",
    # PP-OCR serves East Slavic with a narrower recognizer than `cyrillic`.
    "ru-Cyrl": "eslav",
    "uk-Cyrl": "eslav",
    "be-Cyrl": "eslav",
    # PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.
    "kn-Knda": "ka",
    "ka-Geor": None,
}

# ISO 15924 script -> PP-OCR script-family token. Internal routing only: users
# name a language and this finds the script-wide recognizer that covers it, for
# the many languages PP-OCR serves no other way.
_SCRIPT_TO_TOKEN: dict[str, str] = {
    "Latn": "latin",
    "Cyrl": "cyrillic",
    "Arab": "arabic",
    "Deva": "devanagari",
}

# Reverse of the language table, for rendering a vocabulary back as tags. The
# script recognizers are not reversed: they are named back as themselves, which
# is what the user types to select one.
_TOKEN_TO_CANONICAL: dict[str, list[str]] = {}
for _tag, _token in _CANONICAL_TO_TOKEN.items():
    if _token is not None:
        _TOKEN_TO_CANONICAL.setdefault(_token, []).append(_tag)

# PP-OCRv6 tokens that duplicate a language already reachable by its subtag.
_REDUNDANT_TOKENS = frozenset({"french", "german"})


def ppocr_token(language: OcrLanguage, vocabulary: frozenset[str]) -> str | None:
    """Map a canonical tag onto a PP-OCR token, or `None` if there is no model.

    `vocabulary` is the union of token sets the caller can actually reach, so
    the resolution never returns a token the backend cannot serve.
    """
    if language.is_passthrough:
        # `arabic`, `cyrillic`: a recognizer named after a script, handed over as
        # the user wrote it.
        return language.native if language.native in vocabulary else None
    if language.is_multilingual:
        return None

    if language.tag in _CANONICAL_TO_TOKEN:
        token = _CANONICAL_TO_TOKEN[language.tag]
        return token if token is not None and token in vocabulary else None

    # The primary subtag identifies the recognizer only when the language is
    # written in its usual script: PP-OCR's `az` and `uz` are the Latin ones.
    if language.has_default_script and language.language in vocabulary:
        return language.language

    # PP-OCR serves many languages only through a script-wide recognizer: there
    # is no `ar` or `hi` model, and on the PP-OCRv4 backbone most of the
    # vocabulary is script models. This routing is internal -- users name a
    # language and docling finds the recognizer that covers it.
    family = _SCRIPT_TO_TOKEN.get(language.script or "")
    if family is not None and family in vocabulary:
        return family
    return None


def ppocr_supported_tags(vocabulary: frozenset[str], kind: str) -> list[str]:
    """Render a PP-OCR token vocabulary back as the canonical tags it serves.

    `kind` is the calling engine's `OcrOptions.kind`: local RapidOCR and the
    KServe v2 client address the same recognizers, and either one selects the
    PP-OCR vocabulary.
    """
    tags: set[str] = set()
    for token in vocabulary:
        if token in _REDUNDANT_TOKENS:
            continue
        if token in _TOKEN_TO_CANONICAL:
            tags.update(_TOKEN_TO_CANONICAL[token])
            continue
        try:
            tags.add(OcrLanguageResolver.parse_ocr_language(token, kind).tag)
        except ValueError:
            # A token that is not a language code and has no reverse entry;
            # it is unreachable from a canonical tag anyway.
            continue
    return sorted(tags)
