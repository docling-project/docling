# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re

# Pre-compiled regex patterns for text quality and font sanity
GLYPH_RE = re.compile(r"GLYPH<[0-9A-Fa-f]+>")
SLASH_G_RE = re.compile(r"(?:/G\d+){2,}")
FRAG_RE = re.compile(r"\b[A-Za-z](?:/[a-z]{1,3}\.[a-z]{1,3}){2,}\b")
SLASH_NUMBER_GARBAGE_RE = re.compile(r"(?:/\w+\s*){2,}")
PUA_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
INTRA_WORD_SYMBOL_RE = re.compile(
    r"[A-Za-z0-9\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u0D7F\u4E00-\u9FFF][\%\"^~&|\\$#*+<=>`][A-Za-z0-9\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u0D7F\u4E00-\u9FFF]"
)


def rate_text_quality(text: str) -> float:
    """Rate extracted text quality and detect font corruption/mojibake.

    Returns a score between 0.0 (severely corrupted) and 1.0 (clean text).
    """
    if not text:
        return 1.0

    # Hard errors: replacement char, PUA characters, control characters, GLYPH tags, slash-garbage
    if (
        "\ufffd" in text
        or PUA_RE.search(text)
        or CONTROL_CHAR_RE.search(text)
        or GLYPH_RE.search(text)
        or SLASH_G_RE.search(text)
        or SLASH_NUMBER_GARBAGE_RE.match(text)
    ):
        return 0.0

    # Intra-word symbol noise check (e.g. 8-bit font remapping mojibake like d%b&c)
    tokens = text.split()
    if tokens:
        intra_symbol_count = sum(1 for t in tokens if INTRA_WORD_SYMBOL_RE.search(t))
        if (intra_symbol_count / len(tokens)) >= 0.25:
            return 0.0

    penalty = 0.0
    frag_matches = FRAG_RE.findall(text)
    if len(frag_matches) >= 3:
        penalty += 0.1 * len(frag_matches)

    return max(1.0 - penalty, 0.0)
