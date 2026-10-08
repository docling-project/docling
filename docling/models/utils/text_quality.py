# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re
import unicodedata

# Core PDF dump artifact regexes
GLYPH_RE = re.compile(r"GLYPH<[0-9A-Fa-f]+>")
SLASH_G_RE = re.compile(r"(?:/G\d+){2,}")
FRAG_RE = re.compile(r"\b[A-Za-z](?:/[a-z]{1,3}\.[a-z]{1,3}){2,}\b")
SLASH_NUMBER_GARBAGE_RE = re.compile(r"(?:/\w+\s*){2,}")

# Unprintable control characters (excluding standard whitespace \t, \n, \r)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

# Genuinely corrupt intra-word mojibake noise (e.g. backslash, caret, tilde, pipe, backtick inside words)
# Legitimate characters like &, +, =, -, *, /, $, % are explicitly permitted
MOJIBAKE_SYMBOL_RE = re.compile(
    r"[A-Za-z0-9\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u0D7F\u4E00-\u9FFF][\\^~|`#][A-Za-z0-9\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0900-\u0D7F\u4E00-\u9FFF]"
)


def _is_pua(ch: str) -> bool:
    """Check if a character belongs to Unicode Private Use Area."""
    return unicodedata.category(ch) == "Co" or (
        "\ue000" <= ch <= "\uf8ff"
        or "\U000f0000" <= ch <= "\U000ffffd"
        or "\U00100000" <= ch <= "\U0010fffd"
    )


def _has_corrupt_pua(text: str) -> bool:
    """Detect unmapped PUA font dumps vs benign presentation bullet/icon fonts."""
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return False
    pua_chars = [ch for ch in non_space if _is_pua(ch)]
    if not pua_chars:
        return False

    # Icons / bullets repeat 1 or 2 unique code points; unmapped font alphabets vary
    if len(set(pua_chars)) >= 3:
        return True

    # If PUA chars dominate the text block (>= 20%), it is an unmapped font dump
    if len(pua_chars) / len(non_space) >= 0.20:
        return True

    return False


def rate_text_quality(text: str) -> float:
    """Rate extracted text quality and detect font corruption/mojibake.

    Returns a score between 0.0 (severely corrupted) and 1.0 (clean text).
    """
    if not text:
        return 1.0

    # 1. Hard parser errors: replacement chars, control characters, GLYPH dumps, slash-garbage
    if (
        "\ufffd" in text
        or CONTROL_CHAR_RE.search(text)
        or GLYPH_RE.search(text)
        or SLASH_G_RE.search(text)
        or SLASH_NUMBER_GARBAGE_RE.match(text)
    ):
        return 0.0

    # 2. Corrupt unmapped Private Use Area alphabets (while allowing single/repeated bullet icons)
    if _has_corrupt_pua(text):
        return 0.0

    # 3. Mojibake intra-word noise (e.g., d\b^c~e)
    tokens = text.split()
    if tokens:
        mojibake_count = sum(1 for t in tokens if MOJIBAKE_SYMBOL_RE.search(t))
        if (mojibake_count / len(tokens)) >= 0.25:
            return 0.0

    # 4. Repeated font fragment paths penalty
    penalty = 0.0
    frag_matches = FRAG_RE.findall(text)
    if len(frag_matches) >= 3:
        penalty += 0.1 * len(frag_matches)

    return max(1.0 - penalty, 0.0)
