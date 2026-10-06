# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look at a page's whole text layer and only at patterns broken text
produces, such as unmapped font encodings.
"""

import unicodedata

_MIN_CHARS = 20  # Less text than this is left alone.
_MIN_LATIN_LETTERS = 50  # Too few Latin letters to judge their mix.
_CONTROL_SHARE = 0.03  # Control or U+FFFD characters.
_PRIVATE_USE_SHARE = 0.5  # Private-use characters; icon fonts stay far below.
_NON_ASCII_LATIN_SHARE = 0.5  # Latin letters outside ASCII.

# Not "Cn" (unassigned): that depends on the Unicode version of the Python running.
_CONTROL = frozenset({"Cc", "Cs"})


def _share(text: str, categories: frozenset[str], *, replacement: bool) -> float:
    """Share of non-space characters in the given Unicode categories."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    hits = sum(
        unicodedata.category(ch) in categories or (replacement and ch == "\ufffd")
        for ch in chars
    )
    return hits / len(chars)


def _non_ascii_latin_share(text: str) -> float:
    """Share of Latin letters outside ASCII, on text where Latin letters dominate."""
    letters = [ch for ch in text if ch.isalpha()]
    latin = [
        ch
        for ch in letters
        if "LATIN" in unicodedata.name(ch, "") and not "\uff00" <= ch <= "\uffef"
    ]
    if len(latin) < _MIN_LATIN_LETTERS or len(latin) * 2 < len(letters):
        return 0.0
    return sum(ord(ch) > 127 for ch in latin) / len(latin)


def is_broken_text_layer(text: str) -> bool:
    """Whether a page's PDF text does not match the text drawn on the page.

    Args:
        text: All text of one page's PDF text layer, cells joined by newlines.

    Returns:
        ``True`` only for text a broken layer produces. When unsure it returns
        ``False``, so pages keep their PDF text.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS:
        return False

    # A font whose glyphs were mapped to control or private-use code points.
    if _share(stripped, _CONTROL, replacement=True) >= _CONTROL_SHARE:
        return True
    if _share(stripped, frozenset({"Co"}), replacement=False) >= _PRIVATE_USE_SHARE:
        return True

    # A non-Latin font whose glyphs were mapped to accented Latin letters.
    return _non_ascii_latin_share(stripped) >= _NON_ASCII_LATIN_SHARE
