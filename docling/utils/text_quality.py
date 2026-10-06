# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look at a page's whole text layer and only at patterns broken text
produces, such as unmapped font encodings or leftover OCR noise.
"""

import re
import unicodedata
from collections.abc import Sequence

# A short layer that only names an image file, left behind by a scanner.
_IMAGE_FILENAME_RE = re.compile(r"(?i)\.(?:tif|tiff|jpe?g|png|gif|bmp|pict)\b")
_COMMON_PUNCT = frozenset(
    ".,;:'\"!?()-[]/%$&+#@=<>/—\N{EN DASH}\N{RIGHT SINGLE QUOTATION MARK}“”«»"
)
# Symbols ordinary documents are full of; never evidence of a broken layer.
_ORDINARY_SYMBOLS = frozenset(
    "*•▪■□◆◇●○►▶✓✔☐☒☑_®©™±°|≤≥≈\N{MULTIPLICATION SIGN}÷\N{MINUS SIGN}\N{EN DASH}—…§¶†‡€£¥¢‰\N{PRIME}″←→↑↓"
)
# Symbols that, inside a word, mark a non-Latin font read as Latin letters.
_IN_WORD_SYMBOLS = frozenset("$`©«»¥÷½{}^~±§@#%\\<>¤¦¨¬®¯°¹²³¼¾\N{MULTIPLICATION SIGN}")
# Characters a punctuation-soup token may hold without counting: leaders, $, dashes.
_SOUP_OK = frozenset(".,-\N{EN DASH}—$()%:;'\"/…_•▪*")
_BROKEN_CATEGORIES = frozenset({"Cc", "Co", "Cn", "Cs"})

_MIN_CHARS = 20  # Less text than this is left alone.
_SHORT = 120  # Below this length, ratio checks are unreliable.
_BROKEN_CHAR_SHARE = 0.10  # Control, private-use or U+FFFD characters.
_SYMBOL_WORD_SHARE = 0.20  # Words with symbols between their letters.
_NON_ASCII_LATIN_SHARE = 0.5  # Latin letters outside ASCII.
_SOUP_SHARE = 0.30  # Words made mostly of symbols.
_UNUSUAL_PER_100_LETTERS = 2.0  # Unusual characters on a scan.


def _letter_scripts(text: str) -> dict[str, int]:
    counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "arabic": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif any(
            token in name
            for token in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "IDEOGRAPH")
        ):
            counts["cjk"] += 1
        elif "ARABIC" in name:
            counts["arabic"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def _is_ordinary(ch: str) -> bool:
    """Not a letter, but nothing a broken layer produces either."""
    category = unicodedata.category(ch)
    return (
        ch.isalnum()
        or ch.isspace()
        or ch in _COMMON_PUNCT
        or ch in _ORDINARY_SYMBOLS
        or category[0] == "M"  # combining marks: Thai / Devanagari vowel signs
        or category == "Co"  # private use: icon-font glyphs
    )


def _unusual_per_100_letters(text: str) -> float:
    letters = 0
    unusual = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
        elif not _is_ordinary(ch):
            unusual += 1
    return 100.0 * unusual / letters if letters else 0.0


def _broken_char_share(text: str) -> float:
    """Share of non-space characters that are control, private-use or U+FFFD."""
    chars = [ch for ch in text if not ch.isspace()]
    broken = sum(
        1
        for ch in chars
        if unicodedata.category(ch) in _BROKEN_CATEGORIES or ch == "\ufffd"
    )
    return broken / len(chars) if chars else 0.0


def _symbol_word_share(words: Sequence[str]) -> float:
    """Share of lettered words with two or more symbols between their letters."""
    lettered = [w for w in words if len(w) >= 3 and any(ch.isalpha() for ch in w)]
    if not lettered:
        return 0.0
    mixed = 0
    for word in lettered:
        letters = [i for i, ch in enumerate(word) if ch.isalpha()]
        inner = word[letters[0] : letters[-1]]
        if sum(ch in _IN_WORD_SYMBOLS for ch in inner) >= 2:
            mixed += 1
    return mixed / len(lettered)


def _non_ascii_latin_share(text: str) -> float:
    """Share of Latin letters outside ASCII, when Latin carries the page."""
    letters = [ch for ch in text if ch.isalpha()]
    latin = [
        ch
        for ch in letters
        if "LATIN" in unicodedata.name(ch, "") and not "\uff00" <= ch <= "\uffef"
    ]
    if len(latin) < 50 or len(latin) * 2 < len(letters):
        return 0.0
    return sum(ord(ch) > 127 for ch in latin) / len(latin)


def _soup_share(words: Sequence[str]) -> float:
    """Share of words that are mostly symbols beyond leaders, currency and dashes."""
    if not words:
        return 0.0
    soup = sum(
        1
        for word in words
        if sum(ch.isalnum() for ch in word) * 2 < len(word)
        and any(not ch.isalnum() and ch not in _SOUP_OK for ch in word)
    )
    return soup / len(words)


def is_broken_text_layer(text: str, *, has_bitmap: bool) -> bool:
    """Whether a page's PDF text is not the text drawn on the page.

    Args:
        text: All text of one page's PDF text layer, cells joined by newlines.
        has_bitmap: Whether the page holds an image, i.e. may be a scan.

    Returns:
        ``True`` only for text a broken layer produces. When unsure it returns
        ``False``, so pages keep their PDF text.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    if len(stripped) < _SHORT and _IMAGE_FILENAME_RE.search(stripped):
        return True

    # A broken font encoding, whether or not the page holds an image.
    if (
        len(stripped) >= _MIN_CHARS
        and _broken_char_share(stripped) >= _BROKEN_CHAR_SHARE
    ):
        return True

    words = stripped.split()
    if len(stripped) >= _SHORT and (
        _symbol_word_share(words) >= _SYMBOL_WORD_SHARE
        or _non_ascii_latin_share(stripped) >= _NON_ASCII_LATIN_SHARE
        or _soup_share(words) >= _SOUP_SHARE
    ):
        return True

    # The remaining checks only apply to scans with a hidden OCR layer.
    if not has_bitmap or len(stripped) < _MIN_CHARS:
        return False

    scripts = _letter_scripts(stripped)
    if scripts["cyrillic"] >= 30 or scripts["cjk"] >= 20 or scripts["arabic"] >= 20:
        return False

    letters = sum(ch.isalpha() for ch in stripped)
    unusual = _unusual_per_100_letters(stripped)
    if len(stripped) < _SHORT and (
        letters / len(stripped) < 0.5 or unusual >= _UNUSUAL_PER_100_LETTERS
    ):
        return True
    return (
        scripts["latin"] >= max(1, letters * 0.5)
        and unusual >= _UNUSUAL_PER_100_LETTERS
    )
