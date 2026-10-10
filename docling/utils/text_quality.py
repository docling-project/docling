# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look at a page's whole text layer and at each of its lines, and only
at patterns broken text produces, such as unmapped font encodings.
"""

import math
import unicodedata
from itertools import groupby

_MIN_CHARS = 20  # Less text than this is left alone.
_MIN_LATIN_LETTERS = 50  # Too few Latin letters to judge their mix.
_CONTROL_SHARE = 0.03  # Control or U+FFFD characters.
_PRIVATE_USE_SHARE = 0.5  # Private-use characters; icon fonts stay far below.
_NON_ASCII_LATIN_SHARE = 0.5  # Latin letters outside ASCII.
_ASCII_LETTER_SHARE = 1 / 3  # Mojibake leaves few ASCII letters.

# One line is one text cell.
_PRIVATE_USE_RUN = 4  # Different private-use characters in a row.
_MIN_LINE_CHARS = 5  # Less text than this in a line or token is not judged.
_TOKEN_CONTROLS = 2  # Control characters in one token.
_TOKEN_CONTROL_SHARE = 0.05

# U+FFFD on the page.
_SHORT_PAGE_CHARS = 80  # On a page this short, a run of 2 is enough.
_SHORT_PAGE_RUN = 2
_MANY_REPLACEMENTS = 12
_MANY_REPLACEMENTS_SHARE = 0.05
_REPLACEMENT_LINES = 3
_LONG_REPLACEMENT_RUN = 8
_SOME_REPLACEMENTS_SHARE = 0.025

# Letters shifted by a broken ToUnicode map, as in a substitution cipher.
_MIN_CIPHER_LETTERS = 200  # Too few letters to judge their frequencies.
_MAX_VOWEL_SHARE = 0.30  # Latin-script text stays above this.
_MAX_ENGLISH_COSINE = 0.60  # Letters not where English puts them...
_MIN_SHAPE_COSINE = 0.90  # ...but with the frequency profile of a language.
# English letter frequencies, percent, a to z (Lewand, "Cryptological Mathematics").
_ENGLISH_LETTER_FREQ = (
    8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
    6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07,
)  # fmt: skip

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


def _is_private_use(ch: str) -> bool:
    return unicodedata.category(ch) == "Co"


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


def _ascii_letter_share(text: str) -> float:
    """Share of non-space characters that are ASCII letters."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isascii() and ch.isalpha() for ch in chars) / len(chars)


def _is_broken_line(line: str) -> bool:
    """Whether one line holds a run or a token that only a broken font produces."""
    # Icons repeat a few private-use code points; the letters of words vary.
    for is_private_use, run in groupby(line, key=_is_private_use):
        if is_private_use and len(set(run)) >= _PRIVATE_USE_RUN:
            return True
    for token in line.split():
        if len(token) < _MIN_LINE_CHARS:
            continue
        controls = sum(unicodedata.category(ch) in _CONTROL for ch in token)
        if controls >= _TOKEN_CONTROLS and controls >= _TOKEN_CONTROL_SHARE * len(
            token
        ):
            return True
    return False


def _has_replacement_evidence(text: str, lines: list[str]) -> bool:
    """Whether U+FFFD is dense or clustered enough to mark the page broken."""
    count = text.count("\ufffd")
    if not count:
        return False
    chars = sum(not ch.isspace() for ch in text)
    longest = max((len(list(run)) for ch, run in groupby(text) if ch == "\ufffd"))
    if chars <= _SHORT_PAGE_CHARS and longest >= _SHORT_PAGE_RUN:
        return True
    share = count / chars
    lines_hit = sum("\ufffd" in line for line in lines)
    return (
        (count >= _MANY_REPLACEMENTS and share >= _MANY_REPLACEMENTS_SHARE)
        or (lines_hit >= _REPLACEMENT_LINES and share >= _SOME_REPLACEMENTS_SHARE)
        or (longest >= _LONG_REPLACEMENT_RUN and share >= _SOME_REPLACEMENTS_SHARE)
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 1.0


def _is_shifted_latin(text: str) -> bool:
    """Whether ASCII letters look like language with its letters swapped.

    A broken ToUnicode map often shifts every letter by a constant. That keeps
    the shape of the letter-frequency profile but moves each letter. Only tokens
    with a lowercase letter count, so acronyms and part numbers are skipped.
    """
    counts = [0.0] * 26
    other_letters = 0
    for token in text.split():
        if not any("a" <= ch <= "z" for ch in token):
            continue
        for ch in token:
            if "a" <= ch.lower() <= "z":
                counts[ord(ch.lower()) - ord("a")] += 1
            elif ch.isalpha():
                other_letters += 1
    letters = sum(counts)
    if letters < _MIN_CIPHER_LETTERS or other_letters > letters:
        return False
    vowels = sum(counts[ord(v) - ord("a")] for v in "aeiou")
    if vowels / letters > _MAX_VOWEL_SHARE:
        return False
    english = list(_ENGLISH_LETTER_FREQ)
    return (
        _cosine(counts, english) < _MAX_ENGLISH_COSINE
        and _cosine(sorted(counts, reverse=True), sorted(english, reverse=True))
        >= _MIN_SHAPE_COSINE
    )


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
    lines = stripped.splitlines()

    # A font whose glyphs were mapped to control or private-use code points.
    if _share(stripped, _CONTROL, replacement=True) >= _CONTROL_SHARE:
        return True
    if _share(stripped, frozenset({"Co"}), replacement=False) >= _PRIVATE_USE_SHARE:
        return True
    if _has_replacement_evidence(stripped, lines):
        return True
    # The same, in one cell of an otherwise good page.
    if any(_is_broken_line(line) for line in lines):
        return True

    # A non-Latin font whose glyphs were mapped to accented Latin letters.
    if (
        _non_ascii_latin_share(stripped) >= _NON_ASCII_LATIN_SHARE
        and _ascii_letter_share(stripped) < _ASCII_LETTER_SHARE
    ):
        return True

    # A Latin font whose glyphs were mapped to the wrong ASCII letters.
    return _is_shifted_latin(stripped)
