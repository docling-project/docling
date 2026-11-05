# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look only at patterns that broken font mappings produce. Two of them
apply to every script: characters that do not belong in text, on the page or
in many of its lines. Two apply to Latin letters only, and use no language
profile: accented letters that stand for another script, and too few vowels.
"""

import re
import unicodedata

_MIN_CHARS = 20  # Less text than this is left alone.
_JUNK_SHARE = 0.03  # Control, U+FFFD or private-use characters on the page.
_JUNK_LINE_SHARE = 0.10  # Lines with U+FFFD or a control character...
_MIN_JUNK_LINES = 3  # ...and at least this many.

# Latin letters: only judged when there are enough of them.
_MIN_LATIN_LETTERS = 50
_ACCENTED_LATIN_SHARE = 0.5  # Real text in Latin script stays far below.
_MIN_WORD_LETTERS = 200
_MIN_VOWEL_SHARE = 0.25  # Real text in Latin script stays above about 0.3.
# Shorter tokens are math variables; longer ones are base64 or hashes.
_WORD_LENGTH = range(3, 21)
# Base vowels, with æ, œ, ø, dotless i and schwa, which have no base letter.
_VOWELS = frozenset("aeiouy\u00e6\u0153\u00f8\u0131\u0259")
_GLYPH_PLACEHOLDER = re.compile(r"GLYPH<[^>]*>")  # A parser's unmapped glyph.


def _is_junk(ch: str) -> bool:
    """Whether a character is a control, surrogate, private-use or U+FFFD."""
    category = unicodedata.category(ch)
    return (
        ch == "�" or category in ("Cs", "Co") or (category == "Cc" and not ch.isspace())
    )


def _has_replacement(line: str) -> bool:
    """Whether a line has U+FFFD or a control character after its first character.

    Private-use characters are not counted: icon fonts put them in many lines.
    """
    return any(
        ch == "�" or (unicodedata.category(ch) == "Cc" and not ch.isspace())
        for ch in line.lstrip()[1:]
    )


def _without_bullet(line: str) -> str:
    """Return the line without a lone junk character at its start.

    An unmapped bullet glyph is a lone junk character before the item text; a
    broken font puts junk inside words and numbers.
    """
    stripped = line.lstrip()
    if stripped[:1] and _is_junk(stripped[0]) and stripped[1:2].isspace():
        return stripped[1:]
    if len(stripped) == 1 and _is_junk(stripped):
        return ""
    return line


def _is_latin(ch: str) -> bool:
    return ch.isalpha() and "LATIN" in unicodedata.name(ch, "")


def _is_wrong_latin(text: str) -> bool:
    """Whether the Latin letters stand for another script or were moved.

    A non-Latin 8-bit font read as Latin-1 gives mostly accented letters
    (`Îò÷ò`). A broken ToUnicode map that moves letters, or a legacy font that
    puts another script on ASCII letters, leaves few vowels. Accented vowels
    count by their base letter, so no Latin-script language is penalized.
    """
    text = _GLYPH_PLACEHOLDER.sub(" ", text)
    latin = [ch for ch in text if _is_latin(ch)]
    if len(latin) < _MIN_LATIN_LETTERS:
        return False
    # 8-bit code pages decode to U+00C0-U+024F; IPA and Vietnamese tones lie above.
    accented = sum("À" <= ch <= "ɏ" for ch in latin)
    if accented >= _ACCENTED_LATIN_SHARE * len(latin):
        return True
    words = []
    for token in text.split():
        letters = [ch for ch in token if _is_latin(ch)]
        if len(letters) in _WORD_LENGTH and any(ch.islower() for ch in letters):
            words.extend(letters)
    if len(words) < _MIN_WORD_LETTERS:
        return False
    vowels = sum(unicodedata.normalize("NFD", ch)[0].lower() in _VOWELS for ch in words)
    return vowels < _MIN_VOWEL_SHARE * len(words)


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
    lines = [_without_bullet(line) for line in stripped.splitlines()]
    chars = [ch for line in lines for ch in line if not ch.isspace()]
    if not chars:
        return False

    # A font whose glyphs were mapped to control, private-use or no code points.
    if sum(map(_is_junk, chars)) >= _JUNK_SHARE * len(chars):
        return True
    # The same in many lines, such as unmapped ligatures (`publica�on`).
    junk_lines = sum(map(_has_replacement, lines))
    if junk_lines >= max(_MIN_JUNK_LINES, _JUNK_LINE_SHARE * len(lines)):
        return True

    return _is_wrong_latin(stripped)
