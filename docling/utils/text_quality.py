# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look only at patterns that broken font mappings produce. Two of them
apply to every script: characters that do not belong in text, on the page or
in many of its lines. Two apply to Latin letters only, and use no language
profile: accented letters that stand for another script, and too few vowels.

An optional check covers Indian scripts, whose broken layers often hold valid
letters in the wrong places, such as words that start with a vowel sign.
"""

import itertools
import re
import unicodedata
from collections import Counter

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

# Indian scripts: only judged when the check is on and the page has enough words
# in mostly one of these scripts (the first word of a character's Unicode name).
_INDIC_SCRIPTS = frozenset(
    {
        "BENGALI",
        "DEVANAGARI",
        "GUJARATI",
        "GURMUKHI",
        "KANNADA",
        "MALAYALAM",
        "ORIYA",
        "SINHALA",
        "TAMIL",
        "TELUGU",
    }
)
_INDIC_LETTER = re.compile("[ऀ-෿]")  # Devanagari to Sinhala.
_MIN_INDIC_WORDS = 20
_SCRIPT_SHARE = 0.5  # Of the page's letters and signs.
_MISSPELLED_SHARE = 0.01  # Real text stays near 0.


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


def _script(ch: str) -> str:
    return unicodedata.name(ch, " ").split(" ", 1)[0]


def _is_misspelled(word: str) -> bool:
    """Whether a word starts with a sign or has two vowel signs in a row.

    Both are impossible in real text. A broken layer gives them when the glyphs
    of a vowel drawn before its consonant are read in drawing order.
    """
    if unicodedata.category(word[0]) in ("Mn", "Mc"):
        return True
    signs = ["VOWEL SIGN" in unicodedata.name(ch, "") for ch in word]
    return any(a and b for a, b in itertools.pairwise(signs))


def _indic_words(text: str) -> tuple[str | None, list[str]]:
    """The page's main Indian script and its words in that script."""
    text = unicodedata.normalize("NFC", text)
    scripts = Counter(_script(ch) for ch in text if unicodedata.category(ch)[0] in "LM")
    if not scripts:
        return None, []
    script, count = scripts.most_common(1)[0]
    if script not in _INDIC_SCRIPTS or count < _SCRIPT_SHARE * scripts.total():
        return None, []
    words = []
    for token in text.split():
        # Strip punctuation and symbols, such as quotes and the danda.
        start, end = 0, len(token)
        while start < end and unicodedata.category(token[start])[0] in "PS":
            start += 1
        while end > start and unicodedata.category(token[end - 1])[0] in "PS":
            end -= 1
        word = token[start:end]
        if word and any(_script(ch) == script for ch in word):
            words.append(word)
    return script, words


def _is_wrong_indic(text: str) -> bool:
    """Whether the text of an Indian script has letters in the wrong places."""
    if not _INDIC_LETTER.search(text):
        return False
    script, words = _indic_words(text)
    if script is None or len(words) < _MIN_INDIC_WORDS:
        return False
    return sum(map(_is_misspelled, words)) >= _MISSPELLED_SHARE * len(words)


def is_broken_text_layer(text: str, check_indic: bool = False) -> bool:
    """Whether a page's PDF text does not match the text drawn on the page.

    Args:
        text: All text of one page's PDF text layer, cells joined by newlines.
        check_indic: Also check text in Indian scripts.

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

    if _is_wrong_latin(stripped):
        return True
    return check_indic and _is_wrong_indic(stripped)
