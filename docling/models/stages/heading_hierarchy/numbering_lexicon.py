# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keyword markers of the heading numbering signal ("PART I", "Chapter 2", "Art. 5").

The words are data, not code: :class:`KeywordLexicon` merges the built-in English words with
the words of ``HeadingHierarchyOptions.numbering_keywords`` and compiles them once, so a
document in another language is read with the same rules as an English one.
"""

import re
import unicodedata
from collections.abc import Iterable, Mapping

_DEFAULT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "part": ("part", "title", "book"),
    "chapter": ("chapter",),
    "division": ("division",),
    "subdivision": ("subdivision",),
    "article": (
        "article",
        "section",
        "clause",
        "schedule",
        "annex",
        "appendix",
        "rule",
    ),
}
_DEFAULT_NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy"
    " eighty ninety first second third fourth fifth sixth seventh eighth ninth tenth"
    " eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth"
    " nineteenth twentieth".split()
)
# "Division"/"Subdivision" are stricter than the other families: they only count as numbering
# when an explicit enumerator follows; see _ENUMERATOR.
_ENUMERATOR_ONLY = frozenset({"division", "subdivision"})

# Canonical Roman-numeral validator (1..3999), case-insensitive.
_ROMAN_RE = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE
)

# What may follow a keyword. An abbreviation carries its dot ("art."), so the enumerator may
# follow it without a space ("Art.5").
_GAP = r"(?:\s+|(?<=\.))"
# Keywords are ordinary words too ("Rule Definition", "Part-Year Method"), so they only count
# as numbering when they stand alone ("APPENDIX", "Annex:") or the next token is an
# enumerator; see KeywordLexicon._is_enumerator.
_TAIL = re.compile(
    _GAP + r"([^\s.,:;)\]\u2013\u2014-]+)|\s*(?:[.:\u2013\u2014]|-\s|$)", re.IGNORECASE
)
# An Arabic index, a single letter or a Roman numeral. It must be terminated by whitespace,
# end-of-string, or separator punctuation, so a French elision such as "Division d'appel"
# (a letter followed by an apostrophe) is not read as a marker.
_ENUMERATOR = re.compile(
    _GAP + r"(\d+|[a-z]|[ivxlcdm]+)(?=\s|$|[.,:;)\]\u2013\u2014-])", re.IGNORECASE
)
# Statute citations like "Chapter A-15" (letter-dash-digit) are not chapters.
_CITATION = re.compile(r"\s+[a-z]-\d", re.IGNORECASE)
# A keyword ends where its tail can start; a keyword with a dot of its own needs no boundary.
_BOUNDARY = r"(?=\s|$|[.:\u2013\u2014-])"


def is_roman_numeral(token: str) -> bool:
    return bool(token) and _ROMAN_RE.fullmatch(token) is not None


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _word_pattern(word: str) -> str:
    return r"\s+".join(re.escape(part) for part in word.split())


class KeywordLexicon:
    """The keyword markers and number words of one configuration, compiled once."""

    def __init__(
        self,
        keywords: Mapping[str, Iterable[str]] | None = None,
        number_words: Iterable[str] = (),
    ) -> None:
        """Merge the built-in words with ``keywords`` (family -> words) and ``number_words``.

        A word in ``keywords`` replaces the built-in family of that word.
        """
        by_word: dict[str, tuple[str, str]] = {}
        for source in (_DEFAULT_KEYWORDS, keywords or {}):
            for family, words in source.items():
                for word in words:
                    spelling = " ".join(_nfc(word).split())
                    by_word[spelling.casefold()] = (spelling, family)
        # Longest first, so "article" is tried before "art".
        entries = sorted(by_word.values(), key=lambda entry: -len(entry[0]))

        self._families = [family for _, family in entries]
        self._keyword = re.compile(
            "^(?:"
            + "|".join(
                f"({_word_pattern(word)})" + ("" if word.endswith(".") else _BOUNDARY)
                for word, _ in entries
            )
            + ")",
            re.IGNORECASE,
        )
        self._number_words = _DEFAULT_NUMBER_WORDS | {
            _nfc(word).lower() for word in number_words
        }
        self._leading_marker = re.compile(
            r"^\s*(?:(?:"
            + "|".join(
                _word_pattern(word) + (r"\b" if word[-1].isalnum() else "")
                for word, _ in entries
            )
            + r")[\s.:]*[0-9ivxlcdm]*"
            r"|§+\s*[0-9.]+"
            r"|\(?[0-9]+(?:\.[0-9]+)*[.)\]]?"
            r"|\(?[A-Za-z]{1,2}(?:[.)\]]|(?=\s*[:\-\u2013\u2014]))"
            r")[\s.:)\-\u2013\u2014]*",
            re.IGNORECASE,
        )

    def family(self, text: str) -> str | None:
        """Scheme family of the keyword marker that starts ``text``, or None."""
        text = _nfc(text)
        keyword = self._keyword.match(text)
        if keyword is None or keyword.lastindex is None:
            return None
        family = self._families[keyword.lastindex - 1]
        end = keyword.end()

        if family in _ENUMERATOR_ONLY:
            enumerator = _ENUMERATOR.match(text, end)
            if enumerator is None or not self._is_enumerator(enumerator.group(1)):
                return None
            return family

        if family == "chapter" and _CITATION.match(text, end):
            return None
        tail = _TAIL.match(text, end)
        if tail is None:
            return None
        token = tail.group(1)
        if token is None or self._is_enumerator(token, with_words=True):
            return family
        return None

    def strip_marker(self, text: str) -> str:
        """``text`` without its leading numbering marker, for fuzzy title matching.

        The pattern is compiled from NFC spellings, so normalize first: a heading
        in NFD (decomposed accents) must strip the same marker as its NFC twin.
        The result feeds matching only; the heading text itself is never rewritten.
        """
        return self._leading_marker.sub("", _nfc(text or ""), count=1)

    def _is_enumerator(self, token: str, with_words: bool = False) -> bool:
        """True for a number, a single letter or a valid Roman numeral.

        Multi-letter tokens are only accepted when they are valid Roman numerals, so
        "Division Civil Remedies" (all Roman-numeral letters) is not mistaken for a marker.
        ``with_words`` also accepts a token that contains a digit ("Chapter 2D",
        "Rule 12(b)") and a number word ("CHAPTER ONE", "ARTICLE FIRST").
        """
        if with_words and (
            any(char.isdigit() for char in token) or token.lower() in self._number_words
        ):
            return True
        return token.isdigit() or len(token) == 1 or is_roman_numeral(token)
