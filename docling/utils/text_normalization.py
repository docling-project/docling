# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Character-level normalization applied to text extracted from documents."""

import re
import unicodedata
from collections.abc import Iterable

from docling_core.types.doc.page import TextCell

# Typographic characters replaced by plain equivalents.
_TYPOGRAPHIC_MAP: dict[str, str] = {
    "⁄": "/",  # noqa: RUF001
    "’": "'",  # noqa: RUF001
    "‘": "'",  # noqa: RUF001
    "“": '"',
    "”": '"',
    "•": "·",
}

# Ligature normalization map.
# Covers the Unicode Alphabetic Presentation Forms block (U+FB00-U+FB06),
# the Latin capital/small ligature IJ (U+0132/U+0133), and the Private-Use
# Area character U+F0A0 (emitted by some PDF fonts as a spurious glyph).
_LIGATURE_MAP: dict[str, str] = {
    "ﬀ": "ff",  # ﬀ Latin small ligature ff
    "ﬁ": "fi",  # ﬁ Latin small ligature fi
    "ﬂ": "fl",  # ﬂ Latin small ligature fl
    "ﬃ": "ffi",  # ﬃ Latin small ligature ffi
    "ﬄ": "ffl",  # ﬄ Latin small ligature ffl
    "ﬅ": "st",  # ﬅ Latin small ligature long s t
    "ﬆ": "st",  # ﬆ Latin small ligature st
    "Ĳ": "IJ",  # Ĳ Latin capital ligature IJ (used in Dutch)
    "ĳ": "ij",  # ĳ Latin small ligature ij (used in Dutch)
    "": "",  # Private-use glyph emitted by some PDF fonts; discard
}
# Matches any ligature character in the map, optionally followed by a spurious
# space before a word character (to absorb spaces inserted by PDF parsers
# between a ligature glyph and the rest of the word, e.g. "ﬁ eld" → "field").
# Note: U+0132/U+0133 and U+F0A0 are listed as alternates (not a range) to
# avoid an invalid descending range with U+FB06.
_LIGATURE_RE = re.compile(r"([ﬀ-ﬆ]|Ĳ|ĳ|)( (?=\w))?")

# Arabic Presentation Forms-A (U+FB50-U+FDFF) and Forms-B (U+FE70-U+FEFF).
# Some PDF fonts encode Arabic with these compatibility codepoints, which carry
# one codepoint per contextual letter shape (initial/medial/final/isolated)
# instead of the canonical letter. Text stored that way is not searchable or
# comparable against normally-encoded Arabic, since a shaped codepoint never
# matches its U+0600 block equivalent.
_ARABIC_PRESENTATION_RE = re.compile("[ﭐ-﷿ﹰ-﻿]+")


def cell_text_lines(cells: Iterable[TextCell]) -> list[str]:
    """Return the stripped, non-empty texts of the cells.

    Some PDF fonts emit STX (U+0002) for the hyphen glyph; it is restored to "-".
    """
    return [
        cell.text.replace("\x02", "-").strip()
        for cell in cells
        if len(cell.text.strip()) > 0
    ]


def normalize_arabic_presentation_forms(text: str) -> str:
    """Fold Arabic presentation forms back to canonical Arabic letters.

    NFKC is applied per matched run rather than to the whole string, so it can
    never rewrite other scripts: a global NFKC would also fold superscripts,
    subscripts and full-width forms (e.g. "x²" -> "x2").
    """
    return _ARABIC_PRESENTATION_RE.sub(
        lambda m: unicodedata.normalize("NFKC", m.group(0)), text
    )


def normalize_extracted_text(text: str) -> str:
    """Apply all character-level normalization to text extracted from a page.

    Typographic characters are replaced by plain equivalents, ligatures are
    expanded, and Arabic presentation forms are folded to canonical letters.
    """
    for raw, clean in _TYPOGRAPHIC_MAP.items():
        text = text.replace(raw, clean)
    # For the traditional fb00-fb06 ligatures (e.g. ﬁ, ﬂ), any spurious space
    # inserted by the PDF parser between the glyph and the rest of the word is
    # absorbed (e.g. "ﬁ eld" → "field").
    # For U+0132/U+0133 (Dutch IJ/ij) and U+F0A0 (PUA discard glyph), any
    # captured trailing space is re-emitted so that real word boundaries are
    # preserved (e.g. "Ĳ is" → "IJ is", "hello\uf0a0 world" → "hello world").
    text = _LIGATURE_RE.sub(
        lambda m: (
            _LIGATURE_MAP[m.group(1)]
            + ("" if "\ufb00" <= m.group(1) <= "\ufb06" else (m.group(2) or ""))
        ),
        text,
    )
    return normalize_arabic_presentation_forms(text)
