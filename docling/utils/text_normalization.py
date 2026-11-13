# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Character-level normalization applied to text extracted from documents."""

import re
import unicodedata
from collections.abc import Iterable

from docling_core.types.doc.page import PdfTextCell, TextCell

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

# Adobe "Symbol" font encoding, byte -> Unicode.
#
# A PDF that draws text with the Symbol font and supplies no usable /ToUnicode
# map -- or supplies one that targets the Private-Use Area, which some
# producers do -- leaves the text layer holding U+F000 + <symbol byte> rather
# than real characters. Those code points render as nothing outside the Symbol
# font, so Greek letters and math operators silently vanish from extracted
# text: "x = Asin(\uf077t + \uf066)" instead of "x = Asin(omega t + phi)".
#
# Deliberately left out:
#  - U+F0A0, which _LIGATURE_MAP above already discards as a spurious glyph.
#  - The eight code points list_marker_processor treats as list markers
#    (0x6E, 0x76, 0xA7, 0xA8, 0xB7, 0xD8, 0xE0, 0xFC). Normalization runs
#    before list_marker_processor, so claiming them would change the list
#    detection added in #5043. Leaving them out keeps it byte-identical.
#  - Glyphs whose own Adobe mapping is a private-use code point: the radical
#    overbar, the arrow extenders, and the multi-part parenthesis, bracket and
#    brace builders. They are drawing fragments with no Unicode equivalent.
#  - The rarer end of the Symbol table: card suits, fraktur and script
#    letters, florin, aleph, lozenge, set-theory operators, angle brackets and
#    the integral halves. Those need the same treatment, but nothing here
#    exercises them, so they are left for a follow-up rather than mapped blind.
_SYMBOL_ENCODING: dict[int, str] = {
    0x20: "\u0020",  # space
    0x21: "\u0021",  # exclamation mark
    0x22: "\u2200",  # for all
    0x23: "\u0023",  # number sign
    0x24: "\u2203",  # there exists
    0x25: "\u0025",  # percent sign
    0x26: "\u0026",  # ampersand
    0x27: "\u220b",  # contains as member
    0x28: "\u0028",  # left parenthesis
    0x29: "\u0029",  # right parenthesis
    0x2A: "\u2217",  # asterisk operator
    0x2B: "\u002b",  # plus sign
    0x2C: "\u002c",  # comma
    0x2D: "\u2212",  # minus sign
    0x2E: "\u002e",  # full stop
    0x2F: "\u002f",  # solidus
    0x30: "\u0030",  # digit zero
    0x31: "\u0031",  # digit one
    0x32: "\u0032",  # digit two
    0x33: "\u0033",  # digit three
    0x34: "\u0034",  # digit four
    0x35: "\u0035",  # digit five
    0x36: "\u0036",  # digit six
    0x37: "\u0037",  # digit seven
    0x38: "\u0038",  # digit eight
    0x39: "\u0039",  # digit nine
    0x3A: "\u003a",  # colon
    0x3B: "\u003b",  # semicolon
    0x3C: "\u003c",  # less-than sign
    0x3D: "\u003d",  # equals sign
    0x3E: "\u003e",  # greater-than sign
    0x3F: "\u003f",  # question mark
    0x41: "\u0391",  # Alpha
    0x42: "\u0392",  # Beta
    0x43: "\u03a7",  # Chi
    0x44: "\u0394",  # Delta
    0x45: "\u0395",  # Epsilon
    0x46: "\u03a6",  # Phi
    0x47: "\u0393",  # Gamma
    0x48: "\u0397",  # Eta
    0x49: "\u0399",  # Iota
    0x4A: "\u03d1",  # theta symbol
    0x4B: "\u039a",  # Kappa
    0x4C: "\u039b",  # Lambda
    0x4D: "\u039c",  # Mu
    0x4E: "\u039d",  # Nu
    0x4F: "\u039f",  # Omicron
    0x50: "\u03a0",  # Pi
    0x51: "\u0398",  # Theta
    0x52: "\u03a1",  # Rho
    0x53: "\u03a3",  # Sigma
    0x54: "\u03a4",  # Tau
    0x55: "\u03a5",  # Upsilon
    0x56: "\u03c2",  # final sigma
    0x57: "\u03a9",  # Omega
    0x58: "\u039e",  # Xi
    0x59: "\u03a8",  # Psi
    0x5A: "\u0396",  # Zeta
    0x5B: "\u005b",  # left square bracket
    0x5C: "\u2234",  # therefore
    0x5D: "\u005d",  # right square bracket
    0x5E: "\u22a5",  # up tack
    0x5F: "\u005f",  # low line
    0x61: "\u03b1",  # alpha
    0x62: "\u03b2",  # beta
    0x63: "\u03c7",  # chi
    0x64: "\u03b4",  # delta
    0x65: "\u03b5",  # epsilon
    0x66: "\u03c6",  # phi
    0x67: "\u03b3",  # gamma
    0x68: "\u03b7",  # eta
    0x69: "\u03b9",  # iota
    0x6A: "\u03d5",  # phi symbol
    0x6B: "\u03ba",  # kappa
    0x6C: "\u03bb",  # lambda
    0x6D: "\u03bc",  # mu
    0x6F: "\u03bf",  # omicron
    0x70: "\u03c0",  # pi
    0x71: "\u03b8",  # theta
    0x72: "\u03c1",  # rho
    0x73: "\u03c3",  # sigma
    0x74: "\u03c4",  # tau
    0x75: "\u03c5",  # upsilon
    0x77: "\u03c9",  # omega
    0x78: "\u03be",  # xi
    0x79: "\u03c8",  # psi
    0x7A: "\u03b6",  # zeta
    0x7B: "\u007b",  # left curly bracket
    0x7C: "\u007c",  # vertical line
    0x7D: "\u007d",  # right curly bracket
    0x7E: "\u223c",  # tilde operator
    0xA3: "\u2264",  # less-than or equal to
    0xAE: "\u2192",  # rightwards arrow
    0xB0: "\u00b0",  # degree
    0xB1: "\u00b1",  # plus-minus
    0xB3: "\u2265",  # greater-than or equal to
    0xB4: "\u00d7",  # multiplication sign
    0xB8: "\u00f7",  # division sign
    0xB9: "\u2260",  # not equal to
    0xBA: "\u2261",  # identical to
    0xBB: "\u2248",  # almost equal to
    0xD6: "\u221a",  # square root
    0xDE: "\u21d2",  # rightwards double arrow
    0xE5: "\u2211",  # n-ary summation
    0xF2: "\u222b",  # integral
}

# Keyed by the code point a PDF text layer actually carries, U+F000 + byte.
_SYMBOL_PUA_MAP: dict[str, str] = {
    chr(0xF000 + byte): char for byte, char in _SYMBOL_ENCODING.items()
}
_SYMBOL_PUA_CHARS = "".join(sorted(_SYMBOL_PUA_MAP))
# Two alternatives: a glyph sitting where a list marker sits -- at the start of
# a line, after optional indent, followed by whitespace or the line end -- or
# any other mapped glyph. The marker branch is first, so it wins at a line
# start. MULTILINE is required because a page backend can put several lines,
# and so several bullets, inside one TextCell.
_SYMBOL_PUA_RE = re.compile(
    r"(?P<marker>^[^\S\n]*[" + _SYMBOL_PUA_CHARS + r"](?=\s|$))"
    r"|(?P<glyph>[" + _SYMBOL_PUA_CHARS + r"])",
    re.MULTILINE,
)


# Subset-embedded fonts are prefixed with six uppercase letters and a plus
# (PDF 32000-1 9.6.4), as in "/AAAAAA+SymbolMT".
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

# Font families whose glyphs this fold applies to.
_SYMBOL_FAMILIES = frozenset({"symbol", "symbolmt", "symbolps"})

# Font families that draw pictures, not letters. Their glyphs occupy the same
# private-use code points as Symbol -- both wingding.ttf and symbol.ttf name
# the glyph for byte 0x6C "uniF06C" -- but a Wingdings U+F06C is a bullet, not
# a lambda. Folding one would put a wrong Greek letter on the page, so text
# drawn with these fonts is never folded.
_PICTORIAL_FAMILIES = frozenset(
    {
        "wingdings",
        "wingdings2",
        "wingdings3",
        "webdings",
        "zapfdingbats",
        "dingbats",
        "marlett",
    }
)


def _font_family(font_name: str) -> str:
    """Reduce a PDF font name to a comparable family key.

    "/AAAAAA+Wingdings-Regular" -> "wingdings". Returns "" when the name is a
    bare resource key such as "/F1", which docling-parse falls back to when the
    font dictionary carries no descriptive name.
    """
    name = _SUBSET_PREFIX.sub("", (font_name or "").lstrip("/"), count=1)
    family = re.split(r"[-_,+ ]", name, maxsplit=1)[0].lower()
    # A bare resource key ("F1", "C2") names no family.
    return "" if re.fullmatch(r"[a-z]\d+", family) else family


def _fold_symbol_pua(text: str, *, protect_markers: bool = True) -> str:
    """Fold Symbol glyphs stranded in the Private-Use Area to real Unicode.

    ``protect_markers`` keeps a glyph that sits where a list marker sits -- at
    the start of a line, after optional indent, followed by whitespace or the
    line end. It is used when the font behind the glyph is not known, because
    there a leading glyph may be a bullet from a pictorial font. Leaving it
    alone keeps the output exactly as it was before this fold existed, so a
    bullet can never get worse, only stay unfixed.

    When the font is known to be Symbol the guard is dropped, because every
    glyph in such a cell is a Symbol glyph, including a leading one.

    shortcut: with the guard on, only the first glyph on a line is protected,
    so a nested marker pair in one line still folds the second one. Revisit if
    a report shows a backend emitting two markers on a single line.
    """

    def replace(match: re.Match[str]) -> str:
        if protect_markers and match.group("marker") is not None:
            return match.group(0)
        return _SYMBOL_PUA_MAP[match.group(0)]

    return _SYMBOL_PUA_RE.sub(replace, text)


def fold_symbol_pua_for_cell(cell: TextCell, text: str) -> str:
    """Apply the Symbol fold to one cell's text, using the cell's font.

    Three cases, because a private-use code point means different things in
    different fonts and the backend does not always know which font drew it:

    * a pictorial font (Wingdings and friends) -- never fold, the glyph is a
      picture and has no Unicode letter behind it;
    * the Symbol font -- fold every glyph, the font settles the meaning;
    * anything else, including the bare resource keys docling-parse reports
      for a line that mixes fonts -- fold, but keep list-marker positions.
    """
    family = _font_family(cell.font_name) if isinstance(cell, PdfTextCell) else ""
    if family in _PICTORIAL_FAMILIES:
        return text
    return _fold_symbol_pua(text, protect_markers=family not in _SYMBOL_FAMILIES)


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

    Symbol glyphs stranded in the Private-Use Area are folded here, per cell,
    rather than in :func:`normalize_extracted_text`. The callers join many
    cells before normalizing, and by then the font behind each glyph is lost;
    a cell still carries it on ``PdfTextCell.font_name``.
    """
    return [
        fold_symbol_pua_for_cell(cell, cell.text.replace("\x02", "-").strip())
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

    This function takes a plain string, so it cannot know which font drew any
    character. The Symbol private-use fold needs that, and runs per cell in
    :func:`cell_text_lines` instead.
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
