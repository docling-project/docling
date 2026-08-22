"""Read weight and slant out of a PDF font name.

PDF font names are the only styling metadata Docling gets from the digital text layer:
``PdfTextCell.font_name`` carries the font dictionary's name (e.g. ``/Helvetica-Bold``,
``/NKDKGK+HelveticaNeueLTPro-Bd``, ``/KIDKQO+Times-Italic``). There is no standard for encoding
weight and slant in that string -- only foundry conventions -- so :func:`parse_font_style`
recognizes the common ones and reports everything else as *unknown* rather than guessing.

Two rules keep the parser conservative, because a false "bold" silently rewrites a heading level
while a miss only falls back to the previous behavior:

1. **Style words are matched as whole tokens**, after splitting the name on separators and
   camel-case boundaries. ``Avenir-Book`` is a regular weight, but the family ``Bookman`` is not.
2. **Abbreviations are only honored as a whole separator-delimited part.** ``-Bd`` is bold, but
   the ``TB`` in ``LinLibertineTB`` and the ``LT`` in ``HelveticaNeueLTPro`` are not read as
   styles -- foundry tags glued onto a family name look exactly like weight abbreviations.

:func:`tally_cell_styles` lifts the same reading from one font name to the cells of one layout
cluster, and :func:`formatting_from_cells` turns that tally into the item-level
:class:`~docling_core.types.doc.common.formatting.Formatting` of a text item.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from docling_core.types.doc.common.formatting import Formatting
from docling_core.types.doc.page import PdfCellRenderingMode, PdfTextCell, TextCell

# Weight of text whose font name says nothing about weight.
REGULAR_WEIGHT = 400
# Lightest weight that reads as bold.
BOLD_WEIGHT = 700

# Subset-embedded fonts are prefixed with six uppercase letters and a plus (PDF 32000-1 9.6.4).
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_SEPARATORS = re.compile(r"[-_,+ ]+")
# camelCase / digit boundaries: "HelveticaNeueLTPro" -> Helvetica, Neue, LT, Pro
_TOKENS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

# Whole style words. Deliberately excludes short foundry tags (LT, MT, PS, Std, Pro, Com).
_WEIGHT_TOKENS = {
    "thin": 100,
    "hairline": 100,
    "extralight": 200,
    "ultralight": 200,
    "light": 300,
    "book": REGULAR_WEIGHT,
    "normal": REGULAR_WEIGHT,
    "plain": REGULAR_WEIGHT,
    "regular": REGULAR_WEIGHT,
    "roman": REGULAR_WEIGHT,  # upright, as in Times-Roman -- not a Roman numeral, not italic
    "medium": 500,
    "demi": 600,
    "demibold": 600,
    "semi": 600,
    "semibold": 600,
    "bold": BOLD_WEIGHT,
    "extrabold": 800,
    "ultrabold": 800,
    "black": 900,
    "fat": 900,
    "heavy": 900,
    "poster": 900,
    "ultra": 900,
}

_ITALIC_TOKENS = frozenset(
    {"italic", "ital", "inclined", "kursiv", "oblique", "slanted"}
)

# Camel-case splitting separates the modifier from the weight ("SemiBold" -> semi, bold), so
# recombine the pairs before looking tokens up individually.
_WEIGHT_MODIFIERS = {
    "semi": {"bold": 600, "light": 350},
    "demi": {"bold": 600, "light": 350},
    "extra": {"bold": 800, "light": 200, "black": 900},
    "ultra": {"bold": 800, "light": 200, "black": 900},
    "x": {"bold": 800, "light": 200},
}

# Abbreviations, honored only when they form a complete separator-delimited part.
# Values are ``(weight, italic)``; ``None`` leaves that aspect unset.
_PART_ABBREVIATIONS: dict[str, tuple[int | None, bool | None]] = {
    "b": (700, None),
    "bd": (700, None),
    "bi": (700, True),
    "bdit": (700, True),
    "blk": (900, None),
    "i": (None, True),
    "ita": (None, True),
    "it": (None, True),
    "lt": (300, None),
    "md": (500, None),
    "obl": (None, True),
    "reg": (REGULAR_WEIGHT, None),
    "rg": (REGULAR_WEIGHT, None),
    "rom": (REGULAR_WEIGHT, None),
    "sb": (600, None),
}


@dataclass(frozen=True)
class _FontStyle:
    """Weight and slant read from a font name (internal)."""

    weight: int = REGULAR_WEIGHT
    italic: bool = False
    known: bool = False  # False when the name carried no recognizable style


def weight_class(weight: int) -> int:
    """Bucket a numeric weight into ``0`` (light/regular), ``1`` (medium/semibold), ``2`` (bold+).

    Coarse on purpose: heading levels are derived from the distinct classes present in a document,
    so a finer scale would split near-identical styles into separate levels.
    """
    if weight >= 700:
        return 2
    if weight >= 500:
        return 1
    return 0


@lru_cache(maxsize=1024)
def parse_font_style(font_name: str) -> _FontStyle:
    """Read weight and slant from a PDF font name.

    Returns a regular, upright style with ``known=False`` when the name carries no recognizable
    styling -- an unstyled family, a foundry-tagged name, or a bare resource key such as ``/F1``
    (docling-parse falls back to the key, or to the literal ``"null"``, when the font dictionary
    has no descriptive name).
    """
    name = _SUBSET_PREFIX.sub("", (font_name or "").lstrip("/"), count=1)
    if not name:
        return _FontStyle()

    weight: int | None = None
    italic: bool | None = None

    for part in _SEPARATORS.split(name):
        if not part:
            continue
        abbreviation = _PART_ABBREVIATIONS.get(part.lower())
        if abbreviation is not None:
            part_weight, part_italic = abbreviation
            weight = part_weight if part_weight is not None else weight
            italic = part_italic if part_italic is not None else italic
            continue

        tokens = [token.lower() for token in _TOKENS.findall(part)]
        index = 0
        while index < len(tokens):
            token = tokens[index]
            modifier = _WEIGHT_MODIFIERS.get(token)
            if modifier is not None and index + 1 < len(tokens):
                combined = modifier.get(tokens[index + 1])
                if combined is not None:
                    weight = combined
                    index += 2
                    continue
            if token in _WEIGHT_TOKENS:
                weight = _WEIGHT_TOKENS[token]
            elif token in _ITALIC_TOKENS:
                italic = True
            index += 1

    if weight is None and italic is None:
        return _FontStyle()
    return _FontStyle(
        weight=weight if weight is not None else REGULAR_WEIGHT,
        italic=bool(italic),
        known=True,
    )


# Share of an item's characters that must come from cells with a readable style before the
# item's styling counts as determined at all.
_STYLE_COVERAGE = 0.8
# Share of those characters that must agree before the attribute is set on the item.
_STYLE_AGREEMENT = 0.8

# Rendering modes that stroke the glyph outline, the classic way of faking a bold face when no
# bold font is embedded (PDF 32000-1 9.3.6).
_STROKED_MODES = frozenset(
    {PdfCellRenderingMode.STROKE_TEXT, PdfCellRenderingMode.FILL_THEN_STROKE}
)


@dataclass(frozen=True)
class _StyleTally:
    """Character-weighted styling of a set of cells (internal)."""

    chars: int = 0  # characters in cells that carry text
    styled_chars: int = 0  # of those, characters whose styling could be read
    italic_chars: int = 0  # of styled_chars, characters in an italic font
    weight_chars: Counter[int] = field(
        default_factory=Counter
    )  # weight class -> characters

    @property
    def coverage(self) -> float:
        return self.styled_chars / self.chars if self.chars else 0.0

    @property
    def italic_share(self) -> float:
        return self.italic_chars / self.styled_chars if self.styled_chars else 0.0

    @property
    def bold_share(self) -> float:
        if not self.styled_chars:
            return 0.0
        return self.weight_chars[weight_class(BOLD_WEIGHT)] / self.styled_chars

    def dominant_weight_class(self) -> int:
        """The weight class carrying the most characters; on a tie the heavier class wins."""
        return max(
            self.weight_chars, key=lambda cls: (self.weight_chars[cls], cls), default=0
        )


def tally_cell_styles(
    cells: Iterable[TextCell], *, use_rendering_mode: bool = False
) -> _StyleTally:
    """Weigh the styling of ``cells`` by the number of characters each one contributes.

    Cells whose styling cannot be read -- OCR output, the pypdfium2 backend, or font names that
    carry no recognizable styling -- still count towards ``chars``, so that a caller can tell a
    confidently plain run from one that was simply never legible.

    With ``use_rendering_mode``, a cell whose font name says nothing but which is drawn with a
    stroked outline is read as bold. It is a fallback for silent font names only: an explicit
    ``-Regular`` is never overridden.
    """
    chars = styled_chars = italic_chars = 0
    weight_chars: Counter[int] = Counter()
    for cell in cells:
        text = (cell.text or "").strip()
        if not text:
            continue
        chars += len(text)
        if not isinstance(cell, PdfTextCell):
            continue
        style = parse_font_style(cell.font_name)
        if style.known:
            styled_chars += len(text)
            weight_chars[weight_class(style.weight)] += len(text)
            if style.italic:
                italic_chars += len(text)
        elif use_rendering_mode and cell.rendering_mode in _STROKED_MODES:
            styled_chars += len(text)
            weight_chars[weight_class(BOLD_WEIGHT)] += len(text)
    return _StyleTally(
        chars=chars,
        styled_chars=styled_chars,
        italic_chars=italic_chars,
        weight_chars=weight_chars,
    )


def formatting_from_cells(
    cells: Iterable[TextCell], *, use_rendering_mode: bool = False
) -> Formatting | None:
    """Item-level bold/italic for the cells of one layout cluster, or ``None`` when undetermined.

    ``Formatting`` describes a whole text item, while a PDF carries a font per cell, so an
    attribute is only reported when the cells agree on it. Two gates, both character-weighted:
    most of the item must be legible at all, and most of the legible part must share the
    attribute. Anything less returns ``None`` -- an emphasis that is wrong is written verbatim
    into the exported text as ``**`` or ``*``, while a miss merely reproduces the behavior of a
    pipeline that reads no styling.

    ``None`` rather than an all-false ``Formatting`` for the same reason: the field is serialized
    with ``exclude_none``, and "not determined" is what actually happened.

    Underline, strikethrough and script stay unset -- a font name carries no signal for them.
    """
    tally = tally_cell_styles(cells, use_rendering_mode=use_rendering_mode)
    if tally.coverage < _STYLE_COVERAGE:
        return None
    bold = tally.bold_share >= _STYLE_AGREEMENT
    italic = tally.italic_share >= _STYLE_AGREEMENT
    return Formatting(bold=bold, italic=italic) if (bold or italic) else None
