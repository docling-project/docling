# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Weight of text whose font name says nothing about weight.
REGULAR_WEIGHT = 400

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
    "bold": 700,
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
