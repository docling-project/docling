# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recovery of word boundaries in text lines whose PDF omits space glyphs.

A PDF is free to separate words by advancing the text position rather than by
painting a space glyph. A producer that writes a caption as one kerned ``TJ`` run,
for example ``[(Cause) -150 (of) -150 (death)] TJ``, leaves no space character
anywhere in the content stream. Word segmentation then has to infer the boundaries
from geometry, and a run whose inter-word advance is small enough is read back as a
single token, e.g. ``Causeofdeathperlegaloutcome``.

The character cells still carry the boundary unambiguously: inside a word the
advance closes up exactly, while between words it leaves a visible gap. This module
re-derives the spacing of an affected line from those per-character rectangles.
"""

import logging
from statistics import median
from typing import TypeVar

from docling_core.types.doc.page import TextCell

_log = logging.getLogger(__name__)

_C = TypeVar("_C", bound=TextCell)

#: Fraction of the line's typical character width above which a horizontal gap is
#: read as a word boundary. Chosen well below the ~0.28 ratio a normal inter-word
#: advance produces and well above the ~0.0 ratio of an intra-word advance, so the
#: exact value is not load-bearing.
DEFAULT_GAP_RATIO = 0.2


def _sorted_chars_in_line(line: TextCell, char_cells: list[_C]) -> list[_C]:
    """Return the character cells lying within ``line``, ordered left to right."""
    line_box = line.rect.to_bounding_box()
    top, bottom = min(line_box.t, line_box.b), max(line_box.t, line_box.b)

    inside = []
    for char in char_cells:
        box = char.rect.to_bounding_box()
        mid_y = (box.t + box.b) / 2
        if top <= mid_y <= bottom and line_box.l <= box.l and box.r <= line_box.r:
            inside.append((box.l, char))

    return [char for _, char in sorted(inside, key=lambda pair: pair[0])]


def _respace(chars: list[_C], gap_ratio: float) -> str:
    """Join ``chars`` into a string, inserting a space at every word-sized gap."""
    boxes = [c.rect.to_bounding_box() for c in chars]

    widths = [b.r - b.l for b in boxes if b.r > b.l]
    if not widths:
        return "".join(c.text for c in chars)

    threshold = median(widths) * gap_ratio

    out = [chars[0].text]
    for prev, char, prev_box, box in zip(chars, chars[1:], boxes, boxes[1:]):
        # Only split between two word characters. Advances around punctuation are
        # wide for reasons that have nothing to do with word boundaries, so
        # "add(a," would otherwise come back as "add (a ,".
        splittable = prev.text[-1:].isalnum() and char.text[:1].isalnum()
        if splittable and box.l - prev_box.r > threshold:
            out.append(" ")
        out.append(char.text)

    return "".join(out)


def recover_word_boundaries(
    textline_cells: list[_C],
    char_cells: list[_C],
    gap_ratio: float = DEFAULT_GAP_RATIO,
) -> int:
    """Restore missing inter-word spaces in text lines, in place.

    Only lines that came back with no space at all are considered, so a line whose
    spacing the parser already resolved is never rewritten. A line is repaired only
    when its character cells reproduce its text exactly, which keeps the rewrite off
    any line the cells cannot account for, and a space is only ever inserted between
    two word characters, which keeps it off code and formulas.

    Args:
        textline_cells: The page's text lines. Repaired lines are mutated in place.
        char_cells: The page's character cells, in the same coordinate origin.
        gap_ratio: Gap size that counts as a word boundary, as a fraction of the
            line's median character width.

    Returns:
        The number of text lines that were rewritten.
    """
    if not char_cells:
        return 0

    repaired = 0
    for line in textline_cells:
        if " " in line.text or len(line.text) < 2:
            continue

        chars = _sorted_chars_in_line(line, char_cells)
        if len(chars) < 2 or "".join(c.text for c in chars) != line.text:
            continue

        respaced = _respace(chars, gap_ratio)
        if respaced == line.text:
            continue

        _log.debug("recovered word boundaries: %r -> %r", line.text, respaced)
        line.text = respaced
        line.orig = respaced
        repaired += 1

    return repaired
