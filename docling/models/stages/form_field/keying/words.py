# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Printed words of a page, for key texts finer than the layout's text lines.

A key is often a part of a text line: two keys on one line
("Name ____ Date of birth ____"), or a key glued to the fill run that
marks its blank ("Address:______"). The words of the parsed page, with their
fill runs marked, let the classifier propose such keys.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Characters of fill runs: underscores, dot leaders and ellipses.
FILL_CHARS = "_.…·‥"
_RUN = re.compile(f"[{re.escape(FILL_CHARS)}]+")

# (left, top, right, bottom), top-left origin.
Box = tuple[float, float, float, float]


def _is_fill_run(run: str) -> bool:
    return (
        run.count("_") >= 3
        or len(run) >= 4
        or run.count("…") >= 2
        or run.count("‥") >= 2
    )


def fill_runs(text: str) -> list[tuple[int, int]]:
    """Start and end of every fill run in a word."""
    return [m.span() for m in _RUN.finditer(text) if _is_fill_run(m.group())]


def trim(text: str, box: Box) -> tuple[str, Box] | None:
    """A word without its fill runs, or None when only fill is left.

    The box keeps the horizontal share of the kept characters; fonts are close
    to monospace at the scale of one word. A word without fill comes back
    unchanged.
    """
    runs = fill_runs(text)
    if not runs:
        return text, box
    segments, start = [], 0
    for a, b in runs:
        segments.append((start, a))
        start = b
    segments.append((start, len(text)))
    kept = [(a, b) for a, b in segments if any(c.isalnum() for c in text[a:b])]
    if not kept:
        return None
    first, last = kept[0][0], kept[-1][1]
    while first < last and text[first].isspace():
        first += 1
    n = len(text)
    left, top, right, bottom = box
    width = right - left
    trimmed = " ".join(text[a:b].strip() for a, b in kept if text[a:b].strip())
    return trimmed, (
        round(left + width * first / n, 3),
        top,
        round(left + width * last / n, 3),
        bottom,
    )


@dataclass(frozen=True)
class Word:
    """One printed word: its key text and box, without fill runs."""

    text: str
    box: Box
    raw: Box  # The whole word, fill included.
    fill: bool  # Only a fill run: it bridges gaps but is never key text.
    trimmed: bool  # A fill run was cut off the word.
    line: int | None  # Index of the first text line holding the word.


@dataclass
class PageText:
    words: list[Word]
    lines: list[list[int]]  # Word indices of each text line, in page order.
    h: float  # Median word height.


def page_text(words: Sequence[tuple[str, Box, bool]], lines: Sequence[Box]) -> PageText:
    """Printed words and text lines of a page.

    ``words`` are (text, box, painted) per word cell, in the parser's order;
    a painted word is a widget's own appearance (a prefilled value or a check
    mark) and is left out. ``lines`` are the text line boxes: a line holds the
    words whose centre it contains, and lines that hold none are dropped.
    """
    kept = [
        (text, box)
        for text, box, painted in words
        if not painted and text.strip() and box[2] > box[0] and box[3] > box[1]
    ]
    members: list[list[int]] = []
    for line in lines:
        ids = [
            i
            for i, (_, box) in enumerate(kept)
            if line[0] <= (box[0] + box[2]) / 2 <= line[2]
            and line[1] <= (box[1] + box[3]) / 2 <= line[3]
        ]
        if ids:
            members.append(ids)
    first: dict[int, int] = {}
    for k, ids in enumerate(members):
        for i in ids:
            first.setdefault(i, k)
    out = []
    for i, (text, box) in enumerate(kept):
        cut = trim(text, box)
        if cut is None:
            out.append(Word(text, box, box, True, False, first.get(i)))
        else:
            out.append(Word(cut[0], cut[1], box, False, cut[0] != text, first.get(i)))
    heights = sorted(box[3] - box[1] for _, box in kept)
    h = heights[len(heights) // 2] if heights else 10.0
    return PageText(out, members, h)
