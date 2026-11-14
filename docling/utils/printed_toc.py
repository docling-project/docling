# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read the lines of a printed table of contents from the parsed PDF words.

The layout model labels a contents page as ``DOCUMENT_INDEX`` and the table-structure model turns
it into a table. That table is too lossy to read entries from: entries are split across columns
and most page numbers are lost. The parsed words of the same region keep every line intact, with
its dot leaders and its page label.
"""

import re
from dataclasses import dataclass
from statistics import median

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import SegmentedPdfPage

# Leader characters between an entry and its page label.
_LEADER_CHARS = ".…·_"
_LEADER = re.compile(rf"^[{_LEADER_CHARS}\-]+$")
_TRAILING_LEADER = re.compile(rf"[{_LEADER_CHARS}]{{2,}}$")
_LEADING_LEADER = re.compile(rf"^[{_LEADER_CHARS}]{{2,}}")
_ARABIC_LABEL = re.compile(r"^\d{1,4}$")
_ROMAN_LABEL = re.compile(
    r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE
)
# Without leaders, a page label must stand this many word heights away from the entry text.
_LABEL_GAP = 2.0


@dataclass(frozen=True)
class TocLine:
    """One line of a printed table of contents."""

    text: str  # the line without its leaders and page label
    page_label: str | None  # the printed page label, such as "12" or "xvii"
    indent: float  # left edge, relative to the leftmost line of the same region
    height: float  # median word height, a font-size proxy


def page_label_value(label: str) -> tuple[str, int] | None:
    """Return ``("arabic", n)`` or ``("roman", n)`` for a printed page label."""
    if not label:
        return None
    if _ARABIC_LABEL.match(label):
        return "arabic", int(label)
    if not _ROMAN_LABEL.match(label):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for char, following in zip(label.lower(), label.lower()[1:] + " "):
        value = values[char]
        total += -value if following != " " and values[following] > value else value
    return "roman", total


def read_toc_lines(region: BoundingBox, parsed_page: SegmentedPdfPage) -> list[TocLine]:
    """Return the text lines of a contents region, from top to bottom."""
    page_height = parsed_page.dimension.height
    box = region.to_top_left_origin(page_height)
    words: list[tuple[BoundingBox, str]] = []
    for cell in parsed_page.word_cells:
        text = cell.text.strip()
        if not text:
            continue
        wbox = cell.rect.to_bounding_box().to_top_left_origin(page_height)
        cx, cy = (wbox.l + wbox.r) / 2, (wbox.t + wbox.b) / 2
        if box.l <= cx <= box.r and box.t <= cy <= box.b:
            words.append((wbox, text))
    words.sort(key=lambda word: ((word[0].t + word[0].b) / 2, word[0].l))

    rows: list[list[tuple[BoundingBox, str]]] = []
    row_center = 0.0
    for wbox, text in words:
        center = (wbox.t + wbox.b) / 2
        if rows and abs(center - row_center) < 0.5 * (wbox.b - wbox.t):
            rows[-1].append((wbox, text))
        else:
            rows.append([(wbox, text)])
            row_center = center

    lines = [line for line in (_read_row(row) for row in rows) if line is not None]
    if not lines:
        return []
    left = min(line.indent for line in lines)
    return [
        TocLine(
            text=line.text,
            page_label=line.page_label,
            indent=line.indent - left,
            height=line.height,
        )
        for line in lines
    ]


def _read_row(row: list[tuple[BoundingBox, str]]) -> TocLine | None:
    row = sorted(row, key=lambda word: word[0].l)
    tokens: list[tuple[BoundingBox, str]] = []
    leader = False
    for wbox, text in row:
        if _LEADER.match(text):
            leader = True
            continue
        stripped = _LEADING_LEADER.sub("", _TRAILING_LEADER.sub("", text))
        leader = leader or stripped != text
        if stripped:
            tokens.append((wbox, stripped))
    if not tokens:
        return None

    page_label: str | None = None
    if len(tokens) >= 2 and page_label_value(tokens[-1][1]) is not None:
        label_box, label = tokens[-1]
        gap = label_box.l - tokens[-2][0].r
        if leader or gap >= _LABEL_GAP * (label_box.b - label_box.t):
            page_label = label
            tokens = tokens[:-1]

    return TocLine(
        text=" ".join(text for _, text in tokens),
        page_label=page_label,
        indent=tokens[0][0].l,
        height=median(wbox.b - wbox.t for wbox, _ in tokens),
    )
