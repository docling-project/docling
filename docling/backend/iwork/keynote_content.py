# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The content a Keynote presentation holds, however its container spells it.

A presentation is a list of slides rather than one flow of text, so it is
modelled here instead of reusing :class:`~docling.backend.iwork.content.Content`,
which a Pages document shapes itself to. What sits *on* a slide is the shared
model: the same paragraphs, tables and pictures, read by the same readers.

Both container generations store a slide's drawables in the order they are
drawn rather than the order they are read, so both need the same repair, and
:func:`reading_order` is here rather than in either reader.
"""

from typing import NamedTuple

from docling.backend.iwork.content import Block, Comment, Geometry, Paragraph

DEFAULT_SLIDE_WIDTH = 1024.0

DEFAULT_SLIDE_HEIGHT = 768.0
"""The slide size Keynote used before widescreen, in points.

It stands in for a presentation whose own size cannot be read, so that every
slide still gets a page of plausible dimensions rather than none.
"""


class Slide(NamedTuple):
    """One slide: what is placed on it, what was said about it, and its notes.

    Presenter notes and comments are kept apart from ``blocks`` rather than
    appended to it: neither is shown when the deck is presented, and both belong
    to the slide as a whole rather than to a position on it.
    """

    blocks: list[Block]
    notes: list[Paragraph] = []
    comments: list[Comment] = []


class Presentation(NamedTuple):
    """Everything one Keynote document holds."""

    slides: list[Slide]
    width: float = DEFAULT_SLIDE_WIDTH
    height: float = DEFAULT_SLIDE_HEIGHT


SLIDE_ROW_TOLERANCE = 3.6
"""How far apart two drawables' top edges may be and still share a row, in points.

0.05 inch, the same band the PowerPoint backend groups shapes into: small
enough that two lines of a slide stay in separate rows, wide enough that an
icon and the label beside it do not.
"""

UNPLACED = 1e9
"""Where a drawable with no readable geometry sorts, in points.

Larger than any slide, so such a drawable falls to the end of it while keeping
its stored position relative to the others there.
"""


def reading_order(placed: list[Geometry | None]) -> list[int]:
    """Order a slide's drawables the way they are read: down, then across.

    Keynote stores them in the order they were stacked, which is the order they
    are drawn in rather than the order they are read in, so a text box added
    after the one above it still comes second. Sorting by position fixes that,
    and keeps a heading next to the body it introduces.

    Drawables whose top edges are within :data:`SLIDE_ROW_TOLERANCE` of the one
    before them share a row and are ordered left to right within it. Adjacency
    is measured against the previous drawable rather than the row's first, so a
    band of shapes that drift downwards stays one row.

    Args:
        placed: Where each drawable sits, in the order the slide stores them.

    Returns:
        Their positions in ``placed``, in reading order.
    """
    entries = [
        (
            geometry.top if geometry is not None else UNPLACED,
            geometry.left if geometry is not None else UNPLACED,
            position,
        )
        for position, geometry in enumerate(placed)
    ]
    entries.sort(key=lambda entry: (entry[0], entry[2]))

    ordered: list[int] = []
    row: list[tuple[float, float, int]] = []
    previous: float | None = None

    for entry in entries:
        if previous is not None and entry[0] - previous > SLIDE_ROW_TOLERANCE:
            ordered.extend(across(row))
            row = []
        row.append(entry)
        previous = entry[0]

    ordered.extend(across(row))
    return ordered


def across(row: list[tuple[float, float, int]]) -> list[int]:
    """Order one row of drawables left to right, ties going to the earlier one."""
    return [entry[2] for entry in sorted(row, key=lambda entry: (entry[1], entry[2]))]
