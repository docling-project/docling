# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The content a Keynote presentation holds, however its container spells it.

A presentation is a list of slides rather than one flow of text, so it is
modelled here instead of reusing :class:`~docling.backend.iwork.content.Content`,
which a Pages document shapes itself to. What sits *on* a slide is the shared
model: the same paragraphs, tables and pictures, read by the same readers.
"""

from typing import NamedTuple

from docling.backend.iwork.content import Block, Comment, Paragraph

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
