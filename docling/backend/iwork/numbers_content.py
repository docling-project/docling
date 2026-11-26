# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The content a Numbers document holds, however its container spells it.

Both container generations describe the same things — sheets of positioned
tables, charts, pictures and comments — so they are modelled once here and read
into that model by :mod:`docling.backend.iwork.numbers_iwa` and
:mod:`docling.backend.iwork.numbers_xml`. Turning the result into a
:class:`~docling_core.types.doc.DoclingDocument` is the backend's job, which is
what keeps the two readers from having to agree on anything else.
"""

from datetime import datetime
from typing import NamedTuple

from docling.backend.iwork.content import Chart, Geometry, Picture

MAX_TABLE_CELLS = 4_000_000
"""Cells one table may declare before it is rejected as implausible.

The row and column counts come from the document, so a corrupt or hostile one can
declare a grid far larger than it stores. Numbers' own ceiling is a million cells
per table, and both readers hold a table to this one.
"""


class Cell(NamedTuple):
    """One cell of a table, already rendered to text.

    ``picture`` is the image the cell is filled with. Numbers draws it behind
    the cell's text, and a cell can hold a picture and no text.
    """

    row: int
    col: int
    text: str
    col_span: int = 1
    picture: Picture | None = None


class Table(NamedTuple):
    """One table on a sheet.

    ``geometry`` is the table's frame on the sheet canvas in points, which is
    what Numbers positions a table with; it is None when the document does not
    say where the table sits.
    """

    name: str
    num_rows: int
    num_cols: int
    header_rows: int
    header_cols: int
    cells: list[Cell]
    geometry: Geometry | None


class PlacedChart(NamedTuple):
    """One chart on a sheet, and where Numbers put it.

    The chart itself is the shared :class:`~docling.backend.iwork.content.Chart`
    every iWork app's charts are read into; what a sheet adds is a position,
    since a sheet is a canvas and the order things are read in comes from where
    they sit.
    """

    chart: Chart
    geometry: Geometry | None


class PlacedPicture(NamedTuple):
    """One picture on a sheet, and where Numbers put it.

    A picture in a group takes the frame of the group, since that is what sits
    on the sheet.
    """

    picture: Picture
    geometry: Geometry | None


class Comment(NamedTuple):
    """One comment on a sheet, which Numbers calls a sticky note."""

    text: str
    author: str
    timestamp: datetime | None
    geometry: Geometry | None


class Sheet(NamedTuple):
    """One sheet of the document, with what is drawn on it in reading order."""

    name: str
    tables: list[Table]
    charts: list[PlacedChart]
    pictures: list[PlacedPicture]
    comments: list[Comment]


Drawable = Table | PlacedChart | PlacedPicture | Comment
"""Anything a sheet places on its canvas."""


def sheet_order(drawable: Drawable) -> tuple[float, float]:
    """Sort key placing a drawable by where it sits, top row first.

    Numbers keeps a sheet's drawables in z-order — the order they were added,
    not the order a reader meets them — so a summary added after the register it
    summarises would otherwise come second.

    A sheet is sorted more simply than a slide is by
    :func:`~docling.backend.iwork.keynote_content.reading_order`: things on a
    sheet are laid out down the page, so top edge then left edge is enough, and
    there is no band of shapes across a row to hold together.
    """
    if drawable.geometry is None:
        return (0.0, 0.0)
    return (drawable.geometry.top, drawable.geometry.left)
