# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The content a Numbers document holds, however its container spells it.

Both container generations describe the same things — sheets of positioned
tables, charts and notes — so they are modelled once here and read into that
model by :mod:`docling.backend.iwork.numbers_iwa` and
:mod:`docling.backend.iwork.numbers_xml`. Turning the result into a
:class:`~docling_core.types.doc.DoclingDocument` is the backend's job, which is
what keeps the two readers from having to agree on anything else.

A spreadsheet cell holds a typed value rather than a string, and Numbers stores
it as one. Rendering happens here too, so that the readers agree on what, say,
``4650`` looks like even though one gets it as an IEEE double and the other as
an XML attribute.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from docling.backend.iwork.content import Geometry

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
"""Instant Numbers counts its dates and times from."""


class Cell(NamedTuple):
    """One cell of a table, already rendered to text."""

    row: int
    col: int
    text: str
    col_span: int = 1


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


class Chart(NamedTuple):
    """One chart on a sheet, with the data it is drawn from.

    Numbers caches a chart's data beside the chart rather than only in the table
    it was built from, so a chart reads as a small table of its own: one row per
    category and one column per series, with ``values[category][series]`` holding
    the point, or None where that series has none for that category.

    Which *kind* of chart it is — pie, bar, line — is stored as an integer, and
    the two container generations number them differently. Neither numbering is
    read here, so every chart is classified as a chart of unspecified kind; what
    the chart plots is recovered either way, which is what a reader is after.
    """

    name: str
    categories: list[str]
    series: list[str]
    values: list[list[Decimal | float | None]]
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
    charts: list[Chart]
    comments: list[Comment]


Drawable = Table | Chart | Comment
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


def format_number(value: Decimal | float) -> str:
    """Render a numeric cell the way the spreadsheet shows it, near enough.

    Numbers keeps a cell's number format beside the value rather than in it, so
    a currency or percentage cell is rendered as the plain number it holds. What
    this does guarantee is that a whole number reads as one — ``4650`` rather
    than ``4650.0`` — and that a fraction does not pick up binary floating point
    noise on the way out.

    Args:
        value: The cell's value, exact when it came from a decimal128.

    Returns:
        The value as text, without an exponent.
    """
    if not isinstance(value, Decimal):
        # repr() of a float is its shortest round-tripping form, so this is the
        # decimal the document meant rather than the full binary expansion.
        try:
            value = Decimal(repr(value))
        except InvalidOperation:  # nan and the infinities
            return repr(value)
    return f"{value.normalize():f}"


def format_date(seconds: float) -> str:
    """Render a date cell, which Numbers stores as seconds from its epoch."""
    return (APPLE_EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Render a duration cell, which Numbers stores as a span in seconds."""
    return str(timedelta(seconds=seconds))


def format_bool(value: float) -> str:
    """Render a boolean cell, which Numbers stores as a number."""
    return "True" if value > 0 else "False"


def moment(seconds: float) -> datetime | None:
    """Turn a span of seconds from the Apple epoch into an instant.

    Args:
        seconds: The span the document recorded.

    Returns:
        The instant, or None when it lands outside what a datetime can hold.
    """
    try:
        return APPLE_EPOCH + timedelta(seconds=seconds)
    except (OverflowError, OSError, ValueError):
        return None
