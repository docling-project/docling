# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the charts of an iWork 2013+ document.

Pages, Numbers and Keynote draw their charts with one shared engine, ``TSCH``,
so a chart on a Keynote slide is the same archive as a chart on a Numbers sheet:
a ``TSCH.ChartDrawableArchive`` placing it on the page, carrying the
``TSCH.ChartArchive`` that describes it.

No app keeps a picture of a chart. The archive holds the model the chart is
drawn from — its kind, the data it plots, and references to its styles — so that
is what is read here. Apple has never published the schemas, but the apps carry
their protobuf descriptors, which is where the names below come from (as dumped
by psobot/keynote-parser); every field was then checked against real charts
Apple wrote. The chart kinds in particular were checked against the charts of
the numbers-parser test suite, which are titled after their kind.
"""

import logging
import math
import struct
from typing import NamedTuple

from docling.backend.iwork.archives import iwa_reference_field, safe_fields
from docling.backend.iwork.content import Chart, ChartKind, ChartSeries
from docling.backend.iwork.iwa import IWAObject

_log = logging.getLogger(__name__)

TSCH_CHART_DRAWABLE = 5021
"""Message type of ``TSCH.ChartDrawableArchive``, a chart placed on a page."""

DRAWABLE_CHART_FIELD = 10000
"""Extension field of the drawable holding its ``TSCH.ChartArchive``.

The drawable itself is little more than a frame. The chart rides along in the
first extension field, which is where an archive keeps everything its drawable
superclass does not know about.
"""

CHART_TYPE_FIELD = 1

CHART_SCATTER_FORMAT_FIELD = 2

CHART_SERIES_DIRECTION_FIELD = 5

CHART_GRID_FIELD = 7

CHART_NON_STYLE_FIELD = 10
"""Fields of ``TSCH.ChartArchive``.

They hold the ``TSCH.ChartType`` of the chart, its ``TSCH.ScatterFormat``, its
``TSCH.SeriesDirection``, the ``TSCH.ChartGridArchive`` holding its data, and a
reference to the ``TSCH.ChartNonStyleArchive`` holding its title.
"""

GRID_ROW_NAME_FIELD = 1

GRID_COLUMN_NAME_FIELD = 2

GRID_ROW_FIELD = 3

GRID_ROW_VALUE_FIELD = 1

GRID_VALUE_NUMBER_FIELD = 1
"""Fields of ``TSCH.ChartGridArchive`` and the ``TSCH.GridRow`` it holds.

The grid is laid out the way the chart data editor shows it: a name per row, a
name per column, and a row of values under each row name. A value is a message
rather than a number, since it can also be a date or a duration; only
``numeric_value`` is read, so anything else leaves its point empty.
"""

SERIES_BY_ROW = 1
"""``TSCH.SeriesDirection`` of a chart plotting each row of its grid as a series.

``series_direction_by_column`` (2) plots each column instead, and is also how a
chart that does not say is read. The direction is not a guess: a two-axis chart
only makes sense one way round, and the one in the numbers-parser suite stores
rainfall and temperature as rows and the twelve months as columns, and says 1.
"""

SCATTER_SHARED_X = 2
"""``TSCH.ScatterFormat`` of a scatter chart sharing one x series.

``scatter_format_separate_x`` (1) pairs the series up instead, each x series
followed by the y series plotted against it.
"""

TSCH_CHART_NON_STYLE = 5023
"""Message type of ``TSCH.ChartNonStyleArchive``, which holds a chart's title."""

NON_STYLE_PROPERTIES_FIELD = 10000

NON_STYLE_SHOW_TITLE_FIELD = 21

NON_STYLE_TITLE_FIELD = 23
"""Fields of the generated property map a ``TSCH.ChartNonStyleArchive`` carries.

``tschchartinfodefaultshowtitle`` and ``tschchartinfodefaulttitle``. A chart
stores a title whether or not it shows one — the hidden one in the
numbers-parser suite is still called "Chart 1" — so the title is only read when
the chart says it is shown.
"""

MAX_CHART_CELLS = 100_000
"""The largest grid a chart's data is read from, as rows times its widest row.

That product is the size of the table the data becomes, and an empty value
costs two bytes of the container, so without a bound a small, hostile document
could declare a table that runs to gigabytes once it is built. It is checked
while the grid is read rather than afterwards, and a grid that outgrows it is
dropped rather than read in part, since a chart missing some of its values would
misrepresent it; the chart keeps its kind and title.
"""


class ChartType(NamedTuple):
    """What one ``TSCH.ChartType`` means, in terms of the content model."""

    kind: ChartKind
    stacked: bool = False
    interactive: bool = False


CHART_TYPES: dict[int, ChartType] = {
    1: ChartType(ChartKind.COLUMN),  # columnChartType2D
    2: ChartType(ChartKind.BAR),  # barChartType2D
    3: ChartType(ChartKind.LINE),  # lineChartType2D
    4: ChartType(ChartKind.AREA),  # areaChartType2D
    5: ChartType(ChartKind.PIE),  # pieChartType2D
    6: ChartType(ChartKind.COLUMN, stacked=True),  # stackedColumnChartType2D
    7: ChartType(ChartKind.BAR, stacked=True),  # stackedBarChartType2D
    8: ChartType(ChartKind.AREA, stacked=True),  # stackedAreaChartType2D
    9: ChartType(ChartKind.SCATTER),  # scatterChartType2D
    10: ChartType(ChartKind.MIXED),  # mixedChartType2D
    11: ChartType(ChartKind.MIXED),  # twoAxisChartType2D
    12: ChartType(ChartKind.COLUMN),  # columnChartType3D
    13: ChartType(ChartKind.BAR),  # barChartType3D
    14: ChartType(ChartKind.LINE),  # lineChartType3D
    15: ChartType(ChartKind.AREA),  # areaChartType3D
    16: ChartType(ChartKind.PIE),  # pieChartType3D
    17: ChartType(ChartKind.COLUMN, stacked=True),  # stackedColumnChartType3D
    18: ChartType(ChartKind.BAR, stacked=True),  # stackedBarChartType3D
    19: ChartType(ChartKind.AREA, stacked=True),  # stackedAreaChartType3D
    20: ChartType(ChartKind.COLUMN, interactive=True),  # multiDataColumnChartType2D
    21: ChartType(ChartKind.BAR, interactive=True),  # multiDataBarChartType2D
    22: ChartType(ChartKind.BUBBLE),  # bubbleChartType2D
    23: ChartType(ChartKind.SCATTER, interactive=True),  # multiDataScatterChartType2D
    24: ChartType(ChartKind.BUBBLE, interactive=True),  # multiDataBubbleChartType2D
    25: ChartType(ChartKind.DONUT),  # donutChartType2D
    26: ChartType(ChartKind.DONUT),  # donutChartType3D
    27: ChartType(ChartKind.RADAR),  # radarChartType2D
}
"""``TSCH.ChartType``, the kind of a chart, by the value the archive stores.

The multi-data kinds are what the apps call interactive charts. Anything not
listed here, including ``undefinedChartType`` (0), is read as ``OTHER``.
"""


def iwa_chart(drawable: IWAObject, objects: dict[int, IWAObject]) -> Chart | None:
    """Read one chart, and the data it was last drawn from.

    Args:
        drawable: The ``TSCH.ChartDrawableArchive`` placing the chart.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The chart, or None when the drawable carries no chart archive.
    """
    raw = safe_fields(drawable.payload).get(DRAWABLE_CHART_FIELD, [None])[0]
    if not isinstance(raw, bytes):
        return None

    fields = safe_fields(raw)
    chart_type = CHART_TYPES.get(
        _first_int(fields, CHART_TYPE_FIELD) or 0, ChartType(ChartKind.OTHER)
    )
    categories, series = iwa_chart_grid(
        fields.get(CHART_GRID_FIELD, [None])[0],
        by_row=_first_int(fields, CHART_SERIES_DIRECTION_FIELD) == SERIES_BY_ROW,
    )
    return Chart(
        kind=chart_type.kind,
        title=iwa_chart_title(raw, objects),
        categories=categories,
        series=series,
        stacked=chart_type.stacked,
        interactive=chart_type.interactive,
        shared_x=_first_int(fields, CHART_SCATTER_FORMAT_FIELD) == SCATTER_SHARED_X,
    )


def iwa_chart_grid(
    raw: int | bytes | None, *, by_row: bool
) -> tuple[tuple[str, ...], tuple[ChartSeries, ...]]:
    """Read a chart's data as the categories and the series plotted across them.

    The grid only holds values where it holds rows of them, so its size is taken
    from the values rather than from the names: a name with nothing under it is
    dropped, and a value with no name is kept under an empty one.

    Args:
        raw: The encoded ``TSCH.ChartGridArchive``.
        by_row: Whether the chart plots each row of the grid as a series, rather
            than each column.

    Returns:
        The categories, and one series per row or column of the grid.
    """
    if not isinstance(raw, bytes):
        return (), ()

    grid = safe_fields(raw)
    row_names = [_text(name) for name in grid.get(GRID_ROW_NAME_FIELD, [])]
    column_names = [_text(name) for name in grid.get(GRID_COLUMN_NAME_FIELD, [])]
    rows: list[list[float | None]] = []
    width = 0
    for row in grid.get(GRID_ROW_FIELD, []):
        if not isinstance(row, bytes):
            continue
        values = safe_fields(row).get(GRID_ROW_VALUE_FIELD, [])
        width = max(width, len(values))
        if (len(rows) + 1) * width > MAX_CHART_CELLS:
            _log.debug("Skipping the data of an iWork chart grid too large to read")
            return (), ()
        rows.append([_grid_value(value) for value in values])

    if by_row:
        categories = _named(column_names, width)
        series = tuple(
            ChartSeries(name, tuple(row) + (None,) * (width - len(row)))
            for name, row in zip(_named(row_names, len(rows)), rows)
        )
        return categories, series

    categories = _named(row_names, len(rows))
    series = tuple(
        ChartSeries(
            name, tuple(row[column] if column < len(row) else None for row in rows)
        )
        for column, name in enumerate(_named(column_names, width))
    )
    return categories, series


def iwa_chart_title(chart: bytes, objects: dict[int, IWAObject]) -> str | None:
    """Read the title a chart shows, from the non-style archive holding it.

    Args:
        chart: The encoded ``TSCH.ChartArchive``.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The title, or None when the chart shows none.
    """
    target = iwa_reference_field(chart, CHART_NON_STYLE_FIELD)
    non_style = objects.get(target) if target is not None else None
    if non_style is None or non_style.message_type != TSCH_CHART_NON_STYLE:
        return None

    raw = safe_fields(non_style.payload).get(NON_STYLE_PROPERTIES_FIELD, [None])[0]
    if not isinstance(raw, bytes):
        return None
    properties = safe_fields(raw)
    if _first_int(properties, NON_STYLE_SHOW_TITLE_FIELD) != 1:
        return None
    return _text(properties.get(NON_STYLE_TITLE_FIELD, [None])[0]) or None


def _first_int(fields: dict[int, list[int | bytes]], field: int) -> int | None:
    """Read the first value of a varint field, or None when it has none."""
    value = fields.get(field, [None])[0]
    return value if isinstance(value, int) else None


def _grid_value(raw: int | bytes) -> float | None:
    """Read the number one ``TSCH.GridValue`` holds, or None when it has none."""
    if not isinstance(raw, bytes):
        return None
    number = safe_fields(raw).get(GRID_VALUE_NUMBER_FIELD, [None])[0]
    if not isinstance(number, bytes) or len(number) != 8:
        return None
    value = struct.unpack("<d", number)[0]
    return value if math.isfinite(value) else None


def _named(names: list[str], count: int) -> tuple[str, ...]:
    """Fit a list of names to ``count`` entries, naming the missing ones ""."""
    return tuple(names[:count]) + ("",) * (count - len(names))


def _text(raw: int | bytes | None) -> str:
    """Decode a string field, or return "" when the field is not one."""
    if not isinstance(raw, bytes):
        return ""
    return raw.decode("utf-8", errors="replace").strip()
