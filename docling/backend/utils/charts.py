# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers shared by the backends that rebuild the data table of a chart.

The PowerPoint and Excel backends read the categories and series of a native
chart in their own way, and then lay them out as a table with the same helpers.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time
from typing import Final

from docling_core.types.doc import PictureClassificationLabel, TableCell, TableData

_log = logging.getLogger(__name__)

MAX_CHART_TABLE_CELLS: Final = 100_000
"""Backstop on the total cells emitted for one chart's reconstructed data table.

Charts whose declared category range far exceeds their plotted data are trimmed
to populated rows first; this cap only binds genuinely huge charts. 100k cells
still admit e.g. a 10k-point, 5-series chart while keeping any single chart's
markdown output to a few megabytes.
"""


def max_chart_data_rows(num_cols: int) -> int:
    """Return the most data rows that a chart table with ``num_cols`` columns keeps.

    The header row and the data rows together stay within
    ``MAX_CHART_TABLE_CELLS`` cells. A table always keeps at least one data row.

    Args:
        num_cols: The number of table columns, category columns included.

    Returns:
        The largest number of data rows.
    """
    return max(1, MAX_CHART_TABLE_CELLS // max(1, num_cols) - 1)


def chart_points_to_list(
    points: Mapping[int, str], max_rows: int, *, warn: bool = True
) -> list[str]:
    """Lay out the cached points of a chart series as one entry per index.

    The list is sized by the points that are present, not by the ``ptCount``
    that the chart part declares, because a part can declare any count. An index
    that is missing between two points gives "", so all the series of a chart
    stay aligned row by row.

    A point at an index of ``max_rows`` or more is dropped. The chart table
    keeps only ``max_rows`` data rows (see ``max_chart_data_rows``), so these
    points would be cut from the table anyway. The points are not compacted
    into the free rows, because that would move them out of line with the
    categories and the other series. Thus one point at a very high index cannot
    make the list longer than ``max_rows``.

    Args:
        points: The text of each point, keyed by its ``idx``.
        max_rows: The most entries to return.
        warn: Log a warning when points are dropped. A caller that reads only
            the first point, such as a series name, turns it off.

    Returns:
        One text per index, from 0 to the highest kept index.
    """
    kept = {index: text for index, text in points.items() if 0 <= index < max_rows}
    if warn and len(kept) < len(points):
        _log.warning(
            "Dropping %d chart points with an index of %d or more.",
            len(points) - len(kept),
            max_rows,
        )
    values = [""] * (max(kept) + 1 if kept else 0)
    for index, text in kept.items():
        values[index] = text
    return values


@dataclass(frozen=True)
class ChartContent:
    """The kind, title and data of a native chart.

    Attributes:
        labels: One classification label per kind of plot, back-most first.
        title: The chart title, or None when the chart shows no title.
        table: The data of the chart, or None when it has no usable data.
    """

    labels: list[PictureClassificationLabel]
    title: str | None
    table: TableData | None


def _last_populated_row(
    categories: list[str], columns: list[tuple[str, list[str]]], num_data_rows: int
) -> int:
    """Return the index of the last data row carrying any chart data.

    A row is populated when its category label or at least one series value is
    non-empty. The scan runs from the end so declared-but-empty trailing ranges
    (e.g. whole-column references such as ``$A$2:$A$1048576``) collapse without
    walking the empty region more than once. Interior gaps are kept: only
    trailing empty rows are trimmed.

    Args:
        categories: Category labels, one per data row (may be shorter or longer
            than ``num_data_rows``).
        columns: ``(series name, values)`` pairs, one per chart series.
        num_data_rows: Untrimmed data-row count.

    Returns:
        The last populated row index, or -1 when no row carries data.
    """
    for row in range(num_data_rows - 1, -1, -1):
        category = categories[row] if row < len(categories) else ""
        if category or any(row < len(values) and values[row] for _, values in columns):
            return row
    return -1


def build_chart_table(
    categories: list[str], columns: list[tuple[str, list[str]]], corner: str = ""
) -> TableData | None:
    """Build the data table of a chart from its categories and series.

    Layout produced (categories down the first column, one column per series):

        | <corner> | <series 0 name> | <series 1 name> | ...
        | cat_0    | val_0,0         | val_1,0         | ...
        | cat_1    | val_0,1         | val_1,1         | ...

    Args:
        categories: Category labels, one per data row.
        columns: ``(series name, values)`` pairs, one per chart series.
        corner: Text of the top-left header cell, such as the title of the
            category axis.

    Returns:
        A TableData, or None when no data row is populated.
    """
    return build_chart_grid([(corner, categories)], columns)


def build_chart_grid(
    category_columns: list[tuple[str, list[str]]],
    value_columns: list[tuple[str, list[str]]],
) -> TableData | None:
    """Build the data table of a chart whose categories can have several levels.

    Layout produced (one column per category level, outer level first, then one
    column per series):

        | <level 0 header> | <level 1 header> | <series 0 name> | ...
        | outer_0          | inner_0          | val_0,0         | ...
        | outer_0          | inner_1          | val_0,1         | ...

    A chart can have no categories, such as a histogram drawn from a list of
    values. Its table then has only the series columns.

    The declared category range can far exceed the plotted data (e.g. a chart
    referencing whole Excel columns declares ~1M categories for a handful of
    points), so trailing rows whose category labels and every series value are
    empty are trimmed and the table is sized by populated data instead. As a
    backstop for genuinely huge charts, the emitted table is additionally capped
    at ``MAX_CHART_TABLE_CELLS`` cells.

    Args:
        category_columns: ``(header, labels)`` pairs, one per category level,
            outer level first. The labels have one entry per data row.
        value_columns: ``(series name, values)`` pairs, one per chart series.

    Returns:
        A TableData, or None when no data row is populated.
    """
    columns = category_columns + value_columns
    num_data_rows = max((len(values) for _, values in columns), default=0)
    if num_data_rows == 0:
        return None

    # Trim trailing rows that carry no data at all; interior gaps are kept.
    num_data_rows = _last_populated_row([], columns, num_data_rows) + 1
    if num_data_rows == 0:
        return None

    num_cols = len(columns)
    if (num_data_rows + 1) * num_cols > MAX_CHART_TABLE_CELLS:
        capped_rows = max_chart_data_rows(num_cols)
        _log.warning(
            "Truncating chart data table from %d to %d data rows: "
            "%d cells would exceed the %d-cell cap.",
            num_data_rows,
            capped_rows,
            (num_data_rows + 1) * num_cols,
            MAX_CHART_TABLE_CELLS,
        )
        num_data_rows = capped_rows

    num_rows = num_data_rows + 1
    cells: list[TableCell] = []

    for col_idx, (label, _) in enumerate(columns):
        cells.append(
            TableCell(
                text=label,
                row_span=1,
                col_span=1,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + 1,
                column_header=True,
                row_header=False,
            )
        )
    for data_row in range(num_data_rows):
        row_idx = data_row + 1
        row_texts = [
            (values[data_row] if data_row < len(values) else "")
            for _, values in columns
        ]
        for col_idx, text in enumerate(row_texts):
            cells.append(
                TableCell(
                    text=text,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=False,
                    row_header=(col_idx < len(category_columns)),
                )
            )

    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def datetime_to_iso(moment: datetime) -> str:
    """Write a date of chart data in ISO 8601.

    A date at midnight becomes a date (``2012-01-01``). A date with a time of
    day becomes a date and time to the second (``2012-01-01T12:30:00``).

    Args:
        moment: The date, such as a category of a date axis.

    Returns:
        The ISO 8601 text.
    """
    if moment.time() == time(0):
        return moment.date().isoformat()
    return moment.isoformat(timespec="seconds")
