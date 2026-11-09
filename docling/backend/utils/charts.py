# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers shared by the backends that rebuild the data table of a chart.

The PowerPoint and Excel backends read the categories and series of a native
chart in their own way, and then lay them out as a table with the same helper.
"""

import logging
from datetime import datetime, time
from typing import Final

from docling_core.types.doc import TableCell, TableData

_log = logging.getLogger(__name__)

MAX_CHART_TABLE_CELLS: Final = 100_000
"""Backstop on the total cells emitted for one chart's reconstructed data table.

Charts whose declared category range far exceeds their plotted data are trimmed
to populated rows first; this cap only binds genuinely huge charts. 100k cells
still admit e.g. a 10k-point, 5-series chart while keeping any single chart's
markdown output to a few megabytes.
"""


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

    The declared category range can far exceed the plotted data (e.g. a chart
    referencing whole Excel columns declares ~1M categories for a handful of
    points), so trailing rows whose category label and every series value are
    empty are trimmed and the table is sized by populated data instead. As a
    backstop for genuinely huge charts, the emitted table is additionally capped
    at ``MAX_CHART_TABLE_CELLS`` cells.

    Args:
        categories: Category labels, one per data row.
        columns: ``(series name, values)`` pairs, one per chart series.
        corner: Text of the top-left header cell, such as the title of the
            category axis.

    Returns:
        A TableData, or None when no data row is populated.
    """
    num_data_rows = max([len(categories)] + [len(values) for _, values in columns])
    if num_data_rows == 0:
        return None

    # Trim trailing rows that carry no data at all; interior gaps are kept.
    num_data_rows = _last_populated_row(categories, columns, num_data_rows) + 1
    if num_data_rows == 0:
        return None

    num_cols = 1 + len(columns)
    if (num_data_rows + 1) * num_cols > MAX_CHART_TABLE_CELLS:
        capped_rows = max(1, MAX_CHART_TABLE_CELLS // num_cols - 1)
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

    header_labels = [corner] + [name for name, _ in columns]
    for col_idx, label in enumerate(header_labels):
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
        category = categories[data_row] if data_row < len(categories) else ""
        row_texts = [category] + [
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
                    row_header=(col_idx == 0),
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
