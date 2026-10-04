# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers to keep declared table cell spans within the actual table size.

Declared ``rowspan`` / ``colspan`` values can be far larger than the table they
belong to. Clamping them keeps the table grid, and the work spent filling it,
proportional to the cells that are really present.
"""

from collections.abc import Iterable, Sequence

# Upper bounds from the HTML table model
# (https://html.spec.whatwg.org/multipage/tables.html#attr-tdth-colspan).
MAX_COLSPAN = 1000
MAX_ROWSPAN = 65534


def clamp_span(span: int, limit: int) -> int:
    """Clamp a declared span to the range ``[1, limit]``."""
    return max(1, min(span, limit))


def declared_table_width(row_cell_spans: Sequence[Sequence[tuple[int, int]]]) -> int:
    """Return the width the rows declare, ignoring what a rowspan shifts.

    The declared width is the widest row, summing its colspans. Columns to the
    right of the last column in which some cell starts hold no cell of their
    own, so the width stops there, the same way browsers collapse such columns.
    """
    width = 0
    last_start = -1
    for cells in row_cell_spans:
        col = 0
        for col_span, _ in cells:
            last_start = max(last_start, col)
            col += col_span
        width = max(width, col)
    return min(width, last_start + 1)


def table_width(row_cell_spans: Iterable[Sequence[tuple[int, int]]]) -> int:
    """Return the number of columns of a table, given the spans of each row.

    Each row is a sequence of ``(col_span, row_span)`` pairs in document order.
    A cell goes in the first column of its row that no earlier cell still
    covers, so a cell whose rowspan reaches into a later row shifts the cells of
    that row to the right, and the table is wide enough to keep them: without
    that, the shifted cells start at or past the table's last column and the
    grid drops them.

    A cell covers the columns it holds inside the declared width, the same way
    the grid trims a colspan that sticks out of the table: a declared span that
    is already wider than the table cannot also push the rows below it sideways,
    which would widen the grid for every row it reaches.
    """
    rows = list(row_cell_spans)
    declared = declared_table_width(rows)
    # Last row, exclusive, that the cells placed so far cover the column in.
    covered_until: dict[int, int] = {}
    last_start = -1
    for row_idx, cells in enumerate(rows):
        col = 0
        for col_span, row_span in cells:
            while covered_until.get(col, 0) > row_idx:
                col += 1
            last_start = max(last_start, col)
            for c in range(col, min(col + col_span, declared)):
                covered_until[c] = max(covered_until.get(c, 0), row_idx + row_span)
            col += col_span
    return max(declared, last_start + 1)
