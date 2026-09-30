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


def table_width(row_col_spans: Iterable[Sequence[int]]) -> int:
    """Return the number of columns of a table, given the colspans of each row.

    The declared width is the widest row, summing its colspans. Columns to the
    right of the last column in which some cell starts hold no cell of their
    own, so the width stops there, the same way browsers collapse such columns.
    """
    width = 0
    last_start = -1
    for spans in row_col_spans:
        col = 0
        for span in spans:
            last_start = max(last_start, col)
            col += span
        width = max(width, col)
    return min(width, last_start + 1)
