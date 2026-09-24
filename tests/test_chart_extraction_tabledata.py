# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for chart CSV table header semantics."""

import pandas as pd
import pytest

from docling.models.stages.chart_extraction.granite_vision import (
    _dataframe_to_tabledata,
    _extract_csv_to_dataframe,
)


@pytest.mark.parametrize(
    ("csv_text", "header_count", "row_header_coords"),
    [
        ("Category,2020,2021\nNorth,85,15\nSouth,10,60", 3, {(1, 0), (2, 0)}),
        ("Category,1,2,3\nInstitutional,85,15,0", 4, {(1, 0)}),
        ("2020,2021,2022\n85,15,0\n10,60,30", 3, set()),
        ("1,2,3\n10,3,4", 3, set()),
        ("1,2,3\n4,5,6", 0, set()),
        ("1,2,3", 0, set()),
    ],
)
def test_chart_csv_table_header_classification(
    csv_text: str, header_count: int, row_header_coords: set[tuple[int, int]]
) -> None:
    table = _dataframe_to_tabledata(_extract_csv_to_dataframe(csv_text))

    assert len(table.table_cells) == table.num_rows * table.num_cols
    assert table.num_rows == csv_text.count("\n") + 1
    assert sum(cell.column_header for cell in table.table_cells) == header_count
    assert {
        (cell.start_row_offset_idx, cell.start_col_offset_idx)
        for cell in table.table_cells
        if cell.row_header
    } == row_header_coords
    assert all(
        cell.start_row_offset_idx == 0
        for cell in table.table_cells
        if cell.column_header
    )


def test_text_in_data_columns_is_not_a_row_header() -> None:
    csv_text = "Category,Year,Value\nNorth,2024,unknown\nSouth,2025,3"
    table = _dataframe_to_tabledata(_extract_csv_to_dataframe(csv_text))
    cells = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.table_cells
    }

    assert cells[1, 0].row_header is True
    assert cells[1, 2].text == "unknown"
    assert cells[1, 2].row_header is False
    assert cells[2, 0].row_header is True


def test_empty_chart_table_has_no_headers() -> None:
    table = _dataframe_to_tabledata(pd.DataFrame())

    assert table.num_rows == 0
    assert table.num_cols == 0
    assert table.table_cells == []
