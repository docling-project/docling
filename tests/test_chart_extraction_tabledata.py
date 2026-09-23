# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pandas as pd

from docling.models.stages.chart_extraction.granite_vision import (
    _dataframe_to_tabledata,
)


def test_numeric_series_labels_are_column_headers():
    df = pd.DataFrame([["Region", "2020", "2021"], ["North", "10", "n/a"]])

    table = _dataframe_to_tabledata(df)
    cells = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c for c in table.table_cells
    }

    assert table.num_rows == 2
    assert [cells[0, i].text for i in range(3)] == ["Region", "2020", "2021"]
    assert all(cells[0, i].column_header for i in range(3))
    assert cells[1, 0].row_header
    assert not cells[1, 2].row_header


def test_numeric_first_row_remains_data():
    table = _dataframe_to_tabledata(pd.DataFrame([[1, 2], [3, 4]]))

    assert table.num_rows == 2
    assert not any(cell.column_header or cell.row_header for cell in table.table_cells)
