# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the cell spans of PowerPoint tables."""

from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Inches

from .test_backend_pptx import get_converter


def test_pptx_table_merged_cells_keep_their_spans(tmp_path: Path):
    """Merged cells give their spans; the cells they cover give no own cell."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    table = slide.shapes.add_table(3, 3, 0, 0, Inches(6), Inches(3)).table
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(1, 1).merge(table.cell(2, 2))
    for row in range(3):
        for col in range(3):
            if not table.cell(row, col).is_spanned:
                table.cell(row, col).text = f"{row}{col}"
    pptx_path = tmp_path / "merged.pptx"
    prs.save(pptx_path)

    data = get_converter().convert(pptx_path).document.tables[0].data

    assert (data.num_rows, data.num_cols) == (3, 3)
    grid = [[cell.text for cell in row] for row in data.grid]
    assert grid == [
        ["00", "00", "02"],
        ["10", "11", "11"],
        ["20", "11", "11"],
    ]
    spans = {cell.text: (cell.row_span, cell.col_span) for cell in data.table_cells}
    assert spans["00"] == (1, 2)
    assert spans["11"] == (2, 2)


@pytest.mark.parametrize(
    ("grid_span", "row_span", "expected"),
    [
        pytest.param("1000000000", "1", (1, 3), id="wide"),
        pytest.param("1" + "0" * 5000, "2147483647", (2, 3), id="huge"),
        pytest.param("two", "1.5", (1, 1), id="non-integer"),
        pytest.param("0", "-4", (1, 1), id="below-one"),
    ],
)
def test_pptx_table_declared_spans_are_clamped(
    tmp_path: Path, grid_span: str, row_span: str, expected: tuple[int, int]
):
    """A declared span stays within the table and a non-integer one gives 1."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    table = slide.shapes.add_table(2, 3, 0, 0, Inches(6), Inches(2)).table
    table.cell(0, 0).text = "spanning"
    table.cell(1, 2).text = "last"
    tc = table.cell(0, 0)._tc
    tc.set("gridSpan", grid_span)
    tc.set("rowSpan", row_span)
    pptx_path = tmp_path / "spans.pptx"
    prs.save(pptx_path)

    data = get_converter().convert(pptx_path).document.tables[0].data

    assert (data.num_rows, data.num_cols) == (2, 3)
    cell = next(c for c in data.table_cells if c.text == "spanning")
    assert (cell.row_span, cell.col_span) == expected
    assert cell.end_row_offset_idx == expected[0]
    assert cell.end_col_offset_idx == expected[1]
