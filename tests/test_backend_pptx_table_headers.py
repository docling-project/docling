# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the header cells of PowerPoint tables."""

from pathlib import Path

import pytest

from .test_backend_pptx import get_converter


def _header_flags(
    tmp_path: Path, first_row: bool, first_col: bool
) -> dict[tuple[int, int], tuple[bool, bool]]:
    """Convert a one-table deck and map each cell to (column_header, row_header)."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rows = [["City", "Population"], ["Lagos", "15.4"], ["Abuja", "3.8"]]
    table = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(1), Inches(1), Inches(6), Inches(1.5)
    ).table
    # PowerPoint's "Header Row" and "First Column" table style options.
    table.first_row = first_row
    table.first_col = first_col
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    pptx_path = tmp_path / "table_headers.pptx"
    prs.save(pptx_path)
    doc = get_converter().convert(pptx_path).document

    assert len(doc.tables) == 1
    return {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): (
            cell.column_header,
            cell.row_header,
        )
        for cell in doc.tables[0].data.table_cells
    }


@pytest.mark.parametrize(
    "first_row, first_col, column_headers, row_headers",
    [
        # "Header Row" off: no cell is a column header, not even in row 0.
        (False, False, set(), set()),
        # "Header Row" on: row 0 holds the column headers.
        (True, False, {(0, 0), (0, 1)}, set()),
        # "First Column" on: column 0 holds the row headers.
        (False, True, set(), {(0, 0), (1, 0), (2, 0)}),
        # Both on: the top-left cell belongs to the header row only.
        (True, True, {(0, 0), (0, 1)}, {(1, 0), (2, 0)}),
    ],
)
def test_pptx_table_headers_follow_table_style_options(
    tmp_path: Path,
    first_row: bool,
    first_col: bool,
    column_headers: set[tuple[int, int]],
    row_headers: set[tuple[int, int]],
):
    flags = _header_flags(tmp_path, first_row, first_col)

    assert {pos for pos, (col_h, _) in flags.items() if col_h} == column_headers
    assert {pos for pos, (_, row_h) in flags.items() if row_h} == row_headers
