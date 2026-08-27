# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc import BoundingBox, CoordOrigin, TableCell

from docling.models.stages.table_structure.table_structure_model import (
    _reorder_cells_by_geometry,
)


def _cell(
    text: str,
    row: int,
    col: int,
    left: float,
    t: float,
    r: float,
    b: float,
    col_span: int = 1,
) -> TableCell:
    return TableCell(
        text=text,
        row_span=1,
        col_span=col_span,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        bbox=BoundingBox(l=left, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT),
    )


def test_reorder_cells_swapped_columns():
    # #3194: the row header sits on the geometric left but was assigned to the
    # rightmost column; headers are assigned correctly.
    cells = [
        _cell("h0", 0, 0, 90, 20, 150, 40),
        _cell("h1", 0, 1, 237, 20, 300, 40),
        _cell("h2", 0, 2, 385, 20, 450, 40),
        _cell("d0", 1, 0, 237, 40, 300, 60),
        _cell("d1", 1, 1, 385, 40, 450, 60),
        _cell("rh", 1, 2, 90, 40, 150, 60),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    by_text = {tc.text: tc for tc in reordered}
    assert by_text["h0"].start_col_offset_idx == 0
    assert by_text["h1"].start_col_offset_idx == 1
    assert by_text["h2"].start_col_offset_idx == 2
    assert by_text["rh"].start_col_offset_idx == 0
    assert by_text["d0"].start_col_offset_idx == 1
    assert by_text["d1"].start_col_offset_idx == 2


def test_reorder_cells_consistent_table_unchanged():
    cells = [
        _cell("a", 0, 0, 90, 20, 150, 40),
        _cell("b", 0, 1, 237, 20, 300, 40),
        _cell("c", 1, 0, 90, 40, 150, 60),
        _cell("d", 1, 1, 237, 40, 300, 60),
    ]
    before = [
        (
            tc.start_row_offset_idx,
            tc.end_row_offset_idx,
            tc.start_col_offset_idx,
            tc.end_col_offset_idx,
        )
        for tc in cells
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=2)
    after = [
        (
            tc.start_row_offset_idx,
            tc.end_row_offset_idx,
            tc.start_col_offset_idx,
            tc.end_col_offset_idx,
        )
        for tc in reordered
    ]
    assert before == after


def test_reorder_cells_keeps_span_length():
    # a spanning header keeps its span while the columns below are renumbered
    cells = [
        _cell("span", 0, 0, 90, 20, 450, 40, col_span=3),
        _cell("d0", 1, 0, 237, 40, 300, 60),
        _cell("d1", 1, 1, 385, 40, 450, 60),
        _cell("rh", 1, 2, 90, 40, 150, 60),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    by_text = {tc.text: tc for tc in reordered}
    assert (
        by_text["span"].end_col_offset_idx - by_text["span"].start_col_offset_idx == 3
    )
    assert by_text["span"].start_col_offset_idx == 0
    assert by_text["span"].end_col_offset_idx == 3
    assert by_text["rh"].start_col_offset_idx == 0
    assert by_text["d0"].start_col_offset_idx == 1
    assert by_text["d1"].start_col_offset_idx == 2


def test_reorder_cells_collision_keeps_original():
    # two cells of one row nearest the same slot cannot be renumbered
    cells = [
        _cell("h0", 0, 0, 90, 20, 150, 40),
        _cell("h1", 0, 1, 237, 20, 300, 40),
        _cell("x", 1, 0, 90, 40, 300, 60),
        _cell("y", 1, 1, 95, 40, 310, 60),
    ]
    before = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in cells]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=2)
    after = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in reordered]
    assert before == after
