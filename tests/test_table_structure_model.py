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
        _cell("h2", 0, 2, 385, 20, 450, 40),
        _cell("a", 1, 0, 385, 40, 450, 60),
        _cell("b", 1, 1, 90, 40, 102, 60),
        _cell("c", 1, 2, 90, 40, 100, 60),
    ]
    before = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in cells]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    after = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in reordered]
    assert before == after


def test_reorder_rows_by_geometry():
    # rows are renumbered when their vertical order disagrees with the boxes
    cells = [
        _cell("top", 1, 0, 90, 20, 150, 40),
        _cell("top2", 1, 1, 237, 20, 300, 40),
        _cell("bottom", 0, 0, 90, 100, 150, 120),
        _cell("bottom2", 0, 1, 237, 100, 300, 120),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=2)
    by_text = {tc.text: tc for tc in reordered}
    assert by_text["top"].start_row_offset_idx == 0
    assert by_text["top2"].start_row_offset_idx == 0
    assert by_text["bottom"].start_row_offset_idx == 1
    assert by_text["bottom2"].start_row_offset_idx == 1


def test_reorder_cells_invalid_span_kept_as_is():
    # spans that run past the grid (corrupt input) are passed through
    cells = [
        _cell("h0", 0, 0, 90, 20, 150, 40),
        _cell("h1", 0, 1, 237, 20, 300, 40),
        _cell("h2", 0, 2, 385, 20, 450, 40),
        _cell("e", 1, 0, 385, 40, 450, 60),
        _cell("f", 1, 1, 237, 40, 300, 60),
        _cell("wide", 1, 1, 90, 40, 450, 60, col_span=3),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    by_text = {tc.text: tc for tc in reordered}
    # the out-of-range span is untouched even while the row is renumbered
    assert by_text["wide"].start_col_offset_idx == 1
    assert by_text["wide"].end_col_offset_idx == 4
    assert by_text["e"].start_col_offset_idx == 2
    assert by_text["f"].start_col_offset_idx == 1
    assert by_text["h0"].start_col_offset_idx == 0
    assert by_text["h2"].start_col_offset_idx == 2


def test_reorder_cells_too_few_measured_cells_keeps_original():
    # when fewer cells carry a box than there are slots, the axis is kept
    # (the violated row below still triggers the remap attempt)
    cells = [
        _cell("a", 0, 0, 385, 20, 450, 40),
        _cell("b", 0, 1, 90, 20, 150, 40),
        TableCell(
            text="c",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
        ),
    ]
    before = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in cells]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    after = [(tc.start_col_offset_idx, tc.end_col_offset_idx) for tc in reordered]
    assert before == after
