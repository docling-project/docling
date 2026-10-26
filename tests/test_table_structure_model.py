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
    row_span: int = 1,
) -> TableCell:
    return TableCell(
        text=text,
        row_span=row_span,
        col_span=col_span,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
        bbox=BoundingBox(l=left, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT),
    )


def _offsets(tc: TableCell) -> tuple[int, int, int, int]:
    return (
        tc.start_row_offset_idx,
        tc.end_row_offset_idx,
        tc.start_col_offset_idx,
        tc.end_col_offset_idx,
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


def test_reorder_cells_partial_span_keeps_covered_columns():
    # Review example of #3194: the centre of a two-column span falls between
    # slot centres, so the span must stay over the columns it covers instead
    # of shifting onto the neighbour, while the misordered row is renumbered.
    cells = [
        _cell("S", 0, 0, 90, 20, 300, 40, col_span=2),
        _cell("h2", 0, 2, 385, 20, 450, 40),
        _cell("d0", 1, 0, 237, 40, 300, 60),
        _cell("d1", 1, 1, 385, 40, 450, 60),
        _cell("rh", 1, 2, 90, 40, 150, 60),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    by_text = {tc.text: _offsets(tc) for tc in reordered}
    assert by_text["S"] == (0, 1, 0, 2)
    assert by_text["h2"] == (0, 1, 2, 3)
    assert by_text["d0"] == (1, 2, 1, 2)
    assert by_text["d1"] == (1, 2, 2, 3)
    assert by_text["rh"] == (1, 2, 0, 1)


def test_reorder_cells_moves_misordered_span():
    # a span indexed over the wrong columns moves to the columns it covers,
    # so the fix must not freeze spanning cells in place
    cells = [
        _cell("S", 0, 0, 320, 20, 500, 40, col_span=2),
        _cell("a", 0, 2, 90, 20, 150, 40),
        _cell("b", 0, 3, 210, 20, 270, 40),
        _cell("p", 1, 0, 210, 40, 270, 60),
        _cell("q", 1, 1, 330, 40, 390, 60),
        _cell("r", 1, 2, 450, 40, 510, 60),
        _cell("s", 1, 3, 90, 40, 150, 60),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=4)
    by_text = {tc.text: _offsets(tc) for tc in reordered}
    assert by_text["S"] == (0, 1, 2, 4)
    assert by_text["a"] == (0, 1, 0, 1)
    assert by_text["b"] == (0, 1, 1, 2)
    assert by_text["s"] == (1, 2, 0, 1)
    assert by_text["p"] == (1, 2, 1, 2)
    assert by_text["q"] == (1, 2, 2, 3)
    assert by_text["r"] == (1, 2, 3, 4)


def test_reorder_cells_span_overlap_reverts_axis():
    # the span anchors over the columns already used by the cell beside it,
    # so the collision discards the remap of the whole column axis
    cells = [
        _cell("Y", 0, 1, 70, 20, 310, 40, col_span=2),
        _cell("p", 0, 0, 70, 20, 130, 40),
        _cell("u", 1, 0, 220, 40, 280, 60),
        _cell("v", 1, 1, 390, 40, 450, 60),
        _cell("w", 1, 2, 70, 40, 130, 60),
    ]
    before = {tc.text: _offsets(tc) for tc in cells}
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=3)
    after = {tc.text: _offsets(tc) for tc in reordered}
    assert after == before


def test_reorder_rows_keeps_vertical_span():
    # row-axis analog: a two-row span keeps its rows while the cells of the
    # other column are renumbered to match their boxes
    cells = [
        _cell("V", 0, 0, 90, 20, 150, 120, row_span=2),
        _cell("z", 2, 0, 90, 120, 150, 140),
        _cell("c0", 0, 1, 230, 60, 290, 80),
        _cell("c1", 1, 1, 230, 120, 290, 140),
        _cell("c2", 2, 1, 230, 20, 290, 40),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=3, num_cols=2)
    by_text = {tc.text: _offsets(tc) for tc in reordered}
    assert by_text["V"] == (0, 2, 0, 1)
    assert by_text["z"] == (2, 3, 0, 1)
    assert by_text["c2"] == (0, 1, 1, 2)
    assert by_text["c0"] == (1, 2, 1, 2)
    assert by_text["c1"] == (2, 3, 1, 2)


def test_reorder_cells_full_width_span_stays_at_zero():
    # a span covering every column has a single possible anchor, slot 0
    cells = [
        _cell("span", 0, 0, 90, 20, 530, 40, col_span=4),
        _cell("a", 1, 0, 210, 40, 310, 60),
        _cell("b", 1, 1, 330, 40, 390, 60),
        _cell("c", 1, 2, 450, 40, 510, 60),
        _cell("d", 1, 3, 90, 40, 150, 60),
    ]
    reordered = _reorder_cells_by_geometry(cells, num_rows=2, num_cols=4)
    by_text = {tc.text: _offsets(tc) for tc in reordered}
    assert by_text["span"] == (0, 1, 0, 4)
    assert by_text["d"] == (1, 2, 0, 1)
    assert by_text["a"] == (1, 2, 1, 2)
    assert by_text["b"] == (1, 2, 2, 3)
    assert by_text["c"] == (1, 2, 3, 4)
