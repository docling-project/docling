from docling_core.types.doc import TableCell

from docling.models.stages.table_structure.table_structure_reconciler import (
    ColumnGridCandidate,
    TextInterval,
    assign_intervals_to_grid,
    build_column_grid_candidate,
    build_table_cells_from_assignment,
    collect_model_cell_metadata_prior,
    find_repeated_vertical_gutters,
    group_intervals_by_row,
    infer_model_row_bands_from_cells,
    reconcile_column_grid_from_intervals,
    reconcile_column_grid_from_text_cells,
    reconcile_columns_preserving_rows_from_text_cells,
    remap_model_cells_to_column_grid,
    select_column_grid_candidate,
)


def interval(left: float, right: float, y: float, text: str = "x") -> TextInterval:
    return TextInterval(
        left=left,
        right=right,
        top=y - 5,
        bottom=y + 5,
        text=text,
    )


def test_group_intervals_by_row_groups_nearby_y_bands():
    intervals = [
        interval(10, 20, 100),
        interval(30, 40, 101),
        interval(10, 20, 140),
    ]

    rows = group_intervals_by_row(intervals)

    assert len(rows) == 2
    assert len(rows[0]) == 2
    assert len(rows[1]) == 1


def test_repeated_vertical_gutters_ignore_large_missing_cell_gaps():
    intervals = [
        # Header/body rows with stable real gutters.
        interval(114, 142, 100, "ROW LABEL"),
        interval(178, 211, 100, "COL A"),
        interval(255, 296, 100, "COL B"),
        interval(371, 395, 100, "COL C"),
        interval(470, 492, 100, "COL D"),
        interval(116, 140, 130, "ROW GROUP C"),
        interval(181, 208, 130, "VAL G"),
        interval(250, 302, 130, "09:00"),
        interval(365, 405, 130, "ITEM E"),
        interval(472, 490, 130, "STATE A"),
        interval(116, 140, 160, "ROW GROUP A"),
        interval(181, 208, 160, "VAL C"),
        interval(250, 302, 160, "10:30"),
        interval(365, 405, 160, "ITEM A"),
        interval(462, 500, 160, "STATE B"),
        # Missing middle cell row: huge gap from COL B directly to COL D.
        # Its midpoint is around 383 and should not become a boundary.
        interval(181, 208, 190, "VAL A"),
        interval(250, 302, 190, "12:00"),
        interval(462, 500, 190, "STATE B"),
        interval(181, 208, 220, "VAL C"),
        interval(250, 302, 220, "14:00"),
        interval(365, 405, 220, "ITEM B"),
        interval(472, 490, 220, "STATE C"),
    ]

    gutters = find_repeated_vertical_gutters(intervals)
    xs = [candidate.x for candidate in gutters]

    assert any(abs(x - 160) <= 10 for x in xs)
    assert any(abs(x - 230) <= 10 for x in xs)
    assert any(abs(x - 333) <= 10 for x in xs)
    assert any(abs(x - 433) <= 10 for x in xs)

    assert not any(abs(x - 383) <= 10 for x in xs)


def test_repeated_vertical_gutters_require_row_consensus():
    intervals = [
        interval(10, 20, 100),
        interval(100, 120, 100),
        interval(10, 20, 130),
    ]

    gutters = find_repeated_vertical_gutters(intervals)

    assert gutters == []


def test_build_column_grid_candidate_from_repeated_gutters():
    intervals = [
        interval(114, 142, 100, "ROW LABEL"),
        interval(178, 211, 100, "COL A"),
        interval(255, 296, 100, "COL B"),
        interval(371, 395, 100, "COL C"),
        interval(470, 492, 100, "COL D"),
        interval(116, 140, 130, "ROW GROUP C"),
        interval(181, 208, 130, "VAL G"),
        interval(250, 302, 130, "09:00"),
        interval(365, 405, 130, "ITEM E"),
        interval(472, 490, 130, "STATE A"),
        interval(116, 140, 160, "ROW GROUP A"),
        interval(181, 208, 160, "VAL C"),
        interval(250, 302, 160, "10:30"),
        interval(365, 405, 160, "ITEM A"),
        interval(462, 500, 160, "STATE B"),
        # Missing middle cell row should not create a fake boundary near 383.
        interval(181, 208, 190, "VAL A"),
        interval(250, 302, 190, "12:00"),
        interval(462, 500, 190, "STATE B"),
        interval(181, 208, 220, "VAL C"),
        interval(250, 302, 220, "14:00"),
        interval(365, 405, 220, "ITEM B"),
        interval(472, 490, 220, "STATE C"),
    ]

    candidate = build_column_grid_candidate(intervals, model_num_cols=4)

    assert candidate is not None
    assert candidate.num_cols == 5
    assert len(candidate.boundaries) == 6
    assert candidate.score > 0

    boundaries = candidate.boundaries
    assert any(abs(x - 160) <= 10 for x in boundaries)
    assert any(abs(x - 230) <= 10 for x in boundaries)
    assert any(abs(x - 333) <= 10 for x in boundaries)
    assert any(abs(x - 433) <= 10 for x in boundaries)
    assert not any(abs(x - 383) <= 10 for x in boundaries)


def test_build_column_grid_candidate_returns_none_without_consensus():
    intervals = [
        interval(10, 20, 100),
        interval(100, 120, 100),
        interval(10, 20, 130),
    ]

    candidate = build_column_grid_candidate(intervals, model_num_cols=4)

    assert candidate is None


def test_assign_intervals_to_grid_places_text_by_column_boundaries():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 55.0, 125.0, 200.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 30, 100, "A"),
        interval(80, 100, 100, "B"),
        interval(150, 170, 100, "C"),
        interval(10, 30, 130, "D"),
        interval(150, 170, 130, "F"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)

    assert assignment.num_rows == 2
    assert assignment.num_cols == 3
    assert assignment.texts_at(0, 0) == ("A",)
    assert assignment.texts_at(0, 1) == ("B",)
    assert assignment.texts_at(0, 2) == ("C",)
    assert assignment.texts_at(1, 0) == ("D",)
    assert assignment.texts_at(1, 1) == ()
    assert assignment.texts_at(1, 2) == ("F",)
    assert assignment.out_of_bounds == []
    assert assignment.score > 1.0


def test_assign_intervals_to_grid_tracks_out_of_bounds_text():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 55.0, 125.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 30, 100, "inside"),
        interval(150, 170, 100, "outside"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)

    assert assignment.texts_at(0, 0) == ("inside",)
    assert [item.text for item in assignment.out_of_bounds] == ["outside"]
    assert assignment.score < 2.0


def test_select_column_grid_candidate_accepts_clean_column_recovery():
    intervals = [
        interval(114, 142, 100, "ROW LABEL"),
        interval(178, 211, 100, "COL A"),
        interval(255, 296, 100, "COL B"),
        interval(371, 395, 100, "COL C"),
        interval(470, 492, 100, "COL D"),
        interval(116, 140, 130, "ROW GROUP C"),
        interval(181, 208, 130, "VAL G"),
        interval(250, 302, 130, "09:00"),
        interval(365, 405, 130, "ITEM E"),
        interval(472, 490, 130, "STATE A"),
        interval(116, 140, 160, "ROW GROUP A"),
        interval(181, 208, 160, "VAL C"),
        interval(250, 302, 160, "10:30"),
        interval(365, 405, 160, "ITEM A"),
        interval(462, 500, 160, "STATE B"),
        interval(181, 208, 190, "VAL A"),
        interval(250, 302, 190, "12:00"),
        interval(462, 500, 190, "STATE B"),
        interval(181, 208, 220, "VAL C"),
        interval(250, 302, 220, "14:00"),
        interval(365, 405, 220, "ITEM B"),
        interval(472, 490, 220, "STATE C"),
    ]

    selection = select_column_grid_candidate(intervals, model_num_cols=4)

    assert selection is not None
    assert selection.candidate.num_cols == 5
    assert selection.assignment.num_cols == 5
    assert selection.assignment.out_of_bounds == []
    assert "accepted" in selection.reason


def test_select_column_grid_candidate_rejects_when_not_improving_model_cols():
    intervals = [
        interval(10, 20, 100),
        interval(80, 100, 100),
        interval(150, 170, 100),
        interval(10, 20, 130),
        interval(80, 100, 130),
        interval(150, 170, 130),
    ]

    selection = select_column_grid_candidate(intervals, model_num_cols=3)

    assert selection is None


def test_select_column_grid_candidate_rejects_noisy_out_of_bounds_candidate():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 50.0, 100.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 20, 100, "inside"),
        interval(150, 170, 100, "outside"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)

    assert assignment.out_of_bounds


def test_build_table_cells_from_assignment_creates_valid_cells():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 55.0, 125.0, 200.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 30, 100, "A"),
        interval(80, 100, 100, "B"),
        interval(150, 170, 100, "C"),
        interval(10, 30, 130, "D"),
        interval(150, 170, 130, "F"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)
    cells = build_table_cells_from_assignment(assignment)

    by_text = {cell.text: cell for cell in cells}

    assert set(by_text) == {"A", "B", "C", "D", "F"}

    assert by_text["A"].start_row_offset_idx == 0
    assert by_text["A"].end_row_offset_idx == 1
    assert by_text["A"].start_col_offset_idx == 0
    assert by_text["A"].end_col_offset_idx == 1

    assert by_text["F"].start_row_offset_idx == 1
    assert by_text["F"].end_row_offset_idx == 2
    assert by_text["F"].start_col_offset_idx == 2
    assert by_text["F"].end_col_offset_idx == 3


def test_build_table_cells_from_assignment_joins_multiple_texts_in_same_slot():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 100.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 20, 100, "Hello"),
        interval(25, 40, 100, "World"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)
    cells = build_table_cells_from_assignment(assignment)

    assert len(cells) == 1
    assert cells[0].text == "Hello World"
    assert cells[0].row_span == 1
    assert cells[0].col_span == 1


def test_reconcile_column_grid_from_intervals_returns_valid_table_grid():
    intervals = [
        interval(114, 142, 100, "ROW LABEL"),
        interval(178, 211, 100, "COL A"),
        interval(255, 296, 100, "COL B"),
        interval(371, 395, 100, "COL C"),
        interval(470, 492, 100, "COL D"),
        interval(116, 140, 130, "ROW GROUP C"),
        interval(181, 208, 130, "VAL G"),
        interval(250, 302, 130, "09:00"),
        interval(365, 405, 130, "ITEM E"),
        interval(472, 490, 130, "STATE A"),
        interval(116, 140, 160, "ROW GROUP A"),
        interval(181, 208, 160, "VAL C"),
        interval(250, 302, 160, "10:30"),
        interval(365, 405, 160, "ITEM A"),
        interval(462, 500, 160, "STATE B"),
        interval(181, 208, 190, "VAL A"),
        interval(250, 302, 190, "12:00"),
        interval(462, 500, 190, "STATE B"),
        interval(181, 208, 220, "VAL C"),
        interval(250, 302, 220, "14:00"),
        interval(365, 405, 220, "ITEM B"),
        interval(472, 490, 220, "STATE C"),
    ]

    reconciled = reconcile_column_grid_from_intervals(
        intervals,
        model_num_cols=4,
    )

    assert reconciled is not None
    assert reconciled.num_cols == 5
    assert reconciled.num_rows == 5
    assert reconciled.diagnostics.valid
    assert reconciled.otsl_seq

    by_text = {cell.text: cell for cell in reconciled.table_cells}

    assert "ROW GROUP A" in by_text
    assert "ITEM A" in by_text
    assert "ITEM B" in by_text

    assert by_text["ROW GROUP A"].start_col_offset_idx == 0
    assert by_text["ITEM A"].start_col_offset_idx == 3
    assert by_text["ITEM B"].start_col_offset_idx == 3


def test_reconcile_column_grid_from_intervals_returns_none_without_candidate():
    intervals = [
        interval(10, 20, 100),
        interval(100, 120, 100),
        interval(10, 20, 130),
    ]

    reconciled = reconcile_column_grid_from_intervals(
        intervals,
        model_num_cols=4,
    )

    assert reconciled is None


def model_cell(
    *,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    text: str = "model",
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
) -> TableCell:
    return TableCell(
        text=text,
        row_span=end_row - start_row,
        col_span=end_col - start_col,
        start_row_offset_idx=start_row,
        end_row_offset_idx=end_row,
        start_col_offset_idx=start_col,
        end_col_offset_idx=end_col,
        column_header=column_header,
        row_header=row_header,
        row_section=row_section,
    )


def test_collect_model_cell_metadata_prior_reads_header_rows_and_cols():
    prior = collect_model_cell_metadata_prior(
        [
            model_cell(
                start_row=0,
                end_row=2,
                start_col=0,
                end_col=3,
                column_header=True,
            ),
            model_cell(
                start_row=2,
                end_row=5,
                start_col=0,
                end_col=1,
                row_header=True,
            ),
            model_cell(
                start_row=5,
                end_row=6,
                start_col=0,
                end_col=3,
                row_section=True,
            ),
        ]
    )

    assert prior.column_header_rows == frozenset({0, 1})
    assert prior.row_header_cols == frozenset({0})
    assert prior.row_section_rows == frozenset({5})


def test_build_table_cells_from_assignment_preserves_safe_model_metadata():
    candidate = ColumnGridCandidate(
        boundaries=(0.0, 55.0, 125.0),
        gutters=(),
        score=1.0,
    )
    intervals = [
        interval(10, 30, 100, "Header A"),
        interval(80, 100, 100, "Header B"),
        interval(10, 30, 130, "Row label"),
        interval(80, 100, 130, "Value"),
    ]

    assignment = assign_intervals_to_grid(intervals, candidate)
    cells = build_table_cells_from_assignment(
        assignment,
        model_cells=[
            model_cell(
                start_row=0,
                end_row=1,
                start_col=0,
                end_col=2,
                column_header=True,
            ),
            model_cell(
                start_row=1,
                end_row=2,
                start_col=0,
                end_col=1,
                row_header=True,
            ),
        ],
    )

    by_text = {cell.text: cell for cell in cells}

    assert by_text["Header A"].column_header
    assert by_text["Header B"].column_header

    assert by_text["Row label"].row_header
    assert not by_text["Value"].row_header

    assert not by_text["Row label"].column_header
    assert not by_text["Value"].column_header


def test_reconcile_column_grid_from_intervals_preserves_model_header_prior():
    intervals = [
        interval(114, 142, 100, "ROW LABEL"),
        interval(178, 211, 100, "COL A"),
        interval(255, 296, 100, "COL B"),
        interval(371, 395, 100, "COL C"),
        interval(470, 492, 100, "COL D"),
        interval(116, 140, 130, "ROW GROUP A"),
        interval(181, 208, 130, "VAL C"),
        interval(250, 302, 130, "10:30"),
        interval(365, 405, 130, "ITEM A"),
        interval(462, 500, 130, "STATE B"),
        interval(116, 140, 160, "ROW GROUP B"),
        interval(181, 208, 160, "Ira"),
        interval(250, 302, 160, "09:00"),
        interval(365, 405, 160, "ITEM C"),
        interval(472, 490, 160, "STATE C"),
    ]

    reconciled = reconcile_column_grid_from_intervals(
        intervals,
        model_num_cols=4,
        model_cells=[
            model_cell(
                start_row=0,
                end_row=1,
                start_col=0,
                end_col=5,
                column_header=True,
            )
        ],
    )

    assert reconciled is not None

    by_text = {cell.text: cell for cell in reconciled.table_cells}

    assert by_text["ROW LABEL"].column_header
    assert by_text["COL A"].column_header
    assert not by_text["ROW GROUP A"].column_header
    assert not by_text["ITEM A"].column_header


class FakeRect:
    def __init__(self, left: float, right: float, top: float, bottom: float):
        self.r_x0 = left
        self.r_x1 = right
        self.r_x2 = right
        self.r_x3 = left
        self.r_y0 = bottom
        self.r_y1 = bottom
        self.r_y2 = top
        self.r_y3 = top


class FakeTextCell:
    def __init__(
        self,
        text: str,
        left: float,
        right: float,
        y: float,
    ):
        self.text = text
        self.orig = text
        self.rect = FakeRect(left, right, y - 5, y + 5)


def fake_text_cell(
    text: str,
    left: float,
    right: float,
    y: float,
) -> FakeTextCell:
    return FakeTextCell(text, left, right, y)


def test_reconcile_column_grid_from_text_cells_uses_docling_rect_cells():
    text_cells = [
        fake_text_cell("ROW LABEL", 114, 142, 100),
        fake_text_cell("COL A", 178, 211, 100),
        fake_text_cell("COL B", 255, 296, 100),
        fake_text_cell("COL C", 371, 395, 100),
        fake_text_cell("COL D", 470, 492, 100),
        fake_text_cell("ROW GROUP C", 116, 140, 130),
        fake_text_cell("VAL G", 181, 208, 130),
        fake_text_cell("09:00", 250, 302, 130),
        fake_text_cell("ITEM E", 365, 405, 130),
        fake_text_cell("STATE A", 472, 490, 130),
        fake_text_cell("ROW GROUP A", 116, 140, 160),
        fake_text_cell("VAL C", 181, 208, 160),
        fake_text_cell("10:30", 250, 302, 160),
        fake_text_cell("ITEM A", 365, 405, 160),
        fake_text_cell("STATE B", 462, 500, 160),
        fake_text_cell("VAL A", 181, 208, 190),
        fake_text_cell("12:00", 250, 302, 190),
        fake_text_cell("STATE B", 462, 500, 190),
        fake_text_cell("VAL C", 181, 208, 220),
        fake_text_cell("14:00", 250, 302, 220),
        fake_text_cell("ITEM B", 365, 405, 220),
        fake_text_cell("STATE C", 472, 490, 220),
    ]

    reconciled = reconcile_column_grid_from_text_cells(
        text_cells,
        model_num_cols=4,
    )

    assert reconciled is not None
    assert reconciled.num_cols == 5
    assert reconciled.diagnostics.valid

    by_text = {cell.text: cell for cell in reconciled.table_cells}

    assert by_text["ROW GROUP A"].start_col_offset_idx == 0
    assert by_text["ITEM A"].start_col_offset_idx == 3
    assert by_text["ITEM B"].start_col_offset_idx == 3


def test_reconcile_column_grid_from_text_cells_returns_none_without_text_geometry():
    assert reconcile_column_grid_from_text_cells([], model_num_cols=4) is None
    assert reconcile_column_grid_from_text_cells(None, model_num_cols=4) is None


class FakeBBox:
    def __init__(
        self,
        left: float,
        right: float,
        top: float = 0.0,
        bottom: float = 10.0,
    ):
        self.l = left
        self.r = right
        self.t = top
        self.b = bottom


class FakeModelCell:
    def __init__(
        self,
        text: str,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        left: float,
        right: float,
    ):
        self.text = text
        self.start_row_offset_idx = start_row
        self.end_row_offset_idx = end_row
        self.row_span = end_row - start_row
        self.start_col_offset_idx = start_col
        self.end_col_offset_idx = end_col
        self.col_span = end_col - start_col
        self.column_header = False
        self.row_header = False
        self.row_section = False
        self.bbox = FakeBBox(left, right, start_row * 30.0, end_row * 30.0)

    def model_copy(self, deep: bool = False):
        del deep
        return FakeModelCell(
            self.text,
            self.start_row_offset_idx,
            self.end_row_offset_idx,
            self.start_col_offset_idx,
            self.end_col_offset_idx,
            self.bbox.l,
            self.bbox.r,
        )


def fake_model_cell(
    text: str,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    left: float,
    right: float,
) -> FakeModelCell:
    return FakeModelCell(
        text,
        start_row,
        end_row,
        start_col,
        end_col,
        left,
        right,
    )


def test_remap_model_cells_to_column_grid_preserves_row_topology():
    column_grid = ColumnGridCandidate(
        boundaries=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
        gutters=(),
        score=1.0,
    )
    model_cells = [
        fake_model_cell("ROW LABEL", 0, 1, 0, 1, 0, 100),
        fake_model_cell("HEADER A", 0, 1, 1, 2, 100, 300),
        fake_model_cell("HEADER B", 0, 1, 2, 4, 300, 500),
        fake_model_cell("ROW GROUP A", 1, 4, 0, 1, 0, 100),
        fake_model_cell("COL A COL B", 1, 2, 1, 2, 100, 300),
        fake_model_cell("COL C", 1, 2, 2, 3, 300, 400),
        fake_model_cell("COL D", 1, 2, 3, 4, 400, 500),
    ]

    reconciled = remap_model_cells_to_column_grid(
        model_cells,
        column_grid,
        model_num_rows=4,
        model_num_cols=4,
    )

    assert reconciled is not None
    assert reconciled.num_rows == 4
    assert reconciled.num_cols == 5
    assert reconciled.diagnostics.valid

    by_text = {cell.text: cell for cell in reconciled.table_cells}

    assert by_text["ROW GROUP A"].start_row_offset_idx == 1
    assert by_text["ROW GROUP A"].end_row_offset_idx == 4
    assert by_text["ROW GROUP A"].row_span == 3

    assert by_text["COL A COL B"].start_col_offset_idx == 1
    assert by_text["COL A COL B"].end_col_offset_idx == 3
    assert by_text["COL A COL B"].col_span == 2

    assert by_text["HEADER B"].start_col_offset_idx == 3
    assert by_text["HEADER B"].end_col_offset_idx == 5
    assert by_text["HEADER B"].col_span == 2


def test_reconcile_columns_preserving_rows_from_text_cells_freezes_rows():
    text_cells = [
        fake_text_cell("ROW LABEL", 10, 40, 100),
        fake_text_cell("HEADER A", 120, 180, 100),
        fake_text_cell("HEADER B", 330, 470, 100),
        fake_text_cell("ROW GROUP A", 10, 40, 130),
        fake_text_cell("COL A", 120, 180, 130),
        fake_text_cell("COL B", 220, 280, 130),
        fake_text_cell("COL C", 330, 370, 130),
        fake_text_cell("COL D", 430, 470, 130),
        fake_text_cell("VAL C", 120, 180, 160),
        fake_text_cell("VAL D", 220, 280, 160),
        fake_text_cell("ITEM A", 330, 370, 160),
        fake_text_cell("STATE B", 430, 470, 160),
        fake_text_cell("VAL A", 120, 180, 190),
        fake_text_cell("VAL B", 220, 280, 190),
        fake_text_cell("STATE B", 430, 470, 190),
    ]
    model_cells = [
        fake_model_cell("ROW LABEL", 0, 1, 0, 1, 0, 100),
        fake_model_cell("HEADER A", 0, 1, 1, 2, 100, 300),
        fake_model_cell("HEADER B", 0, 1, 2, 4, 300, 500),
        fake_model_cell("ROW GROUP A", 1, 4, 0, 1, 0, 100),
        fake_model_cell("COL A COL B", 1, 2, 1, 2, 100, 300),
        fake_model_cell("COL C", 1, 2, 2, 3, 300, 400),
        fake_model_cell("COL D", 1, 2, 3, 4, 400, 500),
    ]

    reconciled = reconcile_columns_preserving_rows_from_text_cells(
        text_cells,
        model_cells=model_cells,
        model_num_rows=4,
        model_num_cols=4,
        table_bbox=FakeBBox(0, 500),
    )

    assert reconciled is not None
    assert reconciled.num_rows == 4
    assert reconciled.num_cols == 5

    # This is the core generic assertion:
    # row topology from the model is preserved exactly.
    assert any(
        cell.start_row_offset_idx == 1
        and cell.end_row_offset_idx == 4
        and cell.row_span == 3
        for cell in reconciled.table_cells
    )

    assert all(
        cell.end_row_offset_idx - cell.start_row_offset_idx == cell.row_span
        for cell in reconciled.table_cells
    )


def test_infer_model_row_bands_from_cells_uses_fixed_model_rows():
    model_cells = [
        fake_model_cell("A", 0, 1, 0, 1, 0, 100),
        fake_model_cell("B", 1, 2, 0, 1, 0, 100),
        fake_model_cell("C", 2, 3, 0, 1, 0, 100),
    ]

    row_bands = infer_model_row_bands_from_cells(model_cells, num_rows=3)

    assert len(row_bands) == 3
    assert [row.row_idx for row in row_bands] == [0, 1, 2]


def test_detect_row_text_contamination_reports_repeated_vertical_bands():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        detect_row_text_contamination,
    )

    model_cells = [
        fake_model_cell("A B", 2, 3, 1, 2, 100, 200),
        fake_model_cell("C D", 2, 3, 2, 3, 200, 300),
        fake_model_cell("E F", 2, 3, 3, 4, 300, 400),
    ]
    model_cells[0].bbox = FakeBBox(100, 200, 100, 140)
    model_cells[1].bbox = FakeBBox(200, 300, 100, 140)
    model_cells[2].bbox = FakeBBox(300, 400, 100, 140)

    intervals = [
        interval(120, 150, 105, "A"),
        interval(120, 150, 132, "B"),
        interval(220, 250, 106, "C"),
        interval(220, 250, 133, "D"),
        interval(320, 350, 107, "E"),
        interval(320, 350, 134, "F"),
    ]

    contaminations = detect_row_text_contamination(model_cells, intervals)

    assert len(contaminations) == 3
    assert {item.text for item in contaminations} == {"A B", "C D", "E F"}
    assert all(item.supporting_columns == 3 for item in contaminations)
    assert all(item.band_count == 2 for item in contaminations)


def test_detect_row_text_contamination_ignores_isolated_multiline_cell():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        detect_row_text_contamination,
    )

    model_cells = [
        fake_model_cell("A B", 2, 3, 1, 2, 100, 200),
        fake_model_cell("C", 2, 3, 2, 3, 200, 300),
    ]
    model_cells[0].bbox = FakeBBox(100, 200, 100, 140)
    model_cells[1].bbox = FakeBBox(200, 300, 100, 140)

    intervals = [
        interval(120, 150, 105, "A"),
        interval(120, 150, 132, "B"),
        interval(220, 250, 118, "C"),
    ]

    contaminations = detect_row_text_contamination(model_cells, intervals)

    assert contaminations == []


def test_detect_row_text_contamination_uses_text_membership_when_bbox_is_between_bands():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        detect_row_text_contamination,
    )

    model_cells = [
        fake_model_cell("A1 A2", 2, 3, 1, 2, 100, 200),
        fake_model_cell("B1 B2", 2, 3, 2, 3, 200, 300),
        fake_model_cell("C1 C2", 2, 3, 3, 4, 300, 400),
    ]

    # The model bbox is between the two raw text bands. Strict bbox containment
    # would miss both bands, but this is still valid row/text contamination
    # evidence because the same vertical split repeats across columns.
    model_cells[0].bbox = FakeBBox(100, 200, 115, 125)
    model_cells[1].bbox = FakeBBox(200, 300, 115, 125)
    model_cells[2].bbox = FakeBBox(300, 400, 115, 125)

    intervals = [
        interval(120, 150, 105, "A1"),
        interval(120, 150, 135, "A2"),
        interval(220, 250, 106, "B1"),
        interval(220, 250, 136, "B2"),
        interval(320, 350, 107, "C1"),
        interval(320, 350, 137, "C2"),
    ]

    contaminations = detect_row_text_contamination(model_cells, intervals)

    assert len(contaminations) == 3
    assert {item.text for item in contaminations} == {"A1 A2", "B1 B2", "C1 C2"}
    assert all(item.supporting_columns == 3 for item in contaminations)


def test_propose_row_boundary_splits_from_repeated_vertical_contamination():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        propose_row_boundary_splits,
    )

    model_cells = [
        fake_model_cell("A1 A2", 2, 3, 1, 2, 100, 200),
        fake_model_cell("B1 B2", 2, 3, 2, 3, 200, 300),
        fake_model_cell("C1 C2", 2, 3, 3, 4, 300, 400),
    ]
    model_cells[0].bbox = FakeBBox(100, 200, 115, 125)
    model_cells[1].bbox = FakeBBox(200, 300, 115, 125)
    model_cells[2].bbox = FakeBBox(300, 400, 115, 125)

    intervals = [
        interval(120, 150, 105, "A1"),
        interval(120, 150, 135, "A2"),
        interval(220, 250, 106, "B1"),
        interval(220, 250, 136, "B2"),
        interval(320, 350, 107, "C1"),
        interval(320, 350, 137, "C2"),
    ]

    proposals = propose_row_boundary_splits(model_cells, intervals)

    assert len(proposals) == 1
    assert proposals[0].start_row == 2
    assert proposals[0].end_row == 3
    assert proposals[0].supporting_columns == 3
    assert proposals[0].supporting_cell_indices == (0, 1, 2)


def test_propose_row_boundary_splits_rejects_single_column_evidence():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        propose_row_boundary_splits,
    )

    model_cells = [
        fake_model_cell("A1 A2", 2, 3, 1, 2, 100, 200),
        fake_model_cell("B1", 2, 3, 2, 3, 200, 300),
    ]
    model_cells[0].bbox = FakeBBox(100, 200, 115, 125)
    model_cells[1].bbox = FakeBBox(200, 300, 115, 125)

    intervals = [
        interval(120, 150, 105, "A1"),
        interval(120, 150, 135, "A2"),
        interval(220, 250, 120, "B1"),
    ]

    proposals = propose_row_boundary_splits(model_cells, intervals)

    assert proposals == []


def test_apply_row_boundary_split_inserts_row_and_reassigns_text_by_y_band():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        apply_row_boundary_split,
        propose_row_boundary_splits,
    )

    model_cells = [
        fake_model_cell("Header", 1, 2, 0, 3, 0, 300),
        fake_model_cell("G2", 2, 3, 0, 1, 0, 100),
        fake_model_cell("A1 A2", 2, 3, 1, 2, 100, 200),
        fake_model_cell("B1 B2", 2, 3, 2, 3, 200, 300),
        fake_model_cell("After", 3, 4, 1, 2, 100, 200),
    ]
    model_cells[0].column_header = True

    for cell in model_cells:
        if cell.text in {"A1 A2", "B1 B2"}:
            cell.bbox = FakeBBox(
                cell.start_col_offset_idx * 100,
                cell.end_col_offset_idx * 100,
                115,
                125,
            )
        elif cell.text == "G2":
            cell.bbox = FakeBBox(
                cell.start_col_offset_idx * 100,
                cell.end_col_offset_idx * 100,
                125,
                145,
            )
        else:
            cell.bbox = FakeBBox(
                cell.start_col_offset_idx * 100,
                cell.end_col_offset_idx * 100,
                160,
                180,
            )

    intervals = [
        interval(20, 50, 135, "G2"),
        interval(120, 150, 105, "A1"),
        interval(120, 150, 135, "A2"),
        interval(220, 250, 106, "B1"),
        interval(220, 250, 136, "B2"),
        interval(120, 150, 170, "After"),
    ]

    splits = propose_row_boundary_splits(model_cells, intervals)
    assert len(splits) == 1

    result = apply_row_boundary_split(
        model_cells,
        intervals,
        num_rows=4,
        num_cols=3,
        split=splits[0],
    )

    assert result is not None
    assert result.num_rows == 5
    assert result.num_cols == 3
    assert result.diagnostics.valid

    observed = {
        (
            cell.start_row_offset_idx,
            cell.end_row_offset_idx,
            cell.start_col_offset_idx,
            cell.end_col_offset_idx,
            cell.text,
        )
        for cell in result.table_cells
    }

    assert (1, 2, 0, 3, "Header") in observed
    assert (2, 3, 1, 2, "A1") in observed
    assert (3, 4, 1, 2, "A2") in observed

    split_header_cells = [
        cell
        for cell in result.table_cells
        if cell.start_row_offset_idx == 2
        and cell.end_row_offset_idx == 3
        and cell.text in {"A1", "B1"}
    ]
    assert split_header_cells
    assert all(cell.column_header for cell in split_header_cells)
    assert (2, 3, 2, 3, "B1") in observed
    assert (3, 4, 2, 3, "B2") in observed
    assert (3, 4, 0, 1, "G2") in observed
    assert (4, 5, 1, 2, "After") in observed


def test_apply_row_boundary_split_rejects_cells_crossing_split_row():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        RowBoundarySplitCandidate,
        apply_row_boundary_split,
    )

    model_cells = [
        fake_model_cell("A", 1, 3, 0, 1, 0, 100),
    ]
    model_cells[0].bbox = FakeBBox(0, 100, 20, 60)

    split = RowBoundarySplitCandidate(
        start_row=2,
        end_row=3,
        boundary_y=45.0,
        supporting_columns=2,
        supporting_cell_indices=(0, 1),
    )

    result = apply_row_boundary_split(
        model_cells,
        [],
        num_rows=4,
        num_cols=1,
        split=split,
    )

    assert result is None


def test_reconcile_row_spans_from_empty_cells_assigns_empty_rows_by_nearest_label():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        reconcile_row_spans_from_empty_cells,
    )

    model_cells = [
        fake_model_cell("Header", 0, 1, 0, 2, 0, 200),
        fake_model_cell("A", 1, 2, 0, 1, 0, 100),
        fake_model_cell("d1", 1, 2, 1, 2, 100, 200),
        fake_model_cell("", 2, 3, 0, 1, 0, 100),
        fake_model_cell("d2", 2, 3, 1, 2, 100, 200),
        fake_model_cell("B", 3, 4, 0, 1, 0, 100),
        fake_model_cell("d3", 3, 4, 1, 2, 100, 200),
        fake_model_cell("", 4, 5, 0, 1, 0, 100),
        fake_model_cell("d4", 4, 5, 1, 2, 100, 200),
        fake_model_cell("", 5, 6, 0, 1, 0, 100),
        fake_model_cell("d5", 5, 6, 1, 2, 100, 200),
        fake_model_cell("C", 6, 7, 0, 1, 0, 100),
        fake_model_cell("d6", 6, 7, 1, 2, 100, 200),
    ]
    model_cells[0].column_header = True

    result = reconcile_row_spans_from_empty_cells(
        model_cells,
        num_rows=7,
        num_cols=2,
    )

    assert result is not None
    assert result.diagnostics.valid

    observed = {
        (
            cell.start_row_offset_idx,
            cell.end_row_offset_idx,
            cell.start_col_offset_idx,
            cell.end_col_offset_idx,
            cell.text,
        )
        for cell in result.table_cells
    }

    assert (1, 2, 0, 1, "A") in observed
    assert (2, 5, 0, 1, "B") in observed
    assert (5, 7, 0, 1, "C") in observed
    assert all(cell.text != "" for cell in result.table_cells)


def test_reconcile_row_spans_from_empty_cells_rejects_header_rows():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        reconcile_row_spans_from_empty_cells,
    )

    model_cells = [
        fake_model_cell("Header A", 0, 1, 0, 1, 0, 100),
        fake_model_cell("", 1, 2, 0, 1, 0, 100),
        fake_model_cell("Header B", 2, 3, 0, 1, 0, 100),
    ]
    for cell in model_cells:
        cell.column_header = True

    result = reconcile_row_spans_from_empty_cells(
        model_cells,
        num_rows=3,
        num_cols=1,
    )

    assert result is None


def test_reconcile_row_spans_from_empty_cells_rejects_single_missing_slot_in_dense_column():
    from docling.models.stages.table_structure.table_structure_reconciler import (
        reconcile_row_spans_from_empty_cells,
    )

    model_cells = [
        fake_model_cell("Header", 0, 1, 0, 1, 0, 100),
        fake_model_cell("A", 1, 2, 0, 1, 0, 100),
        fake_model_cell("", 2, 3, 0, 1, 0, 100),
        fake_model_cell("B", 3, 4, 0, 1, 0, 100),
        fake_model_cell("C", 4, 5, 0, 1, 0, 100),
    ]
    model_cells[0].column_header = True

    result = reconcile_row_spans_from_empty_cells(
        model_cells,
        num_rows=5,
        num_cols=1,
    )

    assert result is None
