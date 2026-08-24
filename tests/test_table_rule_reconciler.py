"""Tests for rule-based table row reconciliation (issue #4028).

TableFormer predicts row boundaries visually and, on dense ruled tables, can
cut a wrapped cell one text line too early, emitting the tail of one row as
the head of the next. The reconciler re-bins the matched words into the bands
delimited by the horizontal rules drawn in the PDF's vector layer.

The geometry in these tests mirrors the measured failure: the predicted
row-0/row-1 boundary sits one text line above the drawn rule, so the last
wrapped line overlaps only the following row's predicted cell.
"""

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Table
from docling.utils.table_rule_reconciler import reconcile_table_rows_with_rules

TABLE_BBOX = BoundingBox(l=60.0, t=60.0, r=540.0, b=260.0)

# Interior rules drawn at the true row boundaries.
RULES = [
    BoundingBox(l=62.0, t=120.0, r=538.0, b=120.0),
    BoundingBox(l=62.0, t=180.0, r=538.0, b=120.0 + 60.0),
]


def _rule(y: float, left: float = 62.0, right: float = 538.0) -> BoundingBox:
    return BoundingBox(l=left, t=y, r=right, b=y)


def _word(
    index: int, text: str, left: float, t: float, right: float, b: float
) -> TextCell:
    return TextCell(
        index=index,
        text=text,
        orig=text,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=t, r=right, b=b, coord_origin=CoordOrigin.TOPLEFT)
        ),
        from_ocr=False,
    )


def _cell(
    row: int, col: int, bbox: BoundingBox, text: str, row_span: int = 1
) -> TableCell:
    return TableCell(
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + 1,
        row_span=row_span,
        col_span=1,
        bbox=bbox,
        text=text,
    )


def _words() -> list[TextCell]:
    """Word cells for a 3-row, 2-column glossary with one wrapped definition."""
    return [
        # row 0 (band 60..120)
        _word(0, "Alpha", 65, 65, 110, 75),
        _word(1, "first", 200, 65, 240, 75),
        _word(2, "line", 245, 65, 275, 75),
        _word(3, "second", 200, 80, 250, 90),
        _word(4, "line", 255, 80, 285, 90),
        # the wrapped tail: above the rule at y=120, belongs to row 0
        _word(5, "stolen", 200, 105, 245, 115),
        _word(6, "tail", 250, 105, 275, 115),
        # row 1 (band 120..180)
        _word(7, "Beta", 65, 125, 100, 135),
        _word(8, "row", 200, 125, 228, 135),
        _word(9, "two", 233, 125, 260, 135),
        _word(10, "text", 265, 125, 295, 135),
        # row 2 (band 180..260)
        _word(11, "Gamma", 65, 185, 120, 195),
        _word(12, "third", 200, 185, 238, 195),
        _word(13, "row", 243, 185, 270, 195),
    ]


def _mispredicted_table() -> Table:
    """A table whose predicted row-0/row-1 cut stole the wrapped tail.

    The predicted row-0 definition cell ends above the tail line, and the
    predicted row-1 definition cell starts inside it, so matching assigned
    the tail words to row 1 — the exact failure of issue #4028.
    """
    cells = [
        _cell(0, 0, BoundingBox(l=63, t=63, r=192, b=100), "Alpha"),
        _cell(0, 1, BoundingBox(l=198, t=63, r=537, b=100), "first line second line"),
        _cell(1, 0, BoundingBox(l=63, t=112, r=192, b=165), "Beta"),
        _cell(
            1, 1, BoundingBox(l=198, t=112, r=537, b=165), "stolen tail row two text"
        ),
        _cell(2, 0, BoundingBox(l=63, t=182, r=192, b=225), "Gamma"),
        _cell(2, 1, BoundingBox(l=198, t=182, r=537, b=225), "third row"),
    ]
    return Table(
        otsl_seq=[],
        table_cells=cells,
        num_rows=3,
        num_cols=2,
        id=0,
        page_no=0,
        label=DocItemLabel.TABLE,
        cluster=Cluster(id=0, label=DocItemLabel.TABLE, bbox=TABLE_BBOX),
    )


def _cell_text(table: Table, row: int, col: int) -> str:
    for cell in table.table_cells:
        if cell.start_row_offset_idx == row and cell.start_col_offset_idx == col:
            return cell.text
    raise AssertionError(f"no cell at ({row}, {col})")


def test_moves_wrapped_tail_into_its_ruled_band():
    table = _mispredicted_table()
    changed = reconcile_table_rows_with_rules(
        table, rules=[_rule(120.0), _rule(180.0)], word_cells=_words()
    )
    assert changed is True
    assert _cell_text(table, 0, 1) == "first line second line stolen tail"
    assert _cell_text(table, 1, 1) == "row two text"
    assert _cell_text(table, 1, 0) == "Beta"
    assert _cell_text(table, 2, 1) == "third row"


def test_no_text_lost_or_duplicated():
    table = _mispredicted_table()
    reconcile_table_rows_with_rules(
        table, rules=[_rule(120.0), _rule(180.0)], word_cells=_words()
    )
    rebuilt = " ".join(c.text for c in table.table_cells if c.text).split()
    assert sorted(rebuilt) == sorted(w.text for w in _words())


def test_faithful_no_op_on_correctly_predicted_table():
    table = _mispredicted_table()
    # Repair the prediction: tail belongs to row 0 and the cell boxes agree
    # with the ruled bands.
    for cell in table.table_cells:
        if cell.start_row_offset_idx == 0 and cell.start_col_offset_idx == 1:
            cell.text = "first line second line stolen tail"
            cell.bbox = BoundingBox(l=198, t=63, r=537, b=118)
        if cell.start_row_offset_idx == 1 and cell.start_col_offset_idx == 1:
            cell.text = "row two text"
            cell.bbox = BoundingBox(l=198, t=123, r=537, b=165)
    before = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c.text
        for c in table.table_cells
    }
    reconcile_table_rows_with_rules(
        table, rules=[_rule(120.0), _rule(180.0)], word_cells=_words()
    )
    after = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c.text
        for c in table.table_cells
    }
    assert after == before


def test_declines_without_rules():
    table = _mispredicted_table()
    changed = reconcile_table_rows_with_rules(table, rules=[], word_cells=_words())
    assert changed is False
    assert _cell_text(table, 1, 1) == "stolen tail row two text"


def test_declines_when_bands_disagree_with_predicted_row_count():
    table = _mispredicted_table()
    # One interior rule -> two bands, but the prediction has three rows.
    changed = reconcile_table_rows_with_rules(
        table, rules=[_rule(120.0)], word_cells=_words()
    )
    assert changed is False
    assert _cell_text(table, 1, 1) == "stolen tail row two text"


def test_declines_on_spanning_cells():
    table = _mispredicted_table()
    table.table_cells[0] = _cell(
        0, 0, BoundingBox(l=63, t=63, r=192, b=165), "Alpha", row_span=2
    )
    changed = reconcile_table_rows_with_rules(
        table, rules=[_rule(120.0), _rule(180.0)], word_cells=_words()
    )
    assert changed is False


def test_ignores_short_rules_that_do_not_span_the_table():
    table = _mispredicted_table()
    # An underline-like stroke: horizontal but far too short to be a row rule.
    changed = reconcile_table_rows_with_rules(
        table,
        rules=[_rule(120.0), _rule(180.0), _rule(95.0, left=200.0, right=260.0)],
        word_cells=_words(),
    )
    assert changed is True
    assert _cell_text(table, 0, 1) == "first line second line stolen tail"
