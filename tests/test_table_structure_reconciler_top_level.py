from types import SimpleNamespace

from docling_core.types.doc import TableCell

from docling.models.stages.table_structure import (
    table_structure_reconciler as reconciler_module,
)
from docling.models.stages.table_structure.table_structure_reconciler import (
    reconcile_table_structure,
)
from docling.models.stages.table_structure.table_topology import (
    validate_table_topology,
)


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


V1_PRODUCTION_RECONCILIATION_FLAGS = {
    "enable_undersegmentation_fallback": False,
    "enable_overspan_fallback": True,
    "allow_same_row_count": True,
    "allow_column_count_growth": False,
    "enable_column_reconciliation": False,
    "enable_row_boundary_reconciliation": False,
    "enable_row_span_reconciliation": False,
}


V2_PRODUCTION_RECONCILIATION_FLAGS = {
    "enable_undersegmentation_fallback": True,
    "enable_overspan_fallback": False,
    "allow_same_row_count": False,
    "allow_column_count_growth": True,
    "enable_column_reconciliation": True,
    "enable_row_boundary_reconciliation": True,
    "enable_row_span_reconciliation": True,
}


def geometry_text_cell(text: str, *, row: int, col: int):
    center_x = 50 + col * 100
    center_y = 10 + row * 20
    return SimpleNamespace(
        text=text,
        bbox=SimpleNamespace(
            l=center_x - 20,
            t=center_y - 5,
            r=center_x + 20,
            b=center_y + 5,
        ),
    )


def attach_geometry_bbox(cell, *, left: float, top: float, right: float, bottom: float):
    object.__setattr__(
        cell,
        "bbox",
        SimpleNamespace(l=left, t=top, r=right, b=bottom),
    )
    return cell


def _cell_signature(cells):
    return [
        (
            cell.start_row_offset_idx,
            cell.end_row_offset_idx,
            cell.start_col_offset_idx,
            cell.end_col_offset_idx,
            cell.text,
        )
        for cell in cells
    ]


def _assert_forced_fallback_preserves_incumbent(
    monkeypatch,
    baseline_cells,
    fallback_cells,
    *,
    num_rows: int,
    num_cols: int,
    otsl_seq: list[str],
    source_cells=None,
):
    def force_overspan_detection(*args, **kwargs):
        return True

    def force_fallback(cells, num_rows, num_cols, otsl_seq, **kwargs):
        return (
            fallback_cells,
            num_rows,
            num_cols,
            list(otsl_seq),
            validate_table_topology(
                fallback_cells,
                num_rows,
                num_cols,
            ),
        )

    monkeypatch.setattr(
        reconciler_module,
        "looks_overspanned_by_text_geometry",
        force_overspan_detection,
    )
    monkeypatch.setattr(
        reconciler_module,
        "infer_table_from_text_geometry",
        force_fallback,
    )

    result = reconcile_table_structure(
        baseline_cells,
        num_rows=num_rows,
        num_cols=num_cols,
        otsl_seq=otsl_seq,
        text_cells=source_cells or [geometry_text_cell("A", row=0, col=0)],
        **V1_PRODUCTION_RECONCILIATION_FLAGS,
    )

    assert _cell_signature(result.table_cells) == _cell_signature(baseline_cells)
    assert "incumbent_preserved" in result.notes


def test_reconcile_v1_production_flags_returns_same_row_count_overspan_repair():
    cells = [
        model_cell(start_row=0, end_row=1, start_col=1, end_col=4, text="Seminar"),
        model_cell(start_row=1, end_row=2, start_col=1, end_col=3, text="Schedule"),
        model_cell(start_row=1, end_row=3, start_col=3, end_col=4, text="Topic"),
        model_cell(start_row=4, end_row=7, start_col=0, end_col=1, text="Tuesday"),
        model_cell(start_row=4, end_row=7, start_col=3, end_col=4, text="XPath"),
    ]

    text_cells = [
        geometry_text_cell("Seminar", row=0, col=2),
        geometry_text_cell("Day", row=1, col=0),
        geometry_text_cell("Schedule", row=1, col=2),
        geometry_text_cell("Topic", row=1, col=3),
        geometry_text_cell("Begin", row=2, col=1),
        geometry_text_cell("End", row=2, col=2),
        geometry_text_cell("Monday", row=3, col=0),
        geometry_text_cell("8:00 a.m.", row=3, col=1),
        geometry_text_cell("5.00 p.m.", row=3, col=2),
        geometry_text_cell("Introduction", row=3, col=3),
        geometry_text_cell("Tuesday", row=4, col=0),
        geometry_text_cell("8:00 a.m.", row=4, col=1),
        geometry_text_cell("11:00 a.m.", row=4, col=2),
        geometry_text_cell("XPath", row=4, col=3),
        geometry_text_cell("11:00 a.m.", row=5, col=1),
        geometry_text_cell("2:00 p.m.", row=5, col=2),
        geometry_text_cell("2:00 p.m.", row=6, col=1),
        geometry_text_cell("5:00 p.m.", row=6, col=2),
        geometry_text_cell("XSL Transformations", row=6, col=3),
        geometry_text_cell("Wednesday", row=7, col=0),
        geometry_text_cell("8:00 a.m.", row=7, col=1),
        geometry_text_cell("12:00 p.m.", row=7, col=2),
        geometry_text_cell("XSL Formatting Object", row=7, col=3),
    ]

    result = reconcile_table_structure(
        cells,
        num_rows=8,
        num_cols=4,
        otsl_seq=["fcel", "fcel", "fcel", "fcel", "nl"] * 8,
        text_cells=text_cells,
        **V1_PRODUCTION_RECONCILIATION_FLAGS,
    )

    by_text = {cell.text: cell for cell in result.table_cells}

    assert result.changed is True
    assert "incumbent_preserved" not in result.notes
    assert result.num_rows == 8
    assert result.num_cols == 4

    assert by_text["Tuesday"].start_row_offset_idx == 4
    assert by_text["Tuesday"].end_row_offset_idx == 7

    assert by_text["XPath"].start_row_offset_idx == 4
    assert by_text["XPath"].end_row_offset_idx == 6

    assert by_text["XSL Transformations"].start_row_offset_idx == 6
    assert by_text["XSL Transformations"].end_row_offset_idx == 7


def test_reconcile_rejects_unsafe_same_shape_text_swap(monkeypatch):
    baseline_cells = [
        model_cell(start_row=0, end_row=1, start_col=0, end_col=1, text="A"),
        model_cell(start_row=0, end_row=1, start_col=1, end_col=2, text="B"),
    ]
    swapped_cells = [
        model_cell(start_row=0, end_row=1, start_col=0, end_col=1, text="B"),
        model_cell(start_row=0, end_row=1, start_col=1, end_col=2, text="A"),
    ]

    _assert_forced_fallback_preserves_incumbent(
        monkeypatch,
        baseline_cells,
        num_rows=1,
        num_cols=2,
        otsl_seq=["fcel", "fcel", "nl"],
        fallback_cells=swapped_cells,
    )


def test_reconcile_accepts_same_shape_topology_repair_when_spans_change():
    cells = [
        model_cell(start_row=4, end_row=7, start_col=3, end_col=4, text="Broad topic"),
        model_cell(
            start_row=6,
            end_row=7,
            start_col=3,
            end_col=4,
            text="Specific topic",
        ),
    ]

    result = reconcile_table_structure(
        cells,
        num_rows=8,
        num_cols=4,
        otsl_seq=[],
        text_cells=[],
        enable_undersegmentation_fallback=False,
        enable_overspan_fallback=False,
        allow_same_row_count=False,
        allow_column_count_growth=False,
        enable_column_reconciliation=False,
        enable_row_boundary_reconciliation=False,
        enable_row_span_reconciliation=False,
    )

    by_text = {cell.text: cell for cell in result.table_cells}

    assert result.changed is True
    assert "topology_repair" in result.notes
    assert "incumbent_preserved" not in result.notes
    assert result.num_rows == 8
    assert result.num_cols == 4
    assert by_text["Broad topic"].start_row_offset_idx == 4
    assert by_text["Broad topic"].end_row_offset_idx == 6
    assert by_text["Specific topic"].start_row_offset_idx == 6
    assert by_text["Specific topic"].end_row_offset_idx == 7


def test_reconcile_accepts_same_shape_row_span_reconciliation_when_spans_change():
    cells = [
        model_cell(start_row=0, end_row=1, start_col=0, end_col=2, text="Header"),
        model_cell(start_row=1, end_row=2, start_col=0, end_col=1, text="A"),
        model_cell(start_row=1, end_row=2, start_col=1, end_col=2, text="d1"),
        model_cell(start_row=2, end_row=3, start_col=0, end_col=1, text=""),
        model_cell(start_row=2, end_row=3, start_col=1, end_col=2, text="d2"),
        model_cell(start_row=3, end_row=4, start_col=0, end_col=1, text="B"),
        model_cell(start_row=3, end_row=4, start_col=1, end_col=2, text="d3"),
        model_cell(start_row=4, end_row=5, start_col=0, end_col=1, text=""),
        model_cell(start_row=4, end_row=5, start_col=1, end_col=2, text="d4"),
        model_cell(start_row=5, end_row=6, start_col=0, end_col=1, text=""),
        model_cell(start_row=5, end_row=6, start_col=1, end_col=2, text="d5"),
        model_cell(start_row=6, end_row=7, start_col=0, end_col=1, text="C"),
        model_cell(start_row=6, end_row=7, start_col=1, end_col=2, text="d6"),
    ]
    cells[0].column_header = True

    result = reconcile_table_structure(
        cells,
        num_rows=7,
        num_cols=2,
        otsl_seq=[],
        text_cells=[],
        enable_undersegmentation_fallback=False,
        enable_overspan_fallback=False,
        allow_same_row_count=False,
        allow_column_count_growth=False,
        enable_column_reconciliation=False,
        enable_row_boundary_reconciliation=False,
        enable_row_span_reconciliation=True,
    )

    by_text = {cell.text: cell for cell in result.table_cells}

    assert result.changed is True
    assert "row_span_reconciliation" in result.notes
    assert "incumbent_preserved" not in result.notes
    assert result.num_rows == 7
    assert result.num_cols == 2
    assert by_text["A"].start_row_offset_idx == 1
    assert by_text["A"].end_row_offset_idx == 2
    assert by_text["B"].start_row_offset_idx == 2
    assert by_text["B"].end_row_offset_idx == 5
    assert by_text["C"].start_row_offset_idx == 5
    assert by_text["C"].end_row_offset_idx == 7
    assert "" not in by_text


def test_reconcile_v2_production_flags_recovers_undersegmented_geometry():
    cells = [
        attach_geometry_bbox(
            model_cell(
                start_row=0,
                end_row=1,
                start_col=1,
                end_col=4,
                text="Seminar",
                column_header=True,
            ),
            left=100,
            top=0,
            right=400,
            bottom=20,
        ),
        attach_geometry_bbox(
            model_cell(
                start_row=1,
                end_row=2,
                start_col=1,
                end_col=3,
                text="Schedule",
                column_header=True,
            ),
            left=100,
            top=20,
            right=300,
            bottom=40,
        ),
        attach_geometry_bbox(
            model_cell(
                start_row=1,
                end_row=3,
                start_col=3,
                end_col=4,
                text="Topic",
                column_header=True,
            ),
            left=300,
            top=20,
            right=400,
            bottom=60,
        ),
        model_cell(start_row=2, end_row=3, start_col=3, end_col=4, text="XPath"),
        model_cell(
            start_row=3,
            end_row=4,
            start_col=3,
            end_col=4,
            text="XSL Transformations",
        ),
    ]

    text_cells = [
        geometry_text_cell("Seminar", row=0, col=2),
        geometry_text_cell("Day", row=1, col=0),
        geometry_text_cell("Schedule", row=1, col=2),
        geometry_text_cell("Topic", row=1, col=3),
        geometry_text_cell("Begin", row=2, col=1),
        geometry_text_cell("End", row=2, col=2),
        geometry_text_cell("Monday", row=3, col=0),
        geometry_text_cell("8:00 a.m.", row=3, col=1),
        geometry_text_cell("5.00 p.m.", row=3, col=2),
        geometry_text_cell("Introduction", row=3, col=3),
        geometry_text_cell("Tuesday", row=4, col=0),
        geometry_text_cell("8:00 a.m.", row=4, col=1),
        geometry_text_cell("11:00 a.m.", row=4, col=2),
        geometry_text_cell("XPath", row=4, col=3),
        geometry_text_cell("11:00 a.m.", row=5, col=1),
        geometry_text_cell("2:00 p.m.", row=5, col=2),
        geometry_text_cell("2:00 p.m.", row=6, col=1),
        geometry_text_cell("5:00 p.m.", row=6, col=2),
        geometry_text_cell("XSL Transformations", row=6, col=3),
        geometry_text_cell("Wednesday", row=7, col=0),
        geometry_text_cell("8:00 a.m.", row=7, col=1),
        geometry_text_cell("12:00 p.m.", row=7, col=2),
        geometry_text_cell("XSL Formatting Object", row=7, col=3),
    ]

    result = reconcile_table_structure(
        cells,
        num_rows=5,
        num_cols=4,
        otsl_seq=["fcel", "fcel", "fcel", "fcel", "nl"] * 5,
        text_cells=text_cells,
        **V2_PRODUCTION_RECONCILIATION_FLAGS,
    )

    by_text = {cell.text: cell for cell in result.table_cells}

    assert result.changed is True
    assert "undersegmentation_geometry_fallback" in result.notes
    assert "incumbent_preserved" not in result.notes
    assert result.num_rows == 8
    assert result.num_cols == 4
    assert by_text["Seminar"].start_col_offset_idx == 1
    assert by_text["Seminar"].end_col_offset_idx == 4
    assert by_text["Tuesday"].start_row_offset_idx == 4
    assert by_text["Tuesday"].end_row_offset_idx == 7
    assert by_text["XPath"].start_row_offset_idx == 4
    assert by_text["XPath"].end_row_offset_idx == 6
    assert by_text["XSL Transformations"].start_row_offset_idx == 6
    assert by_text["XSL Transformations"].end_row_offset_idx == 7
