from docling_core.types.doc import TableCell

from docling.models.stages.table_structure.table_structure_acceptance import (
    accept_reconciled_table_challenger,
)
from docling.models.stages.table_structure.table_topology import (
    validate_table_topology,
)


def _cell(
    text: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    *,
    column_header: bool = False,
    row_header: bool = False,
) -> TableCell:
    return TableCell(
        text=text,
        row_span=row_end - row_start,
        col_span=col_end - col_start,
        start_row_offset_idx=row_start,
        end_row_offset_idx=row_end,
        start_col_offset_idx=col_start,
        end_col_offset_idx=col_end,
        column_header=column_header,
        row_header=row_header,
    )


def test_acceptance_rejects_real_text_token_loss():
    baseline_cells = [_cell("Alpha Beta", 0, 1, 0, 1)]
    candidate_cells = [_cell("Alpha", 0, 1, 0, 1)]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=1,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 1),
        candidate_cells=candidate_cells,
        candidate_rows=1,
        candidate_cols=1,
        candidate_diagnostics=validate_table_topology(candidate_cells, 1, 1),
    )

    assert not report.accepted
    assert report.reason == "text_token_regression"


def test_acceptance_rejects_header_invention_on_same_shape_grid():
    baseline_cells = [_cell("Body", 0, 1, 0, 1)]
    candidate_cells = [_cell("Body", 0, 1, 0, 1, column_header=True)]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=1,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 1),
        candidate_cells=candidate_cells,
        candidate_rows=1,
        candidate_cols=1,
        candidate_diagnostics=validate_table_topology(candidate_cells, 1, 1),
    )

    assert not report.accepted
    assert report.reason == "header_metadata_regression"


def test_acceptance_allows_grid_growth_when_text_and_topology_are_preserved():
    baseline_cells = [_cell("Header Body", 0, 1, 0, 1, column_header=True)]
    candidate_cells = [
        _cell("Header", 0, 1, 0, 1, column_header=True),
        _cell("Body", 1, 2, 0, 1),
    ]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=1,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 1),
        candidate_cells=candidate_cells,
        candidate_rows=2,
        candidate_cols=1,
        candidate_diagnostics=validate_table_topology(candidate_cells, 2, 1),
    )

    assert report.accepted
    assert report.reason == "grid_growth_challenger_accepted"


def test_acceptance_allows_split_cell_text_preservation():
    baseline_cells = [_cell("Alpha Beta Gamma", 0, 1, 0, 1)]
    candidate_cells = [
        _cell("Alpha", 0, 1, 0, 1),
        _cell("Beta", 1, 2, 0, 1),
        _cell("Gamma", 2, 3, 0, 1),
    ]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=1,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 1),
        candidate_cells=candidate_cells,
        candidate_rows=3,
        candidate_cols=1,
        candidate_diagnostics=validate_table_topology(candidate_cells, 3, 1),
    )

    assert report.accepted
    assert report.preserved_token_count == report.baseline_token_count


def test_acceptance_rejects_header_metadata_deletion_on_same_shape_grid():
    baseline_cells = [_cell("Header", 0, 1, 0, 1, column_header=True)]
    candidate_cells = [_cell("Header", 0, 1, 0, 1, column_header=False)]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=1,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 1),
        candidate_cells=candidate_cells,
        candidate_rows=1,
        candidate_cols=1,
        candidate_diagnostics=validate_table_topology(candidate_cells, 1, 1),
    )

    assert not report.accepted
    assert report.reason == "header_metadata_regression"


def test_acceptance_rejects_same_shape_text_slot_swap():
    baseline_cells = [
        _cell("A", 0, 1, 0, 1),
        _cell("B", 0, 1, 1, 2),
    ]
    candidate_cells = [
        _cell("B", 0, 1, 0, 1),
        _cell("A", 0, 1, 1, 2),
    ]

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline_cells,
        baseline_rows=1,
        baseline_cols=2,
        baseline_diagnostics=validate_table_topology(baseline_cells, 1, 2),
        candidate_cells=candidate_cells,
        candidate_rows=1,
        candidate_cols=2,
        candidate_diagnostics=validate_table_topology(candidate_cells, 1, 2),
    )

    assert not report.accepted
    assert report.reason == "text_slot_regression"
