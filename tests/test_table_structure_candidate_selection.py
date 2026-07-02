from dataclasses import dataclass

from docling_core.types.doc import TableCell

from docling.models.stages.table_structure.table_structure_acceptance import (
    accept_reconciled_table_challenger,
)
from docling.models.stages.table_structure.table_structure_candidate_selection import (
    StructureReconciliationCandidate,
    select_reconciled_structure_candidate,
)
from docling.models.stages.table_structure.table_structure_reconciler import (
    _column_growth_supported_by_gutter_evidence,
)
from docling.models.stages.table_structure.table_topology import (
    validate_table_topology,
)


@dataclass
class FakeGutter:
    support_count: int


def _cell(text, row_start, row_end, col_start, col_end):
    return TableCell(
        text=text,
        row_span=row_end - row_start,
        col_span=col_end - col_start,
        start_row_offset_idx=row_start,
        end_row_offset_idx=row_end,
        start_col_offset_idx=col_start,
        end_col_offset_idx=col_end,
    )


def _candidate(name, cells, rows, cols, notes=()):
    return StructureReconciliationCandidate(
        name=name,
        table_cells=cells,
        num_rows=rows,
        num_cols=cols,
        otsl_seq=[],
        diagnostics=validate_table_topology(cells, rows, cols),
        changed=name != "incumbent",
        notes=notes,
    )


def test_column_growth_requires_supported_gutter_evidence():
    assert _column_growth_supported_by_gutter_evidence(
        num_cols=5,
        model_num_cols=4,
        kept_gutters=[FakeGutter(3)],
        row_count=10,
    )

    assert not _column_growth_supported_by_gutter_evidence(
        num_cols=7,
        model_num_cols=4,
        kept_gutters=[FakeGutter(3)],
        row_count=10,
    )


def test_structure_selector_picks_best_accepted_candidate():
    baseline_cells = [_cell("Alpha Beta", 0, 1, 0, 1)]
    candidate_cells = [
        _cell("Alpha", 0, 1, 0, 1),
        _cell("Beta", 1, 2, 0, 1),
    ]

    baseline = _candidate("incumbent", baseline_cells, 1, 1)
    challenger = _candidate(
        "row_growth",
        candidate_cells,
        2,
        1,
        notes=("row_boundary_reconciliation",),
    )

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline.table_cells,
        baseline_rows=baseline.num_rows,
        baseline_cols=baseline.num_cols,
        baseline_diagnostics=baseline.diagnostics,
        candidate_cells=challenger.table_cells,
        candidate_rows=challenger.num_rows,
        candidate_cols=challenger.num_cols,
        candidate_diagnostics=challenger.diagnostics,
    )

    selection = select_reconciled_structure_candidate(
        baseline=baseline,
        candidates=[(challenger, report)],
    )

    assert selection.selected.name == "row_growth"
    assert selection.report.accepted


def test_structure_selector_preserves_incumbent_when_no_candidate_is_accepted():
    baseline_cells = [_cell("Alpha Beta", 0, 1, 0, 1)]
    bad_cells = [_cell("Alpha", 0, 1, 0, 1)]

    baseline = _candidate("incumbent", baseline_cells, 1, 1)
    bad = _candidate("text_loss", bad_cells, 1, 1)

    report = accept_reconciled_table_challenger(
        baseline_cells=baseline.table_cells,
        baseline_rows=baseline.num_rows,
        baseline_cols=baseline.num_cols,
        baseline_diagnostics=baseline.diagnostics,
        candidate_cells=bad.table_cells,
        candidate_rows=bad.num_rows,
        candidate_cols=bad.num_cols,
        candidate_diagnostics=bad.diagnostics,
    )

    selection = select_reconciled_structure_candidate(
        baseline=baseline,
        candidates=[(bad, report)],
    )

    assert selection.selected.name == "incumbent"
    assert not selection.report.accepted
