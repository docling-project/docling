from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from docling.models.stages.table_structure.table_structure_acceptance import (
    accept_reconciled_table_challenger,
)
from docling.models.stages.table_structure.table_structure_candidate_selection import (
    StructureReconciliationCandidate,
    select_reconciled_structure_candidate,
)
from docling.models.stages.table_structure.table_structure_columns import (
    collect_text_intervals,
    reconcile_columns_preserving_rows_from_text_cells,
)
from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _safe_int,
)
from docling.models.stages.table_structure.table_structure_row_boundary import (
    apply_row_boundary_split,
    propose_row_boundary_splits,
)
from docling.models.stages.table_structure.table_structure_row_spans import (
    reconcile_row_spans_from_empty_cells,
)
from docling.models.stages.table_structure.table_topology import (
    TableTopologyDiagnostics,
    cells_to_otsl,
    infer_table_from_text_geometry,
    looks_overspanned_by_text_geometry,
    looks_undersegmented,
    repair_overlapping_cells,
    validate_table_topology,
)


@dataclass(frozen=True)
class TableStructureReconciliationResult:
    table_cells: list[object]
    num_rows: int
    num_cols: int
    otsl_seq: list[str]
    diagnostics: TableTopologyDiagnostics
    changed: bool = False
    notes: tuple[str, ...] = ()


SAME_SHAPE_SPAN_REPAIR_CANDIDATES = frozenset(
    {
        "topology_repair",
        "overspan_geometry_fallback",
        "row_span_reconciliation",
    }
)


def _cell_span_topology(
    cells: list[object],
) -> tuple[tuple[int | None, int | None, int | None, int | None], ...]:
    spans = [
        (
            _safe_int(getattr(cell, "start_row_offset_idx", None)),
            _safe_int(getattr(cell, "end_row_offset_idx", None)),
            _safe_int(getattr(cell, "start_col_offset_idx", None)),
            _safe_int(getattr(cell, "end_col_offset_idx", None)),
        )
        for cell in cells
    ]

    return tuple(
        sorted(
            spans,
            key=lambda span: tuple(-1 if value is None else value for value in span),
        )
    )


def reconcile_table_structure(
    table_cells: list[object],
    *,
    num_rows: int,
    num_cols: int,
    otsl_seq: list[str],
    text_cells: Iterable[object] | None = None,
    table_bbox: object | None = None,
    enable_undersegmentation_fallback: bool = True,
    enable_overspan_fallback: bool = False,
    allow_same_row_count: bool = False,
    allow_column_count_growth: bool = False,
    max_column_count_growth: int | None = None,
    enable_column_reconciliation: bool = True,
    enable_row_boundary_reconciliation: bool = True,
    enable_row_span_reconciliation: bool = True,
) -> TableStructureReconciliationResult:
    """Run the conservative table-structure reconciliation pipeline.

    This is the single orchestration point for topology validation, optional
    geometry fallback, column reconciliation, row-boundary repair, row-span
    repair, final validation, and OTSL regeneration.
    """

    baseline_cells = list(table_cells)
    baseline_rows = num_rows
    baseline_cols = num_cols
    baseline_otsl = list(otsl_seq)
    baseline_diagnostics = validate_table_topology(
        baseline_cells,
        baseline_rows,
        baseline_cols,
    )
    baseline_candidate = StructureReconciliationCandidate(
        name="incumbent",
        table_cells=baseline_cells,
        num_rows=baseline_rows,
        num_cols=baseline_cols,
        otsl_seq=baseline_otsl,
        diagnostics=baseline_diagnostics,
        changed=False,
        notes=(),
    )
    structure_candidates: list[StructureReconciliationCandidate] = []
    source_cells = list(text_cells or [])

    current_cells = list(table_cells)
    current_rows = num_rows
    current_cols = num_cols
    current_otsl = list(otsl_seq)
    changed = False
    notes: list[str] = []

    diagnostics = baseline_diagnostics
    if not diagnostics.valid:
        repaired_cells, repaired_diag = repair_overlapping_cells(
            current_cells,
            current_rows,
            current_cols,
        )
        if repaired_diag.valid:
            current_cells = repaired_cells
            current_otsl = cells_to_otsl(
                current_cells,
                current_rows,
                current_cols,
            )
            diagnostics = repaired_diag
            changed = True
            notes.append("topology_repair")
            structure_candidates.append(
                StructureReconciliationCandidate(
                    name="topology_repair",
                    table_cells=list(current_cells),
                    num_rows=current_rows,
                    num_cols=current_cols,
                    otsl_seq=list(current_otsl),
                    diagnostics=diagnostics,
                    changed=changed,
                    notes=tuple(notes),
                )
            )

    if enable_overspan_fallback and looks_overspanned_by_text_geometry(
        current_cells,
        current_rows,
        num_cols=current_cols,
        text_cells=source_cells,
    ):
        (
            fallback_cells,
            fallback_rows,
            fallback_cols,
            fallback_otsl,
            fallback_diag,
        ) = infer_table_from_text_geometry(
            current_cells,
            current_rows,
            current_cols,
            current_otsl,
            text_cells=source_cells,
            allow_same_row_count=allow_same_row_count,
            allow_column_count_growth=allow_column_count_growth,
        )

        if fallback_diag.valid and fallback_rows >= current_rows:
            current_cells = fallback_cells
            current_rows = fallback_rows
            current_cols = fallback_cols
            current_otsl = fallback_otsl
            diagnostics = fallback_diag
            changed = True
            notes.append("overspan_geometry_fallback")
            structure_candidates.append(
                StructureReconciliationCandidate(
                    name="overspan_geometry_fallback",
                    table_cells=list(current_cells),
                    num_rows=current_rows,
                    num_cols=current_cols,
                    otsl_seq=list(current_otsl),
                    diagnostics=diagnostics,
                    changed=changed,
                    notes=tuple(notes),
                )
            )

    if enable_undersegmentation_fallback and looks_undersegmented(
        current_cells,
        current_rows,
        current_otsl,
        text_cells=source_cells,
        num_cols=current_cols,
    ):
        (
            fallback_cells,
            fallback_rows,
            fallback_cols,
            fallback_otsl,
            fallback_diag,
        ) = infer_table_from_text_geometry(
            current_cells,
            current_rows,
            current_cols,
            current_otsl,
            text_cells=source_cells,
            allow_same_row_count=allow_same_row_count,
            allow_column_count_growth=allow_column_count_growth,
        )

        if fallback_diag.valid and (
            fallback_rows > current_rows or fallback_cols > current_cols
        ):
            current_cells = fallback_cells
            current_rows = fallback_rows
            current_cols = fallback_cols
            current_otsl = fallback_otsl
            diagnostics = fallback_diag
            changed = True
            notes.append("undersegmentation_geometry_fallback")
            structure_candidates.append(
                StructureReconciliationCandidate(
                    name="undersegmentation_geometry_fallback",
                    table_cells=list(current_cells),
                    num_rows=current_rows,
                    num_cols=current_cols,
                    otsl_seq=list(current_otsl),
                    diagnostics=diagnostics,
                    changed=changed,
                    notes=tuple(notes),
                )
            )

    if enable_column_reconciliation:
        column_grid = reconcile_columns_preserving_rows_from_text_cells(
            source_cells,
            table_bbox=table_bbox,
            model_cells=current_cells,
            model_num_rows=current_rows,
            model_num_cols=current_cols,
        )
        if (
            column_grid is not None
            and column_grid.diagnostics.valid
            and column_grid.num_rows == current_rows
            and column_grid.num_cols > current_cols
            and (
                max_column_count_growth is None
                or column_grid.num_cols <= current_cols + max_column_count_growth
            )
        ):
            current_cells = column_grid.table_cells
            current_cols = column_grid.num_cols
            current_otsl = column_grid.otsl_seq
            diagnostics = column_grid.diagnostics
            changed = True
            notes.append("column_reconciliation")
            structure_candidates.append(
                StructureReconciliationCandidate(
                    name="column_reconciliation",
                    table_cells=list(current_cells),
                    num_rows=current_rows,
                    num_cols=current_cols,
                    otsl_seq=list(current_otsl),
                    diagnostics=diagnostics,
                    changed=changed,
                    notes=tuple(notes),
                )
            )

    intervals = collect_text_intervals(source_cells)

    if enable_row_boundary_reconciliation and intervals:
        row_boundary_splits = propose_row_boundary_splits(
            current_cells,
            intervals,
        )
        if len(row_boundary_splits) == 1:
            row_grid = apply_row_boundary_split(
                current_cells,
                intervals,
                num_rows=current_rows,
                num_cols=current_cols,
                split=row_boundary_splits[0],
            )
            if (
                row_grid is not None
                and row_grid.diagnostics.valid
                and row_grid.num_rows == current_rows + 1
                and row_grid.num_cols == current_cols
            ):
                current_cells = row_grid.table_cells
                current_rows = row_grid.num_rows
                current_otsl = row_grid.otsl_seq
                diagnostics = row_grid.diagnostics
                changed = True
                notes.append("row_boundary_reconciliation")
                structure_candidates.append(
                    StructureReconciliationCandidate(
                        name="row_boundary_reconciliation",
                        table_cells=list(current_cells),
                        num_rows=current_rows,
                        num_cols=current_cols,
                        otsl_seq=list(current_otsl),
                        diagnostics=diagnostics,
                        changed=changed,
                        notes=tuple(notes),
                    )
                )
        elif len(row_boundary_splits) > 1:
            notes.append("row_boundary_multiple_candidates_skipped")

    if enable_row_span_reconciliation:
        row_span_grid = reconcile_row_spans_from_empty_cells(
            current_cells,
            num_rows=current_rows,
            num_cols=current_cols,
        )
        if (
            row_span_grid is not None
            and row_span_grid.diagnostics.valid
            and row_span_grid.num_rows == current_rows
            and row_span_grid.num_cols == current_cols
        ):
            current_cells = row_span_grid.table_cells
            current_otsl = row_span_grid.otsl_seq
            diagnostics = row_span_grid.diagnostics
            changed = True
            notes.append("row_span_reconciliation")
            structure_candidates.append(
                StructureReconciliationCandidate(
                    name="row_span_reconciliation",
                    table_cells=list(current_cells),
                    num_rows=current_rows,
                    num_cols=current_cols,
                    otsl_seq=list(current_otsl),
                    diagnostics=diagnostics,
                    changed=changed,
                    notes=tuple(notes),
                )
            )

    final_diagnostics = validate_table_topology(
        current_cells,
        current_rows,
        current_cols,
    )
    if final_diagnostics.valid:
        current_otsl = cells_to_otsl(
            current_cells,
            current_rows,
            current_cols,
        )

    final_candidate = StructureReconciliationCandidate(
        name="final",
        table_cells=current_cells,
        num_rows=current_rows,
        num_cols=current_cols,
        otsl_seq=current_otsl,
        diagnostics=final_diagnostics,
        changed=changed,
        notes=tuple(notes),
    )

    if not structure_candidates or structure_candidates[-1] != final_candidate:
        structure_candidates.append(final_candidate)

    accepted_candidates = []
    for candidate in structure_candidates:
        allow_same_shape_text_slot_change = (
            candidate.name in SAME_SHAPE_SPAN_REPAIR_CANDIDATES
            and candidate.num_rows == baseline_candidate.num_rows
            and candidate.num_cols == baseline_candidate.num_cols
            and _cell_span_topology(candidate.table_cells)
            != _cell_span_topology(baseline_candidate.table_cells)
        )
        acceptance_report = accept_reconciled_table_challenger(
            baseline_cells=baseline_candidate.table_cells,
            baseline_rows=baseline_candidate.num_rows,
            baseline_cols=baseline_candidate.num_cols,
            baseline_diagnostics=baseline_candidate.diagnostics,
            candidate_cells=candidate.table_cells,
            candidate_rows=candidate.num_rows,
            candidate_cols=candidate.num_cols,
            candidate_diagnostics=candidate.diagnostics,
            allow_same_shape_text_slot_change=allow_same_shape_text_slot_change,
            source_cells=source_cells,
        )
        accepted_candidates.append((candidate, acceptance_report))

    selection = select_reconciled_structure_candidate(
        baseline=baseline_candidate,
        candidates=accepted_candidates,
    )
    selected = selection.selected

    return TableStructureReconciliationResult(
        table_cells=selected.table_cells,
        num_rows=selected.num_rows,
        num_cols=selected.num_cols,
        otsl_seq=selected.otsl_seq,
        diagnostics=selected.diagnostics,
        changed=selected.changed,
        notes=(*selected.notes, selection.report.reason),
    )
