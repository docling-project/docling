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
    ColumnGridCandidate,
    ColumnGridSelection,
    GridAssignment,
    GutterCandidate,
    ModelCellMetadataPrior,
    ReconciledTableGrid,
    TextInterval,
    _bbox_horizontal_bounds,
    _cell_horizontal_bounds,
    _column_growth_supported_by_gutter_evidence,
    _column_index_for_x,
    _column_range_for_horizontal_bounds,
    _copy_cell_for_column_remap,
    _proportional_column_range,
    _safe_float,
    _table_horizontal_bounds,
    assign_intervals_to_grid,
    build_column_grid_candidate,
    build_table_cells_from_assignment,
    build_validated_table_from_selection,
    collect_model_cell_metadata_prior,
    collect_text_intervals,
    find_repeated_vertical_gutters,
    group_intervals_by_row,
    reconcile_column_grid_from_intervals,
    reconcile_column_grid_from_text_cells,
    reconcile_columns_preserving_rows_from_text_cells,
    remap_model_cells_to_column_grid,
    select_column_grid_candidate,
)
from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _cell_col_range,
    _cell_column_offsets,
    _cell_row_offsets,
    _cell_row_range,
    _copy_cell_with_offsets,
    _first_float_value,
    _infer_split_upper_row_is_column_header,
    _joined_interval_text,
    _row_has_column_header,
    _safe_int,
    _set_cell_header_flags,
)
from docling.models.stages.table_structure.table_structure_row_boundary import (
    RowBoundarySplitCandidate,
    RowTextContamination,
    _cluster_intervals_into_y_bands,
    _collect_cell_text_intervals,
    _interval_horizontal_bounds,
    _interval_inside_cell_bbox,
    _interval_near_cell_vertical_band,
    _interval_text_matches_cell,
    _interval_vertical_bounds,
    _largest_band_boundary,
    _normalize_cell_text_for_matching,
    _RowTextBandCandidate,
    _split_cell_intervals_by_boundary,
    _text_from_intervals,
    apply_row_boundary_split,
    detect_row_text_contamination,
    propose_row_boundary_splits,
)
from docling.models.stages.table_structure.table_structure_row_reassignment import (
    ModelRowBand,
    _bbox_vertical_bounds,
    _cell_vertical_bounds,
    _copy_cell_with_text_and_columns,
    _estimate_fixed_row_bands_from_cells,
    _row_index_for_interval,
    _row_index_for_y,
    infer_model_row_bands_from_cells,
    reassign_text_preserving_rows,
    reassign_text_to_cells_preserving_rows,
)
from docling.models.stages.table_structure.table_structure_row_spans import (
    _build_slot_to_cell_indices,
    _candidate_empty_rows_for_span_column,
    _cell_is_single_slot,
    _cell_text,
    _header_rows,
    _nearest_label_cell_for_empty_row,
    _row_text_counts,
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
        text_cells=list(text_cells or []),
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
            text_cells=list(text_cells or []),
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
        text_cells=list(text_cells or []),
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
            text_cells=list(text_cells or []),
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
            text_cells,
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

    intervals = collect_text_intervals(text_cells or [])

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
        acceptance_report = accept_reconciled_table_challenger(
            baseline_cells=baseline_candidate.table_cells,
            baseline_rows=baseline_candidate.num_rows,
            baseline_cols=baseline_candidate.num_cols,
            baseline_diagnostics=baseline_candidate.diagnostics,
            candidate_cells=candidate.table_cells,
            candidate_rows=candidate.num_rows,
            candidate_cols=candidate.num_cols,
            candidate_diagnostics=candidate.diagnostics,
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
