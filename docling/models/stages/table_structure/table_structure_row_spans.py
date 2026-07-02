from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median
from typing import Iterable

from docling.models.stages.table_structure.table_structure_columns import (
    ReconciledTableGrid,
)
from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _cell_column_offsets,
    _cell_row_offsets,
    _cell_row_range,
    _copy_cell_with_offsets,
    _row_has_column_header,
    _set_cell_header_flags,
)
from docling.models.stages.table_structure.table_topology import (
    TableTopologyDiagnostics,
    cells_to_otsl,
    repair_overlapping_cells,
    validate_table_topology,
)


def _cell_text(cell) -> str:
    return str(getattr(cell, "text", "") or "").strip()


def _cell_is_single_slot(cell) -> bool:
    start_row, end_row = _cell_row_offsets(cell)
    start_col, end_col = _cell_column_offsets(cell)
    return end_row - start_row == 1 and end_col - start_col == 1


def _build_slot_to_cell_indices(model_cells) -> dict[tuple[int, int], list[int]]:
    slots: dict[tuple[int, int], list[int]] = {}

    for index, cell in enumerate(model_cells):
        start_row, end_row = _cell_row_offsets(cell)
        start_col, end_col = _cell_column_offsets(cell)

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                slots.setdefault((row, col), []).append(index)

    return slots


def _header_rows(model_cells) -> set[int]:
    rows: set[int] = set()
    for cell in model_cells:
        if getattr(cell, "column_header", False):
            rows.update(_cell_row_range(cell))
    return rows


def _row_text_counts(model_cells, *, num_rows: int) -> dict[int, int]:
    counts = dict.fromkeys(range(num_rows), 0)

    for cell in model_cells:
        if not _cell_text(cell):
            continue

        for row in _cell_row_range(cell):
            if row in counts:
                counts[row] += 1

    return counts


def _nearest_label_cell_for_empty_row(
    row: int,
    label_rows: list[tuple[int, int]],
    *,
    max_distance: int,
) -> int | None:
    previous_label: tuple[int, int] | None = None
    next_label: tuple[int, int] | None = None

    for label_row, cell_index in label_rows:
        if label_row < row:
            previous_label = (label_row, cell_index)
        elif label_row > row:
            next_label = (label_row, cell_index)
            break

    if previous_label is None and next_label is None:
        return None

    if previous_label is None:
        distance = next_label[0] - row
        return next_label[1] if distance <= max_distance else None

    if next_label is None:
        distance = row - previous_label[0]
        return previous_label[1] if distance <= max_distance else None

    previous_distance = row - previous_label[0]
    next_distance = next_label[0] - row

    if min(previous_distance, next_distance) > max_distance:
        return None

    # Ties go to the lower/next label. This handles labels predicted one row
    # too low without relying on text values.
    if next_distance <= previous_distance:
        return next_label[1]

    return previous_label[1]


def _candidate_empty_rows_for_span_column(
    model_cells,
    *,
    data_rows: set[int],
    col: int,
    slot_to_indices: dict[tuple[int, int], list[int]],
    text_counts: dict[int, int],
) -> set[int]:
    candidate_empty_rows: set[int] = set()

    for row in sorted(data_rows):
        occupants = slot_to_indices.get((row, col), [])
        non_empty_occupants = [
            index for index in occupants if _cell_text(model_cells[index])
        ]
        if non_empty_occupants:
            continue

        empty_occupants = [
            index
            for index in occupants
            if not _cell_text(model_cells[index])
            and _cell_is_single_slot(model_cells[index])
        ]

        if occupants and len(empty_occupants) != len(occupants):
            continue

        if text_counts.get(row, 0) <= 0:
            continue

        candidate_empty_rows.add(row)

    return candidate_empty_rows


def reconcile_row_spans_from_empty_cells(
    model_cells,
    *,
    num_rows: int,
    num_cols: int,
    max_distance: int = 3,
) -> ReconciledTableGrid | None:
    """Recover vertical row spans from nearby empty cells in data rows.

    The pass is intentionally conservative:
    - it ignores column-header rows
    - it only expands single-column, single-row text cells
    - it treats explicit empty cells and absent cells as empty slots
    - it validates topology before accepting the candidate
    """

    header_row_set = _header_rows(model_cells)
    last_header_row = max(header_row_set, default=-1)
    data_rows = set(range(last_header_row + 1, num_rows))

    if not data_rows:
        return None

    slot_to_indices = _build_slot_to_cell_indices(model_cells)
    text_counts = _row_text_counts(model_cells, num_rows=num_rows)

    assigned_rows_by_cell: dict[int, set[int]] = {}
    empty_cell_indices_to_drop: set[int] = set()

    for col in range(num_cols):
        candidate_empty_rows = _candidate_empty_rows_for_span_column(
            model_cells,
            data_rows=data_rows,
            col=col,
            slot_to_indices=slot_to_indices,
            text_counts=text_counts,
        )

        # Avoid false positives in dense columns where only one text slot is
        # missing. Row-span recovery needs repeated empty-slot evidence.
        if len(candidate_empty_rows) < 2:
            continue

        label_rows: list[tuple[int, int]] = []

        for index, cell in enumerate(model_cells):
            start_row, end_row = _cell_row_offsets(cell)
            start_col, end_col = _cell_column_offsets(cell)

            if start_col != col or end_col != col + 1:
                continue
            if end_row - start_row != 1:
                continue
            if start_row not in data_rows:
                continue
            if not _cell_text(cell):
                continue

            label_rows.append((start_row, index))

        label_rows.sort()
        if not label_rows:
            continue

        for row in sorted(data_rows):
            occupants = slot_to_indices.get((row, col), [])

            non_empty_occupants = [
                index for index in occupants if _cell_text(model_cells[index])
            ]
            if non_empty_occupants:
                continue

            empty_occupants = [
                index
                for index in occupants
                if not _cell_text(model_cells[index])
                and _cell_is_single_slot(model_cells[index])
            ]

            if occupants and len(empty_occupants) != len(occupants):
                continue

            # Do not expand through fully blank rows.
            if text_counts.get(row, 0) <= 0:
                continue

            target_index = _nearest_label_cell_for_empty_row(
                row,
                label_rows,
                max_distance=max_distance,
            )
            if target_index is None:
                continue

            assigned_rows_by_cell.setdefault(target_index, set()).add(row)
            empty_cell_indices_to_drop.update(empty_occupants)

    if not assigned_rows_by_cell:
        return None

    updated_row_offsets: dict[int, tuple[int, int]] = {}

    for cell_index, assigned_rows in assigned_rows_by_cell.items():
        original_start, original_end = _cell_row_offsets(model_cells[cell_index])
        rows = set(assigned_rows)
        rows.add(original_start)

        new_start = min(rows)
        new_end = max(rows) + 1

        if new_end - new_start < 2:
            continue

        if len(rows) != new_end - new_start:
            # Only contiguous vertical spans are safe in this pass.
            continue

        start_col, end_col = _cell_column_offsets(model_cells[cell_index])
        has_conflict = False

        for row in range(new_start, new_end):
            for occupant_index in slot_to_indices.get((row, start_col), []):
                if occupant_index == cell_index:
                    continue
                if occupant_index in empty_cell_indices_to_drop:
                    continue
                has_conflict = True
                break
            if has_conflict:
                break

        if has_conflict or end_col - start_col != 1:
            continue

        updated_row_offsets[cell_index] = (new_start, new_end)

    if not updated_row_offsets:
        return None

    new_cells = []

    for index, cell in enumerate(model_cells):
        if index in empty_cell_indices_to_drop:
            continue

        if index in updated_row_offsets:
            new_start, new_end = updated_row_offsets[index]
            new_cells.append(
                _copy_cell_with_offsets(
                    cell,
                    start_row=new_start,
                    end_row=new_end,
                )
            )
        else:
            new_cells.append(cell)

    diagnostics = validate_table_topology(new_cells, num_rows, num_cols)
    if not diagnostics.valid:
        return None

    return ReconciledTableGrid(
        table_cells=new_cells,
        num_rows=num_rows,
        num_cols=num_cols,
        otsl_seq=cells_to_otsl(new_cells, num_rows, num_cols),
        diagnostics=diagnostics,
    )
