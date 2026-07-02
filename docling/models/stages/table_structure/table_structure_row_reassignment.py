from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median
from typing import Iterable

from docling.models.stages.table_structure.table_structure_columns import (
    ColumnGridCandidate,
    ReconciledTableGrid,
    TextInterval,
    _column_index_for_x,
    _copy_cell_for_column_remap,
)
from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _cell_column_offsets,
    _cell_row_offsets,
    _first_float_value,
    _joined_interval_text,
    _safe_int,
)
from docling.models.stages.table_structure.table_topology import (
    TableTopologyDiagnostics,
    cells_to_otsl,
    repair_overlapping_cells,
    validate_table_topology,
)


def _bbox_vertical_bounds(bbox: object | None) -> tuple[float, float] | None:
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        top = _first_float_value(
            bbox.get("t"),
            bbox.get("top"),
            bbox.get("y0"),
        )
        bottom = _first_float_value(
            bbox.get("b"),
            bbox.get("bottom"),
            bbox.get("y1"),
        )
    else:
        top = _first_float_value(
            getattr(bbox, "t", None),
            getattr(bbox, "top", None),
            getattr(bbox, "y0", None),
        )
        bottom = _first_float_value(
            getattr(bbox, "b", None),
            getattr(bbox, "bottom", None),
            getattr(bbox, "y1", None),
        )

    if top is None or bottom is None:
        return None

    if bottom < top:
        top, bottom = bottom, top

    if bottom <= top:
        return None

    return top, bottom


def _cell_vertical_bounds(cell: object) -> tuple[float, float] | None:
    bounds = _bbox_vertical_bounds(getattr(cell, "bbox", None))
    if bounds is not None:
        return bounds

    return _bbox_vertical_bounds(getattr(cell, "cell_bbox", None))


def _estimate_fixed_row_bands_from_cells(
    cells: Iterable[object],
    *,
    num_rows: int,
) -> tuple[float, ...] | None:
    centers_by_row: list[list[float]] = [[] for _ in range(num_rows)]
    all_tops: list[float] = []
    all_bottoms: list[float] = []

    for cell in cells:
        bounds = _cell_vertical_bounds(cell)
        if bounds is None:
            continue

        top, bottom = bounds
        all_tops.append(top)
        all_bottoms.append(bottom)

        start_row = _safe_int(getattr(cell, "start_row_offset_idx", None))
        end_row = _safe_int(getattr(cell, "end_row_offset_idx", None))

        if start_row is None or end_row is None:
            continue

        # Use single-row cells to estimate row centers. Multi-row cells are
        # allowed in the model topology, but they should not define row bands.
        if end_row - start_row != 1:
            continue

        if 0 <= start_row < num_rows:
            centers_by_row[start_row].append((top + bottom) / 2.0)

    row_centers: list[float | None] = [
        median(samples) if samples else None for samples in centers_by_row
    ]

    known = [
        (idx, center) for idx, center in enumerate(row_centers) if center is not None
    ]
    if len(known) < 2:
        return None

    diffs = [
        next_center - current_center
        for (_, current_center), (_, next_center) in pairwise(known)
        if next_center > current_center
    ]
    if not diffs:
        return None

    typical_gap = median(diffs)

    filled_centers: list[float] = []
    for row_idx, center in enumerate(row_centers):
        if center is not None:
            filled_centers.append(center)
            continue

        previous_known = [
            (idx, known_center) for idx, known_center in known if idx < row_idx
        ]
        next_known = [
            (idx, known_center) for idx, known_center in known if idx > row_idx
        ]

        if previous_known and next_known:
            prev_idx, prev_center = previous_known[-1]
            next_idx, next_center = next_known[0]
            ratio = (row_idx - prev_idx) / max(next_idx - prev_idx, 1)
            filled_centers.append(prev_center + ratio * (next_center - prev_center))
        elif previous_known:
            prev_idx, prev_center = previous_known[-1]
            filled_centers.append(prev_center + (row_idx - prev_idx) * typical_gap)
        else:
            next_idx, next_center = next_known[0]
            filled_centers.append(next_center - (next_idx - row_idx) * typical_gap)

    if any(
        next_center <= current_center
        for current_center, next_center in pairwise(filled_centers)
    ):
        return None

    boundaries: list[float] = []
    first_gap = filled_centers[1] - filled_centers[0]
    boundaries.append(
        min(all_tops) if all_tops else filled_centers[0] - first_gap / 2.0
    )

    for current_center, next_center in pairwise(filled_centers):
        boundaries.append((current_center + next_center) / 2.0)

    last_gap = filled_centers[-1] - filled_centers[-2]
    boundaries.append(
        max(all_bottoms) if all_bottoms else filled_centers[-1] + last_gap / 2.0
    )

    return tuple(boundaries)


def _row_index_for_y(
    y: float,
    row_bands: tuple[float, ...],
) -> int | None:
    if len(row_bands) < 2:
        return None

    for row_idx in range(len(row_bands) - 1):
        if row_bands[row_idx] <= y <= row_bands[row_idx + 1]:
            return row_idx

    if y < row_bands[0]:
        return 0

    if y > row_bands[-1]:
        return len(row_bands) - 2

    return None


def _joined_interval_text(intervals: Iterable[TextInterval]) -> str:
    ordered = sorted(intervals, key=lambda item: (item.center_y, item.center_x))
    return " ".join(item.text.strip() for item in ordered if item.text.strip())


def reassign_text_to_cells_preserving_rows(
    cells: Iterable[object],
    intervals: list[TextInterval],
    column_grid: ColumnGridCandidate,
    *,
    num_rows: int,
    num_cols: int,
) -> ReconciledTableGrid | None:
    from docling.models.stages.table_structure.table_topology import (
        cells_to_otsl,
        validate_table_topology,
    )

    original_cells = list(cells)
    row_bands = _estimate_fixed_row_bands_from_cells(
        original_cells,
        num_rows=num_rows,
    )
    if row_bands is None:
        return None

    positioned_intervals: list[tuple[int, int, TextInterval]] = []
    for interval in intervals:
        row_idx = _row_index_for_y(interval.center_y, row_bands)
        col_idx = _column_index_for_x(interval.center_x, column_grid.boundaries)

        if row_idx is None or col_idx is None:
            continue

        positioned_intervals.append((row_idx, col_idx, interval))

    if not positioned_intervals:
        return None

    reassigned_cells: list[object] = []

    for cell in original_cells:
        start_row = _safe_int(getattr(cell, "start_row_offset_idx", None))
        end_row = _safe_int(getattr(cell, "end_row_offset_idx", None))
        start_col = _safe_int(getattr(cell, "start_col_offset_idx", None))
        end_col = _safe_int(getattr(cell, "end_col_offset_idx", None))

        if (
            start_row is None
            or end_row is None
            or start_col is None
            or end_col is None
            or end_row <= start_row
            or end_col <= start_col
        ):
            reassigned_cells.append(cell)
            continue

        matched = [
            (row_idx, col_idx, interval)
            for row_idx, col_idx, interval in positioned_intervals
            if start_row <= row_idx < end_row and start_col <= col_idx < end_col
        ]

        if not matched:
            reassigned_cells.append(cell)
            continue

        cols_with_text = sorted({col_idx for _, col_idx, _ in matched})

        if end_col - start_col > 1 and len(cols_with_text) > 1:
            for col_idx in cols_with_text:
                col_intervals = [
                    interval
                    for _, matched_col_idx, interval in matched
                    if matched_col_idx == col_idx
                ]
                text = _joined_interval_text(col_intervals)
                if not text:
                    continue

                split_cell = _copy_cell_for_column_remap(cell)
                split_cell.text = text
                split_cell.start_col_offset_idx = col_idx
                split_cell.end_col_offset_idx = col_idx + 1
                split_cell.col_span = 1

                # Column-only text reassignment hard rule:
                # rows and row spans are preserved exactly.
                reassigned_cells.append(split_cell)
        else:
            copied_cell = _copy_cell_for_column_remap(cell)
            text = _joined_interval_text(interval for _, _, interval in matched)
            if text:
                copied_cell.text = text
            reassigned_cells.append(copied_cell)

    diagnostics = validate_table_topology(
        reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    if not diagnostics.valid:
        return None

    otsl_seq = cells_to_otsl(
        reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )

    return ReconciledTableGrid(
        table_cells=reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
        otsl_seq=otsl_seq,
        diagnostics=diagnostics,
        selection=None,
    )


@dataclass(frozen=True)
class ModelRowBand:
    row_idx: int
    top: float
    bottom: float

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def _bbox_vertical_bounds(bbox: object | None) -> tuple[float, float] | None:
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        top = _first_float_value(
            bbox.get("t"),
            bbox.get("top"),
            bbox.get("y0"),
        )
        bottom = _first_float_value(
            bbox.get("b"),
            bbox.get("bottom"),
            bbox.get("y1"),
        )
    else:
        top = _first_float_value(
            getattr(bbox, "t", None),
            getattr(bbox, "top", None),
            getattr(bbox, "y0", None),
        )
        bottom = _first_float_value(
            getattr(bbox, "b", None),
            getattr(bbox, "bottom", None),
            getattr(bbox, "y1", None),
        )

    if top is None or bottom is None:
        return None

    if bottom < top:
        top, bottom = bottom, top

    if bottom <= top:
        return None

    return top, bottom


def _cell_vertical_bounds(cell: object) -> tuple[float, float] | None:
    bounds = _bbox_vertical_bounds(getattr(cell, "bbox", None))
    if bounds is not None:
        return bounds

    return _bbox_vertical_bounds(getattr(cell, "cell_bbox", None))


def infer_model_row_bands_from_cells(
    model_cells: Iterable[object],
    *,
    num_rows: int,
) -> tuple[ModelRowBand, ...]:
    row_tops: dict[int, list[float]] = {}
    row_bottoms: dict[int, list[float]] = {}
    row_heights: list[float] = []

    for cell in model_cells:
        start_row = _safe_int(getattr(cell, "start_row_offset_idx", None))
        end_row = _safe_int(getattr(cell, "end_row_offset_idx", None))
        vertical_bounds = _cell_vertical_bounds(cell)

        if (
            start_row is None
            or end_row is None
            or end_row != start_row + 1
            or vertical_bounds is None
            or not (0 <= start_row < num_rows)
        ):
            continue

        top, bottom = vertical_bounds
        row_tops.setdefault(start_row, []).append(top)
        row_bottoms.setdefault(start_row, []).append(bottom)
        row_heights.append(bottom - top)

    if not row_tops:
        return ()

    typical_height = median(row_heights) if row_heights else 1.0
    known_centers: dict[int, float] = {}

    for row_idx in row_tops:
        top = median(row_tops[row_idx])
        bottom = median(row_bottoms[row_idx])
        known_centers[row_idx] = (top + bottom) / 2.0

    bands: list[ModelRowBand] = []

    for row_idx in range(num_rows):
        if row_idx in known_centers:
            center_y = known_centers[row_idx]
        else:
            previous_known = [
                known_row for known_row in known_centers if known_row < row_idx
            ]
            next_known = [
                known_row for known_row in known_centers if known_row > row_idx
            ]

            if previous_known and next_known:
                previous_row = max(previous_known)
                next_row = min(next_known)
                previous_center = known_centers[previous_row]
                next_center = known_centers[next_row]
                ratio = (row_idx - previous_row) / (next_row - previous_row)
                center_y = previous_center + ratio * (next_center - previous_center)
            elif previous_known:
                previous_row = max(previous_known)
                center_y = (
                    known_centers[previous_row]
                    + (row_idx - previous_row) * typical_height
                )
            elif next_known:
                next_row = min(next_known)
                center_y = (
                    known_centers[next_row] - (next_row - row_idx) * typical_height
                )
            else:
                return ()

        bands.append(
            ModelRowBand(
                row_idx=row_idx,
                top=center_y - typical_height / 2.0,
                bottom=center_y + typical_height / 2.0,
            )
        )

    return tuple(bands)


def _row_index_for_interval(
    interval: TextInterval,
    row_bands: tuple[ModelRowBand, ...],
) -> int | None:
    if not row_bands:
        return None

    return min(
        row_bands,
        key=lambda row_band: abs(interval.center_y - row_band.center_y),
    ).row_idx


def _copy_cell_with_text_and_columns(
    cell: object,
    *,
    text: str,
    start_col: int,
    end_col: int,
) -> object:
    copied_cell = _copy_cell_for_column_remap(cell)
    copied_cell.text = text
    copied_cell.start_col_offset_idx = start_col
    copied_cell.end_col_offset_idx = end_col
    copied_cell.col_span = end_col - start_col
    return copied_cell


def reassign_text_preserving_rows(
    model_cells: Iterable[object],
    intervals: list[TextInterval],
    column_boundaries: tuple[float, ...],
    *,
    num_rows: int,
    num_cols: int,
) -> ReconciledTableGrid | None:
    from docling.models.stages.table_structure.table_topology import (
        cells_to_otsl,
        validate_table_topology,
    )

    model_cells = list(model_cells)
    row_bands = infer_model_row_bands_from_cells(model_cells, num_rows=num_rows)
    if not row_bands:
        return None

    reassigned_cells: list[object] = []

    for cell in model_cells:
        row_offsets = _cell_row_offsets(cell)
        col_offsets = _cell_column_offsets(cell)

        if row_offsets is None or col_offsets is None:
            reassigned_cells.append(cell)
            continue

        start_row, end_row = row_offsets
        start_col, end_col = col_offsets
        original_text = str(getattr(cell, "text", "") or "").strip()

        # Conservative rule:
        # This pass may only split existing non-empty horizontally spanning cells.
        # It must not fill empty cells and must not rewrite single-column cells.
        if not original_text or end_col - start_col <= 1:
            reassigned_cells.append(cell)
            continue

        vertical_bounds = _cell_vertical_bounds(cell)
        if vertical_bounds is None:
            reassigned_cells.append(cell)
            continue

        top, bottom = vertical_bounds
        height = bottom - top
        tolerance = max(1.0, min(3.0, height * 0.15))

        grouped_text: dict[int, list[str]] = {}

        for interval in intervals:
            # Do not pull text from neighboring visual rows. The frozen model row
            # prior can be imperfect, so the cell's own vertical bbox is a stricter
            # local gate than nearest-row assignment alone.
            if interval.center_y < top - tolerance:
                continue
            if interval.center_y > bottom + tolerance:
                continue

            row_idx = _row_index_for_interval(interval, row_bands)
            col_idx = _column_index_for_x(interval.center_x, column_boundaries)

            if row_idx is None or col_idx is None:
                continue

            if not (start_row <= row_idx < end_row):
                continue

            if not (start_col <= col_idx < end_col):
                continue

            text = interval.text.strip()
            if not text:
                continue

            grouped_text.setdefault(col_idx, []).append(text)

        if len(grouped_text) <= 1:
            reassigned_cells.append(cell)
            continue

        for col_idx in sorted(grouped_text):
            text = " ".join(grouped_text[col_idx]).strip()
            if not text:
                continue

            reassigned_cells.append(
                _copy_cell_with_text_and_columns(
                    cell,
                    text=text,
                    start_col=col_idx,
                    end_col=col_idx + 1,
                )
            )

    diagnostics = validate_table_topology(
        reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )

    if not diagnostics.valid:
        return None

    otsl_seq = cells_to_otsl(
        reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )

    return ReconciledTableGrid(
        table_cells=reassigned_cells,
        num_rows=num_rows,
        num_cols=num_cols,
        otsl_seq=otsl_seq,
        diagnostics=diagnostics,
        selection=None,
    )
