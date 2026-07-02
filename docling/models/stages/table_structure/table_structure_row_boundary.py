from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median
from typing import Iterable

from docling.models.stages.table_structure.table_structure_columns import (
    ReconciledTableGrid,
    TextInterval,
    _cell_horizontal_bounds,
)
from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _cell_column_offsets,
    _cell_row_offsets,
    _copy_cell_with_offsets,
    _infer_split_upper_row_is_column_header,
    _set_cell_header_flags,
)
from docling.models.stages.table_structure.table_structure_row_reassignment import (
    _cell_vertical_bounds,
)
from docling.models.stages.table_structure.table_topology import (
    TableTopologyDiagnostics,
    cells_to_otsl,
    repair_overlapping_cells,
    validate_table_topology,
)


@dataclass(frozen=True)
class RowTextContamination:
    """A geometry-only signal that one cell may contain text from multiple rows."""

    cell_index: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    text: str
    band_count: int
    candidate_boundary_y: float
    supporting_columns: int
    supporting_cell_indices: tuple[int, ...]


def _interval_vertical_bounds(interval) -> tuple[float, float]:
    top = min(float(interval.top), float(interval.bottom))
    bottom = max(float(interval.top), float(interval.bottom))
    return top, bottom


def _interval_horizontal_bounds(interval) -> tuple[float, float]:
    left = min(float(interval.left), float(interval.right))
    right = max(float(interval.left), float(interval.right))
    return left, right


def _interval_inside_cell_bbox(interval, cell, *, tolerance: float = 2.0) -> bool:
    vertical_bounds = _cell_vertical_bounds(cell)
    horizontal_bounds = _cell_horizontal_bounds(cell)
    if vertical_bounds is None or horizontal_bounds is None:
        return False

    cell_top, cell_bottom = vertical_bounds
    cell_left, cell_right = horizontal_bounds

    interval_left, interval_right = _interval_horizontal_bounds(interval)
    interval_top, interval_bottom = _interval_vertical_bounds(interval)

    center_x = (interval_left + interval_right) / 2.0
    center_y = (interval_top + interval_bottom) / 2.0

    return (
        cell_left - tolerance <= center_x <= cell_right + tolerance
        and cell_top - tolerance <= center_y <= cell_bottom + tolerance
    )


def _normalize_cell_text_for_matching(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _interval_text_matches_cell(interval, cell) -> bool:
    interval_text = _normalize_cell_text_for_matching(
        str(getattr(interval, "text", "") or "")
    )
    cell_text = _normalize_cell_text_for_matching(str(getattr(cell, "text", "") or ""))

    if not interval_text or not cell_text:
        return False

    return interval_text in cell_text


def _interval_near_cell_vertical_band(interval, cell) -> bool:
    vertical_bounds = _cell_vertical_bounds(cell)
    if vertical_bounds is None:
        return False

    cell_top, cell_bottom = vertical_bounds
    interval_top, interval_bottom = _interval_vertical_bounds(interval)
    center_y = (interval_top + interval_bottom) / 2.0

    cell_height = max(1.0, cell_bottom - cell_top)

    # Model cell bboxes can be slightly stale after structure reconciliation.
    # When text membership already links a raw interval to the cell text, allow
    # a wider local vertical window so repeated y-band evidence is not missed.
    tolerance = max(12.0, min(35.0, cell_height * 2.0))

    return cell_top - tolerance <= center_y <= cell_bottom + tolerance


def _collect_cell_text_intervals(cell, intervals) -> list:
    """Collect raw text intervals likely assigned to a model cell.

    Prefer text membership plus a relaxed vertical window. This is more robust
    than strict bbox containment when model bboxes are slightly stale. Fall back
    to strict bbox containment when text membership does not produce evidence.
    """

    text_matched = [
        interval
        for interval in intervals
        if _interval_text_matches_cell(interval, cell)
        and _interval_near_cell_vertical_band(interval, cell)
    ]

    if text_matched:
        return text_matched

    return [
        interval for interval in intervals if _interval_inside_cell_bbox(interval, cell)
    ]


def _cluster_intervals_into_y_bands(intervals) -> list[list]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: float(item.center_y))
    heights = [
        max(1.0, abs(float(interval.bottom) - float(interval.top)))
        for interval in sorted_intervals
    ]
    avg_height = sum(heights) / len(heights)
    split_gap = max(3.0, avg_height * 1.5)

    bands: list[list] = [[sorted_intervals[0]]]
    previous_y = float(sorted_intervals[0].center_y)

    for interval in sorted_intervals[1:]:
        current_y = float(interval.center_y)
        if abs(current_y - previous_y) > split_gap:
            bands.append([interval])
        else:
            bands[-1].append(interval)
        previous_y = current_y

    return bands


def _largest_band_boundary(bands: list[list]) -> float | None:
    if len(bands) < 2:
        return None

    best_gap = -1.0
    best_boundary = None

    for upper_band, lower_band in pairwise(bands):
        upper_bottom = max(
            _interval_vertical_bounds(interval)[1] for interval in upper_band
        )
        lower_top = min(
            _interval_vertical_bounds(interval)[0] for interval in lower_band
        )
        gap = lower_top - upper_bottom
        boundary = (upper_bottom + lower_top) / 2.0

        if gap > best_gap:
            best_gap = gap
            best_boundary = boundary

    return best_boundary


@dataclass(frozen=True)
class _RowTextBandCandidate:
    cell_index: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    text: str
    band_count: int
    candidate_boundary_y: float


def detect_row_text_contamination(
    model_cells,
    intervals,
    *,
    min_supporting_columns: int = 2,
    boundary_tolerance: float = 5.0,
    include_weak: bool = False,
) -> list[RowTextContamination]:
    """Detect vertically mixed text using only geometry.

    This function intentionally does not repair anything. It only reports cells
    whose raw text boxes form multiple vertical bands and whose split is
    supported by neighboring columns in the same model row.
    """

    candidates: list[_RowTextBandCandidate] = []

    for cell_index, cell in enumerate(model_cells):
        start_row, end_row = _cell_row_offsets(cell)
        start_col, end_col = _cell_column_offsets(cell)
        text = str(getattr(cell, "text", "") or "").strip()

        if not text:
            continue

        # Row spans can legitimately contain multiple vertical bands.
        # Mixed text inside a single-row cell is the suspicious case.
        if end_row - start_row != 1:
            continue

        cell_intervals = _collect_cell_text_intervals(cell, intervals)
        if len(cell_intervals) < 2:
            continue

        bands = _cluster_intervals_into_y_bands(cell_intervals)
        if len(bands) < 2:
            continue

        boundary_y = _largest_band_boundary(bands)
        if boundary_y is None:
            continue

        candidates.append(
            _RowTextBandCandidate(
                cell_index=cell_index,
                start_row=start_row,
                end_row=end_row,
                start_col=start_col,
                end_col=end_col,
                text=text,
                band_count=len(bands),
                candidate_boundary_y=boundary_y,
            )
        )

    contaminations: list[RowTextContamination] = []

    for candidate in candidates:
        supporters = [
            other
            for other in candidates
            if other.start_row == candidate.start_row
            and other.end_row == candidate.end_row
            and abs(other.candidate_boundary_y - candidate.candidate_boundary_y)
            <= boundary_tolerance
        ]
        supporting_columns = len(
            {(supporter.start_col, supporter.end_col) for supporter in supporters}
        )

        if supporting_columns < min_supporting_columns and not include_weak:
            continue

        contaminations.append(
            RowTextContamination(
                cell_index=candidate.cell_index,
                start_row=candidate.start_row,
                end_row=candidate.end_row,
                start_col=candidate.start_col,
                end_col=candidate.end_col,
                text=candidate.text,
                band_count=candidate.band_count,
                candidate_boundary_y=candidate.candidate_boundary_y,
                supporting_columns=supporting_columns,
                supporting_cell_indices=tuple(
                    supporter.cell_index for supporter in supporters
                ),
            )
        )

    return contaminations


@dataclass(frozen=True)
class RowBoundarySplitCandidate:
    """A proposed missing row boundary supported by repeated vertical text bands."""

    start_row: int
    end_row: int
    boundary_y: float
    supporting_columns: int
    supporting_cell_indices: tuple[int, ...]


def propose_row_boundary_splits(
    model_cells,
    intervals,
    *,
    min_supporting_columns: int = 2,
    boundary_tolerance: float = 5.0,
) -> list[RowBoundarySplitCandidate]:
    """Propose missing row boundaries from repeated row/text contamination.

    This function intentionally does not mutate the table. It only promotes
    repeated vertical contamination evidence into row-boundary proposals.
    """

    contaminations = detect_row_text_contamination(
        model_cells,
        intervals,
        min_supporting_columns=min_supporting_columns,
        boundary_tolerance=boundary_tolerance,
    )

    proposals: list[RowBoundarySplitCandidate] = []

    for contamination in contaminations:
        matching = [
            item
            for item in contaminations
            if item.start_row == contamination.start_row
            and item.end_row == contamination.end_row
            and abs(item.candidate_boundary_y - contamination.candidate_boundary_y)
            <= boundary_tolerance
        ]

        supporting_columns = len({(item.start_col, item.end_col) for item in matching})
        if supporting_columns < min_supporting_columns:
            continue

        boundary_y = sum(item.candidate_boundary_y for item in matching) / len(matching)
        supporting_cell_indices = tuple(sorted({item.cell_index for item in matching}))

        proposal = RowBoundarySplitCandidate(
            start_row=contamination.start_row,
            end_row=contamination.end_row,
            boundary_y=boundary_y,
            supporting_columns=supporting_columns,
            supporting_cell_indices=supporting_cell_indices,
        )

        if proposal not in proposals:
            proposals.append(proposal)

    return proposals


def _text_from_intervals(intervals) -> str:
    ordered = sorted(
        intervals,
        key=lambda interval: (
            float(getattr(interval, "center_y", 0.0)),
            float(getattr(interval, "center_x", 0.0)),
        ),
    )
    return " ".join(
        str(getattr(interval, "text", "") or "").strip()
        for interval in ordered
        if str(getattr(interval, "text", "") or "").strip()
    ).strip()


def _split_cell_intervals_by_boundary(cell, intervals, boundary_y: float):
    cell_intervals = _collect_cell_text_intervals(cell, intervals)

    upper = [
        interval
        for interval in cell_intervals
        if float(getattr(interval, "center_y", 0.0)) <= boundary_y
    ]
    lower = [
        interval
        for interval in cell_intervals
        if float(getattr(interval, "center_y", 0.0)) > boundary_y
    ]

    return upper, lower


def apply_row_boundary_split(
    model_cells,
    intervals,
    *,
    num_rows: int,
    num_cols: int,
    split: RowBoundarySplitCandidate,
) -> ReconciledTableGrid | None:
    """Apply one missing row-boundary split using repeated y-band evidence.

    This is geometry-driven:
    - it does not use specific text values
    - it does not infer word meaning
    - it only splits the target model row at the proposed y boundary
    """

    split_start = split.start_row
    split_end = split.end_row

    # Keep the first mutation conservative: only split one model row at a time.
    if split_end - split_start != 1:
        return None

    new_cells = []

    for cell in model_cells:
        start_row, end_row = _cell_row_offsets(cell)

        if end_row <= split_start:
            new_cells.append(cell)
            continue

        if start_row >= split_end:
            new_cells.append(
                _copy_cell_with_offsets(
                    cell,
                    start_row=start_row + 1,
                    end_row=end_row + 1,
                )
            )
            continue

        if start_row == split_start and end_row == split_end:
            upper_intervals, lower_intervals = _split_cell_intervals_by_boundary(
                cell,
                intervals,
                split.boundary_y,
            )
            upper_text = _text_from_intervals(upper_intervals)
            lower_text = _text_from_intervals(lower_intervals)

            if upper_text:
                new_cells.append(
                    _copy_cell_with_offsets(
                        cell,
                        start_row=split_start,
                        end_row=split_start + 1,
                        text=upper_text,
                    )
                )

            if lower_text:
                new_cells.append(
                    _copy_cell_with_offsets(
                        cell,
                        start_row=split_start + 1,
                        end_row=split_start + 2,
                        text=lower_text,
                    )
                )

            # If there is text but no interval evidence, keep it in the upper
            # row instead of deleting it. This prevents text loss.
            original_text = str(getattr(cell, "text", "") or "").strip()
            if original_text and not upper_text and not lower_text:
                new_cells.append(
                    _copy_cell_with_offsets(
                        cell,
                        start_row=split_start,
                        end_row=split_start + 1,
                    )
                )

            continue

        # A cell crossing the proposed split needs row-span reconciliation.
        # Do not guess inside this pass.
        return None

    if _infer_split_upper_row_is_column_header(
        model_cells,
        new_cells,
        upper_row=split_start,
        lower_row=split_start + 1,
    ):
        new_cells = [
            _set_cell_header_flags(cell, column_header=True)
            if _cell_row_offsets(cell) == (split_start, split_start + 1)
            and str(getattr(cell, "text", "") or "").strip()
            else cell
            for cell in new_cells
        ]

    new_num_rows = num_rows + 1
    diagnostics = validate_table_topology(new_cells, new_num_rows, num_cols)
    if not diagnostics.valid:
        return None

    return ReconciledTableGrid(
        table_cells=new_cells,
        num_rows=new_num_rows,
        num_cols=num_cols,
        otsl_seq=cells_to_otsl(new_cells, new_num_rows, num_cols),
        diagnostics=diagnostics,
    )
