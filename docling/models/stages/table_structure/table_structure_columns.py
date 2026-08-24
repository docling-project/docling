from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from statistics import median
from typing import Iterable

from docling_core.types.doc import BoundingBox, TableCell

from docling.models.stages.table_structure.table_structure_reconciler_common import (
    _cell_col_range,
    _cell_row_range,
    _first_float_value,
    _safe_int,
)
from docling.models.stages.table_structure.table_topology import (
    TableTopologyDiagnostics,
    cells_to_otsl,
    repair_overlapping_cells,
    validate_table_topology,
)


@dataclass(frozen=True)
class TextInterval:
    left: float
    right: float
    top: float
    bottom: float
    text: str = ""

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True)
class GutterCandidate:
    x: float
    support_rows: frozenset[int]
    median_gap: float

    @property
    def support_count(self) -> int:
        return len(self.support_rows)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_from_rect_cell(cell: object) -> TextInterval | None:
    rect = getattr(cell, "rect", None)
    if rect is None:
        return None

    xs = [
        _safe_float(getattr(rect, "r_x0", None)),
        _safe_float(getattr(rect, "r_x1", None)),
        _safe_float(getattr(rect, "r_x2", None)),
        _safe_float(getattr(rect, "r_x3", None)),
    ]
    ys = [
        _safe_float(getattr(rect, "r_y0", None)),
        _safe_float(getattr(rect, "r_y1", None)),
        _safe_float(getattr(rect, "r_y2", None)),
        _safe_float(getattr(rect, "r_y3", None)),
    ]

    if any(value is None for value in xs + ys):
        return None

    text = getattr(cell, "text", "") or getattr(cell, "orig", "") or ""

    left = min(value for value in xs if value is not None)
    right = max(value for value in xs if value is not None)
    top = min(value for value in ys if value is not None)
    bottom = max(value for value in ys if value is not None)

    if right <= left or bottom <= top:
        return None

    return TextInterval(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        text=str(text),
    )


def _interval_from_bbox_cell(cell: object) -> TextInterval | None:
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
        return None

    left = _safe_float(getattr(bbox, "l", None))
    right = _safe_float(getattr(bbox, "r", None))
    top = _safe_float(getattr(bbox, "t", None))
    bottom = _safe_float(getattr(bbox, "b", None))

    if left is None or right is None or top is None or bottom is None:
        return None

    if right <= left or bottom <= top:
        return None

    text = getattr(cell, "text", "") or getattr(cell, "token", "") or ""

    return TextInterval(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        text=str(text),
    )


def collect_text_intervals(text_cells: Iterable[object]) -> list[TextInterval]:
    intervals: list[TextInterval] = []

    for cell in text_cells:
        interval = _interval_from_rect_cell(cell) or _interval_from_bbox_cell(cell)
        if interval is not None and interval.text.strip():
            intervals.append(interval)

    return intervals


def group_intervals_by_row(
    intervals: list[TextInterval],
    row_tolerance: float | None = None,
) -> list[list[TextInterval]]:
    if not intervals:
        return []

    if row_tolerance is None:
        heights = [interval.height for interval in intervals if interval.height > 0]
        row_tolerance = max(4.0, median(heights) * 0.75) if heights else 4.0

    rows: list[list[TextInterval]] = []
    row_centers: list[float] = []

    for interval in sorted(intervals, key=lambda item: item.center_y):
        matched = False

        for idx, row_center in enumerate(row_centers):
            if abs(interval.center_y - row_center) <= row_tolerance:
                rows[idx].append(interval)
                row_centers[idx] = sum(item.center_y for item in rows[idx]) / len(
                    rows[idx]
                )
                matched = True
                break

        if not matched:
            rows.append([interval])
            row_centers.append(interval.center_y)

    return [sorted(row, key=lambda item: item.left) for row in rows]


def _cluster_gaps(
    gaps: list[tuple[float, float, int]],
    tolerance: float,
) -> list[list[tuple[float, float, int]]]:
    clusters: list[list[tuple[float, float, int]]] = []

    for gap in sorted(gaps, key=lambda item: item[0]):
        midpoint = gap[0]

        if not clusters:
            clusters.append([gap])
            continue

        cluster_midpoint = sum(item[0] for item in clusters[-1]) / len(clusters[-1])
        if abs(midpoint - cluster_midpoint) <= tolerance:
            clusters[-1].append(gap)
        else:
            clusters.append([gap])

    return clusters


def find_repeated_vertical_gutters(
    intervals: list[TextInterval],
    *,
    min_gap: float = 1.0,
    gap_tolerance: float | None = None,
    min_support: int | None = None,
    min_support_ratio: float = 0.18,
) -> list[GutterCandidate]:
    rows = group_intervals_by_row(intervals)
    if not rows:
        return []

    positive_gaps: list[tuple[float, float, int]] = []

    for row_idx, row in enumerate(rows):
        if len(row) < 2:
            continue

        for left_item, right_item in pairwise(row):
            gap_width = right_item.left - left_item.right
            if gap_width <= min_gap:
                continue

            midpoint = (left_item.right + right_item.left) / 2.0
            positive_gaps.append((midpoint, gap_width, row_idx))

    if not positive_gaps:
        return []

    sorted_widths = sorted(gap[1] for gap in positive_gaps)
    lower_half = sorted_widths[: max(1, len(sorted_widths) // 2)]
    typical_gap = median(lower_half)

    # Very large gaps usually indicate skipped empty cells or merged cells.
    # Their midpoint is not a trustworthy real column boundary.
    large_gap_threshold = max(typical_gap * 2.2, typical_gap + 45.0)
    structural_gaps = [gap for gap in positive_gaps if gap[1] <= large_gap_threshold]

    if not structural_gaps:
        return []

    if gap_tolerance is None:
        widths = [interval.width for interval in intervals if interval.width > 0]
        gap_tolerance = max(6.0, median(widths) * 0.45) if widths else 6.0

    if min_support is None:
        min_support = max(2, ceil(len(rows) * min_support_ratio))

    candidates: list[GutterCandidate] = []

    for cluster in _cluster_gaps(structural_gaps, gap_tolerance):
        support_rows = frozenset(gap[2] for gap in cluster)
        if len(support_rows) < min_support:
            continue

        x = sum(gap[0] for gap in cluster) / len(cluster)
        median_gap = median(gap[1] for gap in cluster)

        candidates.append(
            GutterCandidate(
                x=x,
                support_rows=support_rows,
                median_gap=median_gap,
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.x)


@dataclass(frozen=True)
class ColumnGridCandidate:
    boundaries: tuple[float, ...]
    gutters: tuple[GutterCandidate, ...]
    score: float
    notes: tuple[str, ...] = ()

    @property
    def num_cols(self) -> int:
        return max(0, len(self.boundaries) - 1)

    @property
    def centers(self) -> tuple[float, ...]:
        return tuple((left + right) / 2.0 for left, right in pairwise(self.boundaries))


def _table_horizontal_bounds(
    intervals: list[TextInterval],
    table_bbox: object | None = None,
) -> tuple[float, float] | None:
    left = _safe_float(getattr(table_bbox, "l", None))
    right = _safe_float(getattr(table_bbox, "r", None))

    if left is not None and right is not None and right > left:
        return left, right

    if not intervals:
        return None

    return (
        min(interval.left for interval in intervals),
        max(interval.right for interval in intervals),
    )


def build_column_grid_candidate(
    intervals: list[TextInterval],
    *,
    table_bbox: object | None = None,
    model_num_cols: int | None = None,
    gutters: list[GutterCandidate] | None = None,
    min_col_width: float | None = None,
) -> ColumnGridCandidate | None:
    if not intervals:
        return None

    if gutters is None:
        gutters = find_repeated_vertical_gutters(intervals)

    if not gutters:
        return None

    bounds = _table_horizontal_bounds(intervals, table_bbox)
    if bounds is None:
        return None

    table_left, table_right = bounds
    if table_right <= table_left:
        return None

    if min_col_width is None:
        widths = [interval.width for interval in intervals if interval.width > 0]
        min_col_width = max(8.0, median(widths) * 0.6) if widths else 8.0

    kept_gutters: list[GutterCandidate] = []
    boundaries = [table_left]

    for gutter in sorted(gutters, key=lambda candidate: candidate.x):
        if gutter.x <= table_left + min_col_width:
            continue

        if gutter.x >= table_right - min_col_width:
            continue

        if gutter.x - boundaries[-1] < min_col_width:
            continue

        kept_gutters.append(gutter)
        boundaries.append(gutter.x)

    if table_right - boundaries[-1] < min_col_width:
        return None

    boundaries.append(table_right)

    if len(boundaries) < 3:
        return None

    num_cols = len(boundaries) - 1

    rows = group_intervals_by_row(intervals)
    row_count = max(1, len(rows))

    if not _column_growth_supported_by_gutter_evidence(
        num_cols=num_cols,
        model_num_cols=model_num_cols,
        kept_gutters=kept_gutters,
        row_count=row_count,
    ):
        return None
    support_ratio = sum(gutter.support_count for gutter in kept_gutters) / (
        max(1, len(kept_gutters)) * row_count
    )

    model_delta = 0
    if model_num_cols is not None and model_num_cols > 0:
        model_delta = abs(num_cols - model_num_cols)

    score = max(0.0, support_ratio - (0.03 * model_delta))

    return ColumnGridCandidate(
        boundaries=tuple(boundaries),
        gutters=tuple(kept_gutters),
        score=score,
        notes=(
            f"num_cols={num_cols}",
            f"gutters={len(kept_gutters)}",
            f"support_ratio={support_ratio:.3f}",
            f"model_delta={model_delta}",
        ),
    )


@dataclass(frozen=True)
class GridAssignment:
    num_rows: int
    num_cols: int
    slots: dict[tuple[int, int], list[TextInterval]]
    out_of_bounds: list[TextInterval]
    score: float
    notes: tuple[str, ...] = ()

    def texts_at(self, row_idx: int, col_idx: int) -> tuple[str, ...]:
        return tuple(
            interval.text
            for interval in self.slots.get((row_idx, col_idx), [])
            if interval.text
        )


def _column_index_for_x(x: float, boundaries: tuple[float, ...]) -> int | None:
    if len(boundaries) < 2:
        return None

    last_col_idx = len(boundaries) - 2

    for col_idx, (left, right) in enumerate(pairwise(boundaries)):
        if left <= x < right:
            return col_idx

        if col_idx == last_col_idx and left <= x <= right:
            return col_idx

    return None


def assign_intervals_to_grid(
    intervals: list[TextInterval],
    candidate: ColumnGridCandidate,
) -> GridAssignment:
    rows = group_intervals_by_row(intervals)
    slots: dict[tuple[int, int], list[TextInterval]] = {}
    out_of_bounds: list[TextInterval] = []

    for row_idx, row in enumerate(rows):
        for interval in row:
            col_idx = _column_index_for_x(interval.center_x, candidate.boundaries)
            if col_idx is None:
                out_of_bounds.append(interval)
                continue

            slots.setdefault((row_idx, col_idx), []).append(interval)

    num_rows = len(rows)
    num_cols = candidate.num_cols

    total_slots = max(1, num_rows * max(1, num_cols))
    occupied_slots = len(slots)
    multi_text_slots = sum(1 for values in slots.values() if len(values) > 1)

    coverage_ratio = occupied_slots / total_slots
    multi_text_ratio = multi_text_slots / max(1, occupied_slots)
    out_of_bounds_ratio = len(out_of_bounds) / max(1, len(intervals))

    score = (
        candidate.score
        + coverage_ratio
        - (0.10 * multi_text_ratio)
        - (0.50 * out_of_bounds_ratio)
    )

    return GridAssignment(
        num_rows=num_rows,
        num_cols=num_cols,
        slots=slots,
        out_of_bounds=out_of_bounds,
        score=score,
        notes=(
            f"rows={num_rows}",
            f"cols={num_cols}",
            f"occupied_slots={occupied_slots}",
            f"coverage_ratio={coverage_ratio:.3f}",
            f"multi_text_slots={multi_text_slots}",
            f"out_of_bounds={len(out_of_bounds)}",
        ),
    )


@dataclass(frozen=True)
class ColumnGridSelection:
    candidate: ColumnGridCandidate
    assignment: GridAssignment
    reason: str


def select_column_grid_candidate(
    intervals: list[TextInterval],
    *,
    table_bbox: object | None = None,
    model_num_cols: int | None = None,
    min_score: float = 0.45,
    min_score_gain: float = 0.08,
) -> ColumnGridSelection | None:
    candidate = build_column_grid_candidate(
        intervals,
        table_bbox=table_bbox,
        model_num_cols=model_num_cols,
    )
    if candidate is None:
        return None

    assignment = assign_intervals_to_grid(intervals, candidate)

    if assignment.out_of_bounds:
        return None

    if assignment.score < min_score:
        return None

    if model_num_cols is not None and model_num_cols > 0:
        # This reconciler path is for column recovery. If the gutter candidate
        # does not reveal additional stable columns, keep the model output.
        if candidate.num_cols <= model_num_cols:
            return None

        model_delta = candidate.num_cols - model_num_cols
        score_gain = assignment.score - candidate.score

        if model_delta > 3:
            return None

        if score_gain < min_score_gain:
            return None

    return ColumnGridSelection(
        candidate=candidate,
        assignment=assignment,
        reason=(
            "stable gutter candidate accepted "
            f"with cols={candidate.num_cols}, score={assignment.score:.3f}"
        ),
    )


@dataclass(frozen=True)
class ModelCellMetadataPrior:
    column_header_rows: frozenset[int]
    row_header_cols: frozenset[int]
    row_section_rows: frozenset[int]


def collect_model_cell_metadata_prior(
    model_cells: Iterable[object] | None,
) -> ModelCellMetadataPrior:
    column_header_rows: set[int] = set()
    row_header_cols: set[int] = set()
    row_section_rows: set[int] = set()

    for cell in model_cells or []:
        if getattr(cell, "column_header", False):
            column_header_rows.update(_cell_row_range(cell))

        if getattr(cell, "row_header", False):
            row_header_cols.update(_cell_col_range(cell))

        if getattr(cell, "row_section", False):
            row_section_rows.update(_cell_row_range(cell))

    return ModelCellMetadataPrior(
        column_header_rows=frozenset(column_header_rows),
        row_header_cols=frozenset(row_header_cols),
        row_section_rows=frozenset(row_section_rows),
    )


def build_table_cells_from_assignment(
    assignment: GridAssignment,
    model_cells: Iterable[object] | None = None,
) -> list[object]:
    table_cells: list[object] = []
    metadata_prior = collect_model_cell_metadata_prior(model_cells)

    for row_idx in range(assignment.num_rows):
        for col_idx in range(assignment.num_cols):
            slot_intervals = assignment.slots.get((row_idx, col_idx), [])
            texts = assignment.texts_at(row_idx, col_idx)
            if not texts:
                continue

            text = " ".join(text.strip() for text in texts if text.strip())
            if not text:
                continue

            table_cells.append(
                TableCell(
                    text=text,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=row_idx in metadata_prior.column_header_rows,
                    row_header=(
                        col_idx in metadata_prior.row_header_cols
                        and row_idx not in metadata_prior.column_header_rows
                    ),
                    row_section=row_idx in metadata_prior.row_section_rows,
                    bbox=_bbox_from_intervals(slot_intervals),
                )
            )

    return table_cells


def _bbox_from_intervals(intervals: Iterable[TextInterval]) -> BoundingBox | None:
    intervals = list(intervals)
    if not intervals:
        return None

    return BoundingBox(
        l=min(min(interval.left, interval.right) for interval in intervals),
        t=min(min(interval.top, interval.bottom) for interval in intervals),
        r=max(max(interval.left, interval.right) for interval in intervals),
        b=max(max(interval.top, interval.bottom) for interval in intervals),
    )


@dataclass(frozen=True)
class ReconciledTableGrid:
    table_cells: list[object]
    num_rows: int
    num_cols: int
    otsl_seq: list[str]
    diagnostics: object
    selection: ColumnGridSelection | None = None


def build_validated_table_from_selection(
    selection: ColumnGridSelection,
    *,
    model_cells: Iterable[object] | None = None,
) -> ReconciledTableGrid | None:
    from docling.models.stages.table_structure.table_topology import (
        cells_to_otsl,
        validate_table_topology,
    )

    assignment = selection.assignment
    table_cells = build_table_cells_from_assignment(
        assignment,
        model_cells=model_cells,
    )

    diagnostics = validate_table_topology(
        table_cells,
        assignment.num_rows,
        assignment.num_cols,
    )

    if not diagnostics.valid:
        return None

    otsl_seq = cells_to_otsl(
        table_cells,
        assignment.num_rows,
        assignment.num_cols,
    )

    return ReconciledTableGrid(
        table_cells=table_cells,
        num_rows=assignment.num_rows,
        num_cols=assignment.num_cols,
        otsl_seq=otsl_seq,
        diagnostics=diagnostics,
        selection=selection,
    )


def reconcile_column_grid_from_intervals(
    intervals: list[TextInterval],
    *,
    table_bbox: object | None = None,
    model_num_cols: int | None = None,
    model_cells: Iterable[object] | None = None,
) -> ReconciledTableGrid | None:
    selection = select_column_grid_candidate(
        intervals,
        table_bbox=table_bbox,
        model_num_cols=model_num_cols,
    )

    if selection is None:
        return None

    return build_validated_table_from_selection(
        selection,
        model_cells=model_cells,
    )


def reconcile_column_grid_from_text_cells(
    text_cells: Iterable[object] | None,
    *,
    table_bbox: object | None = None,
    model_num_cols: int | None = None,
    model_cells: Iterable[object] | None = None,
) -> ReconciledTableGrid | None:
    intervals = collect_text_intervals(text_cells or [])

    if not intervals:
        return None

    return reconcile_column_grid_from_intervals(
        intervals,
        table_bbox=table_bbox,
        model_num_cols=model_num_cols,
        model_cells=model_cells,
    )


def _copy_cell_for_column_remap(cell: object) -> object:
    if hasattr(cell, "model_copy"):
        return cell.model_copy(deep=True)

    return deepcopy(cell)


def _first_float_value(*values: object) -> float | None:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed

    return None


def _bbox_horizontal_bounds(bbox: object | None) -> tuple[float, float] | None:
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        left = _first_float_value(
            bbox.get("l"),
            bbox.get("left"),
            bbox.get("x0"),
        )
        right = _first_float_value(
            bbox.get("r"),
            bbox.get("right"),
            bbox.get("x1"),
        )
    else:
        left = _first_float_value(
            getattr(bbox, "l", None),
            getattr(bbox, "left", None),
            getattr(bbox, "x0", None),
        )
        right = _first_float_value(
            getattr(bbox, "r", None),
            getattr(bbox, "right", None),
            getattr(bbox, "x1", None),
        )

    if left is None or right is None:
        return None

    if right < left:
        left, right = right, left

    if right <= left:
        return None

    return left, right


def _cell_horizontal_bounds(cell: object) -> tuple[float, float] | None:
    bounds = _bbox_horizontal_bounds(getattr(cell, "bbox", None))
    if bounds is not None:
        return bounds

    return _bbox_horizontal_bounds(getattr(cell, "cell_bbox", None))


def _column_range_for_horizontal_bounds(
    left: float,
    right: float,
    boundaries: tuple[float, ...],
) -> tuple[int, int]:
    meaningful_cols: list[int] = []
    best_col_idx: int | None = None
    best_overlap = 0.0
    cell_width = max(right - left, 1e-6)

    for col_idx in range(len(boundaries) - 1):
        col_left = boundaries[col_idx]
        col_right = boundaries[col_idx + 1]
        col_width = max(col_right - col_left, 1e-6)
        overlap = min(right, col_right) - max(left, col_left)

        if overlap <= 1e-6:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best_col_idx = col_idx

        # Column boundaries from gutters are often whitespace midpoints.
        # Full-cell bboxes may slightly cross those midpoints, so tiny edge
        # overlaps should not be interpreted as real col_spans.
        overlap_ratio = overlap / max(min(cell_width, col_width), 1e-6)
        if overlap_ratio >= 0.25:
            meaningful_cols.append(col_idx)

    if meaningful_cols:
        return min(meaningful_cols), max(meaningful_cols) + 1

    if best_col_idx is not None:
        return best_col_idx, best_col_idx + 1

    center = (left + right) / 2.0
    col_idx = _column_index_for_x(center, boundaries)
    if col_idx is None:
        if center < boundaries[0]:
            col_idx = 0
        else:
            col_idx = len(boundaries) - 2

    return col_idx, col_idx + 1


def _proportional_column_range(
    cell: object,
    *,
    model_num_cols: int,
    candidate_num_cols: int,
) -> tuple[int, int]:
    old_start = _safe_int(getattr(cell, "start_col_offset_idx", None))
    old_end = _safe_int(getattr(cell, "end_col_offset_idx", None))

    if old_start is None:
        old_start = 0
    if old_end is None:
        old_end = old_start + 1

    old_start = max(0, min(old_start, model_num_cols))
    old_end = max(old_start + 1, min(old_end, model_num_cols))

    new_start = int((old_start / model_num_cols) * candidate_num_cols)
    new_end = ceil((old_end / model_num_cols) * candidate_num_cols)

    new_start = max(0, min(new_start, candidate_num_cols - 1))
    new_end = max(new_start + 1, min(new_end, candidate_num_cols))

    return new_start, new_end


def remap_model_cells_to_column_grid(
    model_cells: Iterable[object],
    column_grid: ColumnGridCandidate,
    *,
    model_num_rows: int,
    model_num_cols: int,
) -> ReconciledTableGrid | None:
    from docling.models.stages.table_structure.table_topology import (
        cells_to_otsl,
        validate_table_topology,
    )

    remapped_cells: list[object] = []

    for cell in model_cells:
        copied_cell = _copy_cell_for_column_remap(cell)

        bounds = _cell_horizontal_bounds(cell)
        if bounds is not None:
            start_col, end_col = _column_range_for_horizontal_bounds(
                bounds[0],
                bounds[1],
                column_grid.boundaries,
            )
        else:
            start_col, end_col = _proportional_column_range(
                cell,
                model_num_cols=model_num_cols,
                candidate_num_cols=column_grid.num_cols,
            )

        copied_cell.start_col_offset_idx = start_col
        copied_cell.end_col_offset_idx = end_col
        copied_cell.col_span = end_col - start_col

        # Column-only mode hard rule:
        # do not touch row offsets or row_span.
        remapped_cells.append(copied_cell)

    diagnostics = validate_table_topology(
        remapped_cells,
        num_rows=model_num_rows,
        num_cols=column_grid.num_cols,
    )

    if not diagnostics.valid:
        return None

    otsl_seq = cells_to_otsl(
        remapped_cells,
        num_rows=model_num_rows,
        num_cols=column_grid.num_cols,
    )

    return ReconciledTableGrid(
        table_cells=remapped_cells,
        num_rows=model_num_rows,
        num_cols=column_grid.num_cols,
        otsl_seq=otsl_seq,
        diagnostics=diagnostics,
        selection=None,
    )


def reconcile_columns_preserving_rows_from_text_cells(
    text_cells: Iterable[object] | None,
    *,
    model_cells: Iterable[object],
    model_num_rows: int,
    model_num_cols: int,
    table_bbox: object | None = None,
) -> ReconciledTableGrid | None:
    intervals = collect_text_intervals(text_cells or [])
    if not intervals:
        return None

    selection = select_column_grid_candidate(
        intervals,
        table_bbox=table_bbox,
        model_num_cols=model_num_cols,
    )
    if selection is None:
        return None

    reconciled = remap_model_cells_to_column_grid(
        model_cells,
        selection.candidate,
        model_num_rows=model_num_rows,
        model_num_cols=model_num_cols,
    )

    if reconciled is None:
        return None

    from docling.models.stages.table_structure.table_structure_row_reassignment import (
        reassign_text_to_cells_preserving_rows,
    )

    text_reassigned = reassign_text_to_cells_preserving_rows(
        reconciled.table_cells,
        intervals,
        selection.candidate,
        num_rows=reconciled.num_rows,
        num_cols=reconciled.num_cols,
    )

    if text_reassigned is not None:
        return ReconciledTableGrid(
            table_cells=text_reassigned.table_cells,
            num_rows=text_reassigned.num_rows,
            num_cols=text_reassigned.num_cols,
            otsl_seq=text_reassigned.otsl_seq,
            diagnostics=text_reassigned.diagnostics,
            selection=selection,
        )

    return ReconciledTableGrid(
        table_cells=reconciled.table_cells,
        num_rows=reconciled.num_rows,
        num_cols=reconciled.num_cols,
        otsl_seq=reconciled.otsl_seq,
        diagnostics=reconciled.diagnostics,
        selection=selection,
    )


def _column_growth_supported_by_gutter_evidence(
    *,
    num_cols: int,
    model_num_cols: int | None,
    kept_gutters: list[object],
    row_count: int,
) -> bool:
    """Accept column growth only when the extra columns have gutter support."""
    if model_num_cols is None or model_num_cols <= 0:
        return True

    if num_cols <= model_num_cols:
        return True

    extra_cols = num_cols - model_num_cols
    if extra_cols <= 0:
        return True

    if not kept_gutters:
        return False

    min_support_count = max(2, ceil(row_count * 0.20))
    supported_gutters = sum(
        1
        for gutter in kept_gutters
        if int(getattr(gutter, "support_count", 0)) >= min_support_count
    )

    return supported_gutters >= extra_cols
