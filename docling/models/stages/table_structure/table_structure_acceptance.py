from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableStructureAcceptanceReport:
    accepted: bool
    reason: str
    baseline_rows: int
    baseline_cols: int
    candidate_rows: int
    candidate_cols: int
    baseline_token_count: int
    preserved_token_count: int


def _normalized_cell_text(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _nonempty_cell_texts(cells: list[object]) -> tuple[str, ...]:
    return tuple(
        text
        for cell in cells
        if (text := _normalized_cell_text(getattr(cell, "text", "")))
    )


def _cell_text_tokens(cells: list[object]) -> tuple[str, ...]:
    tokens: list[str] = []
    for text in _nonempty_cell_texts(cells):
        tokens.extend(text.split())

    return tuple(tokens)


def _token_counts(tokens: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    return counts


def _add_counts(target: dict[str, int], tokens: tuple[str, ...]) -> None:
    for token in tokens:
        target[token] = target.get(token, 0) + 1


def _count_tokens_preserved(
    *,
    baseline_cells: list[object],
    candidate_cells: list[object],
) -> int:
    baseline_counts = _token_counts(_cell_text_tokens(baseline_cells))
    candidate_counts = _token_counts(_cell_text_tokens(candidate_cells))

    return sum(
        min(count, candidate_counts.get(token, 0))
        for token, count in baseline_counts.items()
    )


def _has_text_token_regression(
    *,
    baseline_cells: list[object],
    candidate_cells: list[object],
    source_cells: list[object] | None = None,
) -> bool:
    baseline_counts = _token_counts(_cell_text_tokens(baseline_cells))
    candidate_counts = _token_counts(_cell_text_tokens(candidate_cells))

    for token, count in baseline_counts.items():
        if candidate_counts.get(token, 0) < count:
            return True

    return _has_unsupported_candidate_text(
        baseline_counts=baseline_counts,
        candidate_cells=candidate_cells,
        source_cells=source_cells or [],
    )


BBox = tuple[float, float, float, float]


def _safe_float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bbox_value(obj: object, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None and name == "l":
        value = getattr(obj, "left", None)
    elif value is None and name == "r":
        value = getattr(obj, "right", None)
    elif value is None and name == "t":
        value = getattr(obj, "top", None)
    elif value is None and name == "b":
        value = getattr(obj, "bottom", None)

    return _safe_float_or_none(value)


def _bbox_bounds(cell: object) -> BBox | None:
    bbox = getattr(cell, "bbox", None)
    if bbox is not None:
        left = _bbox_value(bbox, "l")
        top = _bbox_value(bbox, "t")
        right = _bbox_value(bbox, "r")
        bottom = _bbox_value(bbox, "b")
        if None not in (left, top, right, bottom):
            min_x, max_x = sorted((left, right))
            min_y, max_y = sorted((top, bottom))
            if min_x < max_x and min_y < max_y:
                return (min_x, min_y, max_x, max_y)

    rect = getattr(cell, "rect", None)
    if rect is None:
        return None

    xs = [_safe_float_or_none(getattr(rect, f"r_x{idx}", None)) for idx in range(4)]
    ys = [_safe_float_or_none(getattr(rect, f"r_y{idx}", None)) for idx in range(4)]
    if any(value is None for value in xs + ys):
        return None

    min_x = min(value for value in xs if value is not None)
    max_x = max(value for value in xs if value is not None)
    min_y = min(value for value in ys if value is not None)
    max_y = max(value for value in ys if value is not None)
    if min_x >= max_x or min_y >= max_y:
        return None

    return (min_x, min_y, max_x, max_y)


def _bbox_overlaps_or_contains_center(outer: BBox, inner: BBox) -> bool:
    outer_left, outer_top, outer_right, outer_bottom = outer
    inner_left, inner_top, inner_right, inner_bottom = inner

    center_x = (inner_left + inner_right) / 2.0
    center_y = (inner_top + inner_bottom) / 2.0
    if outer_left <= center_x <= outer_right and outer_top <= center_y <= outer_bottom:
        return True

    return max(outer_left, inner_left) < min(outer_right, inner_right) and max(
        outer_top, inner_top
    ) < min(outer_bottom, inner_bottom)


def _local_source_token_counts(
    *,
    candidate_cell: object,
    source_cells: list[object],
) -> dict[str, int]:
    candidate_bbox = _bbox_bounds(candidate_cell)
    if candidate_bbox is None:
        return {}

    counts: dict[str, int] = {}
    for source_cell in source_cells:
        source_bbox = _bbox_bounds(source_cell)
        if source_bbox is None:
            continue

        if _bbox_overlaps_or_contains_center(candidate_bbox, source_bbox):
            _add_counts(counts, _cell_text_tokens([source_cell]))

    return counts


def _has_unsupported_candidate_text(
    *,
    baseline_counts: dict[str, int],
    candidate_cells: list[object],
    source_cells: list[object],
) -> bool:
    baseline_remaining = dict(baseline_counts)

    for candidate_cell in candidate_cells:
        local_source_counts: dict[str, int] | None = None
        for token in _cell_text_tokens([candidate_cell]):
            available = baseline_remaining.get(token, 0)
            if available > 0:
                baseline_remaining[token] = available - 1
                continue

            if local_source_counts is None:
                local_source_counts = _local_source_token_counts(
                    candidate_cell=candidate_cell,
                    source_cells=source_cells,
                )

            local_available = local_source_counts.get(token, 0)
            if local_available <= 0:
                return True

            local_source_counts[token] = local_available - 1

    return False


def _cell_index_range(cell: object, start_attr: str, end_attr: str) -> range:
    start = getattr(cell, start_attr, None)
    end = getattr(cell, end_attr, None)

    if start is None or end is None:
        return range(0)

    return range(int(start), int(end))


def _column_header_rows(cells: list[object]) -> set[int]:
    rows: set[int] = set()

    for cell in cells:
        if not bool(getattr(cell, "column_header", False)):
            continue

        rows.update(
            _cell_index_range(
                cell,
                "start_row_offset_idx",
                "end_row_offset_idx",
            )
        )

    return rows


def _row_header_cols(cells: list[object]) -> set[int]:
    cols: set[int] = set()

    for cell in cells:
        if not bool(getattr(cell, "row_header", False)):
            continue

        cols.update(
            _cell_index_range(
                cell,
                "start_col_offset_idx",
                "end_col_offset_idx",
            )
        )

    return cols


def _has_header_metadata_regression(
    *,
    baseline_cells: list[object],
    candidate_cells: list[object],
) -> bool:
    baseline_column_header_rows = _column_header_rows(baseline_cells)
    candidate_column_header_rows = _column_header_rows(candidate_cells)

    baseline_row_header_cols = _row_header_cols(baseline_cells)
    candidate_row_header_cols = _row_header_cols(candidate_cells)

    return (
        candidate_column_header_rows != baseline_column_header_rows
        or candidate_row_header_cols != baseline_row_header_cols
    )


def _cell_slot_key(
    cell: object,
) -> tuple[int | None, int | None, int | None, int | None]:
    return (
        getattr(cell, "start_row_offset_idx", None),
        getattr(cell, "end_row_offset_idx", None),
        getattr(cell, "start_col_offset_idx", None),
        getattr(cell, "end_col_offset_idx", None),
    )


def _slot_text_map(
    cells: list[object],
) -> dict[tuple[int | None, int | None, int | None, int | None], str]:
    return {
        _cell_slot_key(cell): _normalized_cell_text(getattr(cell, "text", ""))
        for cell in cells
        if _cell_slot_key(cell) != (None, None, None, None)
    }


def _has_same_shape_text_slot_regression(
    *,
    baseline_cells: list[object],
    candidate_cells: list[object],
) -> bool:
    return _slot_text_map(candidate_cells) != _slot_text_map(baseline_cells)


def _diagnostics_valid(diagnostics: object) -> bool:
    return bool(getattr(diagnostics, "valid", True))


def _diagnostic_issue_count(diagnostics: object) -> int:
    total = 0

    for attr in (
        "overlaps",
        "overlapping_cells",
        "out_of_bounds",
        "out_of_bounds_cells",
        "invalid_spans",
        "conflicts",
    ):
        value = getattr(diagnostics, attr, None)
        if value is None:
            continue

        try:
            total += len(value)
        except TypeError:
            if value:
                total += 1

    return total


def _coverage_ratio(diagnostics: object) -> float | None:
    value = getattr(diagnostics, "coverage_ratio", None)
    if value is None:
        return None

    return float(value)


CellRect = tuple[int | None, int | None, int | None, int | None]


def _normalized_cell_text_without_casefold(cell: object) -> str:
    return " ".join(str(getattr(cell, "text", "") or "").split())


def _safe_int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _cell_rect(cell: object) -> CellRect:
    return (
        _safe_int_or_none(getattr(cell, "start_row_offset_idx", None)),
        _safe_int_or_none(getattr(cell, "end_row_offset_idx", None)),
        _safe_int_or_none(getattr(cell, "start_col_offset_idx", None)),
        _safe_int_or_none(getattr(cell, "end_col_offset_idx", None)),
    )


def _rect_contains(outer: CellRect, inner: CellRect) -> bool:
    if None in outer or None in inner:
        return False

    outer_start_row, outer_end_row, outer_start_col, outer_end_col = outer
    inner_start_row, inner_end_row, inner_start_col, inner_end_col = inner

    return (
        outer_start_row <= inner_start_row
        and inner_end_row <= outer_end_row
        and outer_start_col <= inner_start_col
        and inner_end_col <= outer_end_col
    )


def _text_rects_by_value(cells: list[object]) -> dict[str, list[CellRect]]:
    rects_by_text: dict[str, list[CellRect]] = {}

    for cell in cells:
        text = _normalized_cell_text_without_casefold(cell)
        if not text:
            continue

        rects_by_text.setdefault(text, []).append(_cell_rect(cell))

    return {text: sorted(rects) for text, rects in rects_by_text.items()}


def _same_shape_text_slot_changes_are_span_local(
    *,
    baseline_cells: list[object],
    candidate_cells: list[object],
) -> bool:
    baseline_rects_by_text = _text_rects_by_value(baseline_cells)
    candidate_rects_by_text = _text_rects_by_value(candidate_cells)

    for text, baseline_rects in baseline_rects_by_text.items():
        candidate_rects = candidate_rects_by_text.get(text)
        if candidate_rects is None:
            return False

        if baseline_rects == candidate_rects:
            continue

        if len(baseline_rects) != 1 or len(candidate_rects) != 1:
            return False

        baseline_rect = baseline_rects[0]
        candidate_rect = candidate_rects[0]

        if _rect_contains(candidate_rect, baseline_rect):
            continue

        if _rect_contains(baseline_rect, candidate_rect):
            continue

        return False

    return True


def accept_reconciled_table_challenger(
    *,
    baseline_cells: list[object],
    baseline_rows: int,
    baseline_cols: int,
    baseline_diagnostics: object,
    candidate_cells: list[object],
    candidate_rows: int,
    candidate_cols: int,
    candidate_diagnostics: object,
    allow_same_shape_text_slot_change: bool = False,
    source_cells: list[object] | None = None,
) -> TableStructureAcceptanceReport:
    baseline_token_count = len(_cell_text_tokens(baseline_cells))
    preserved_token_count = _count_tokens_preserved(
        baseline_cells=baseline_cells,
        candidate_cells=candidate_cells,
    )

    def reject(reason: str) -> TableStructureAcceptanceReport:
        return TableStructureAcceptanceReport(
            accepted=False,
            reason=reason,
            baseline_rows=baseline_rows,
            baseline_cols=baseline_cols,
            candidate_rows=candidate_rows,
            candidate_cols=candidate_cols,
            baseline_token_count=baseline_token_count,
            preserved_token_count=preserved_token_count,
        )

    def accept(reason: str) -> TableStructureAcceptanceReport:
        return TableStructureAcceptanceReport(
            accepted=True,
            reason=reason,
            baseline_rows=baseline_rows,
            baseline_cols=baseline_cols,
            candidate_rows=candidate_rows,
            candidate_cols=candidate_cols,
            baseline_token_count=baseline_token_count,
            preserved_token_count=preserved_token_count,
        )

    if candidate_rows <= 0 or candidate_cols <= 0:
        return reject("candidate_empty_grid")

    if not _diagnostics_valid(candidate_diagnostics):
        return reject("candidate_invalid_topology")

    if _has_text_token_regression(
        baseline_cells=baseline_cells,
        candidate_cells=candidate_cells,
        source_cells=source_cells,
    ):
        return reject("text_token_regression")

    grid_same_shape = (
        candidate_rows == baseline_rows and candidate_cols == baseline_cols
    )
    grid_grew = candidate_rows > baseline_rows or candidate_cols > baseline_cols

    # Grid-growth repairs may expand structure, but surplus candidate text must
    # be locally source-supported. Same-shape changes also require slot/span
    # ownership checks before the candidate can replace the incumbent.
    if grid_same_shape and _has_same_shape_text_slot_regression(
        baseline_cells=baseline_cells,
        candidate_cells=candidate_cells,
    ):
        if not allow_same_shape_text_slot_change:
            return reject("text_slot_regression")

        if not _same_shape_text_slot_changes_are_span_local(
            baseline_cells=baseline_cells,
            candidate_cells=candidate_cells,
        ):
            return reject("text_slot_regression")

    # Header row/column indices are directly comparable only when grid shape is
    # stable. Grid-growth repairs may shift coordinates while preserving text.
    if grid_same_shape and _has_header_metadata_regression(
        baseline_cells=baseline_cells,
        candidate_cells=candidate_cells,
    ):
        return reject("header_metadata_regression")

    baseline_issue_count = _diagnostic_issue_count(baseline_diagnostics)
    candidate_issue_count = _diagnostic_issue_count(candidate_diagnostics)

    if (
        _diagnostics_valid(baseline_diagnostics)
        and candidate_issue_count > baseline_issue_count
    ):
        return reject("topology_issue_regression")

    baseline_coverage = _coverage_ratio(baseline_diagnostics)
    candidate_coverage = _coverage_ratio(candidate_diagnostics)

    # Coverage can naturally decrease when an undersegmented table grows into a
    # correct logical grid. Keep this protection only for same-shape changes.
    if (
        grid_same_shape
        and baseline_coverage is not None
        and candidate_coverage is not None
        and candidate_coverage + 0.05 < baseline_coverage
    ):
        return reject("coverage_ratio_regression")

    if grid_grew:
        return accept("grid_growth_challenger_accepted")

    return accept("challenger_accepted")
