from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from docling_core.types.doc import TableCell


@dataclass
class TableTopologyDiagnostics:
    valid: bool
    out_of_bounds_cells: list[int] = field(default_factory=list)
    zero_or_negative_spans: list[int] = field(default_factory=list)
    span_mismatch_cells: list[int] = field(default_factory=list)
    overlapping_slots: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    duplicate_cells: list[int] = field(default_factory=list)
    coverage_ratio: float = 0.0
    unresolved_conflict_count: int = 0
    notes: list[str] = field(default_factory=list)


def _cell_text(cell: TableCell) -> str:
    text = getattr(cell, "text", "") or getattr(cell, "token", "") or ""
    return " ".join(str(text).split())


def _cell_area(cell: TableCell) -> int:
    return max(0, cell.end_row_offset_idx - cell.start_row_offset_idx) * max(
        0, cell.end_col_offset_idx - cell.start_col_offset_idx
    )


def _cell_slots(cell: TableCell) -> Iterable[tuple[int, int]]:
    for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
        for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
            yield (r, c)


def _copy_cell(cell: TableCell) -> TableCell:
    return cell.model_copy(deep=True)


def _normalize_spans(cells: list[TableCell]) -> list[TableCell]:
    normalized: list[TableCell] = []

    for cell in cells:
        new_cell = _copy_cell(cell)

        row_span = new_cell.end_row_offset_idx - new_cell.start_row_offset_idx
        col_span = new_cell.end_col_offset_idx - new_cell.start_col_offset_idx

        new_cell.row_span = row_span
        new_cell.col_span = col_span

        normalized.append(new_cell)

    return normalized


def validate_table_topology(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
) -> TableTopologyDiagnostics:
    slot_owner: dict[tuple[int, int], list[int]] = {}
    out_of_bounds_cells: list[int] = []
    zero_or_negative_spans: list[int] = []
    span_mismatch_cells: list[int] = []
    duplicate_cells: list[int] = []

    seen_rect_text: dict[tuple[int, int, int, int, str], int] = {}

    for idx, cell in enumerate(cells):
        start_row = cell.start_row_offset_idx
        end_row = cell.end_row_offset_idx
        start_col = cell.start_col_offset_idx
        end_col = cell.end_col_offset_idx

        if end_row <= start_row or end_col <= start_col:
            zero_or_negative_spans.append(idx)
            continue

        if start_row < 0 or start_col < 0 or end_row > num_rows or end_col > num_cols:
            out_of_bounds_cells.append(idx)
            continue

        if cell.row_span != end_row - start_row or cell.col_span != end_col - start_col:
            span_mismatch_cells.append(idx)

        rect_text_key = (
            start_row,
            end_row,
            start_col,
            end_col,
            _cell_text(cell),
        )
        if rect_text_key in seen_rect_text:
            duplicate_cells.append(idx)
        else:
            seen_rect_text[rect_text_key] = idx

        for slot in _cell_slots(cell):
            slot_owner.setdefault(slot, []).append(idx)

    overlapping_slots = {
        slot: owners for slot, owners in slot_owner.items() if len(owners) > 1
    }

    total_slots = max(1, num_rows * num_cols)
    occupied_slots = len(slot_owner)
    coverage_ratio = occupied_slots / total_slots

    valid = not (
        out_of_bounds_cells
        or zero_or_negative_spans
        or span_mismatch_cells
        or overlapping_slots
    )

    return TableTopologyDiagnostics(
        valid=valid,
        out_of_bounds_cells=out_of_bounds_cells,
        zero_or_negative_spans=zero_or_negative_spans,
        span_mismatch_cells=span_mismatch_cells,
        overlapping_slots=overlapping_slots,
        duplicate_cells=duplicate_cells,
        coverage_ratio=coverage_ratio,
        unresolved_conflict_count=len(overlapping_slots),
    )


def _contains_cell(outer: TableCell, inner: TableCell) -> bool:
    return (
        outer.start_row_offset_idx <= inner.start_row_offset_idx
        and outer.end_row_offset_idx >= inner.end_row_offset_idx
        and outer.start_col_offset_idx <= inner.start_col_offset_idx
        and outer.end_col_offset_idx >= inner.end_col_offset_idx
    )


def _same_col_band(a: TableCell, b: TableCell) -> bool:
    return (
        a.start_col_offset_idx <= b.start_col_offset_idx
        and a.end_col_offset_idx >= b.end_col_offset_idx
    )


def _same_row_band(a: TableCell, b: TableCell) -> bool:
    return (
        a.start_row_offset_idx <= b.start_row_offset_idx
        and a.end_row_offset_idx >= b.end_row_offset_idx
    )


def _try_shrink_overrun(
    broad: TableCell,
    specific: TableCell,
) -> bool:
    if not _contains_cell(broad, specific):
        return False

    broad_area = _cell_area(broad)
    specific_area = _cell_area(specific)

    if broad_area <= specific_area:
        return False

    specific_text = _cell_text(specific)
    broad_text = _cell_text(broad)

    # Prefer preserving a smaller specific non-empty cell.
    if not specific_text:
        return False

    # Row-span trailing overrun:
    # broad starts before specific and ends at/after specific.
    if _same_col_band(broad, specific):
        if (
            broad.start_row_offset_idx < specific.start_row_offset_idx
            and broad.end_row_offset_idx >= specific.end_row_offset_idx
        ):
            broad.end_row_offset_idx = specific.start_row_offset_idx
            broad.row_span = broad.end_row_offset_idx - broad.start_row_offset_idx
            return broad.row_span > 0

        # Row-span leading overrun:
        if (
            broad.start_row_offset_idx <= specific.start_row_offset_idx
            and broad.end_row_offset_idx > specific.end_row_offset_idx
        ):
            broad.start_row_offset_idx = specific.end_row_offset_idx
            broad.row_span = broad.end_row_offset_idx - broad.start_row_offset_idx
            return broad.row_span > 0

    # Column-span trailing overrun.
    if _same_row_band(broad, specific):
        if (
            broad.start_col_offset_idx < specific.start_col_offset_idx
            and broad.end_col_offset_idx >= specific.end_col_offset_idx
        ):
            broad.end_col_offset_idx = specific.start_col_offset_idx
            broad.col_span = broad.end_col_offset_idx - broad.start_col_offset_idx
            return broad.col_span > 0

        # Column-span leading overrun.
        if (
            broad.start_col_offset_idx <= specific.start_col_offset_idx
            and broad.end_col_offset_idx > specific.end_col_offset_idx
        ):
            broad.start_col_offset_idx = specific.end_col_offset_idx
            broad.col_span = broad.end_col_offset_idx - broad.start_col_offset_idx
            return broad.col_span > 0

    # If both cells have same text and broad is larger, do not shrink based only on that.
    if broad_text and broad_text == specific_text:
        return False

    return False


def repair_overlapping_cells(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
    max_iterations: int = 8,
) -> tuple[list[TableCell], TableTopologyDiagnostics]:
    repaired = _normalize_spans(cells)

    for _ in range(max_iterations):
        diag = validate_table_topology(repaired, num_rows, num_cols)

        if not diag.overlapping_slots:
            return repaired, diag

        changed = False
        cells_to_drop: set[int] = set()

        # Drop exact duplicates first.
        for dup_idx in diag.duplicate_cells:
            cells_to_drop.add(dup_idx)

        if cells_to_drop:
            repaired = [
                cell for idx, cell in enumerate(repaired) if idx not in cells_to_drop
            ]
            changed = True
            continue

        for _, owner_ids in diag.overlapping_slots.items():
            owners = [repaired[idx] for idx in owner_ids]

            # Sort broad cells first, then try to shrink the broad one around
            # a smaller, specific, non-empty cell.
            owners_with_ids = sorted(
                zip(owner_ids, owners),
                key=lambda item: _cell_area(item[1]),
                reverse=True,
            )

            for broad_id, broad_cell in owners_with_ids:
                for specific_id, specific_cell in reversed(owners_with_ids):
                    if broad_id == specific_id:
                        continue

                    if _try_shrink_overrun(broad_cell, specific_cell):
                        repaired[broad_id] = broad_cell
                        changed = True
                        break

                if changed:
                    break

            if changed:
                break

        if not changed:
            break

    final_diag = validate_table_topology(repaired, num_rows, num_cols)
    final_diag.unresolved_conflict_count = len(final_diag.overlapping_slots)
    if final_diag.overlapping_slots:
        final_diag.notes.append("Unresolved overlapping table-cell topology.")
    return repaired, final_diag


def cells_to_otsl(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
) -> list[str]:
    start_slots: dict[tuple[int, int], TableCell] = {}
    covered_slots: dict[tuple[int, int], TableCell] = {}

    for cell in cells:
        start = (cell.start_row_offset_idx, cell.start_col_offset_idx)
        start_slots[start] = cell

        for slot in _cell_slots(cell):
            if slot != start:
                covered_slots[slot] = cell

    seq: list[str] = []

    for r in range(num_rows):
        for c in range(num_cols):
            slot = (r, c)

            if slot in start_slots:
                cell = start_slots[slot]
                if getattr(cell, "column_header", False):
                    seq.append("ched")
                elif getattr(cell, "row_header", False):
                    seq.append("rhed")
                elif getattr(cell, "row_section", False):
                    seq.append("srow")
                elif _cell_text(cell):
                    seq.append("fcel")
                else:
                    seq.append("ecel")
                continue

            if slot in covered_slots:
                owner = covered_slots[slot]
                from_left = (
                    r == owner.start_row_offset_idx and c > owner.start_col_offset_idx
                )
                from_above = (
                    r > owner.start_row_offset_idx and c == owner.start_col_offset_idx
                )

                if from_left:
                    seq.append("lcel")
                elif from_above:
                    seq.append("ucel")
                else:
                    seq.append("xcel")
                continue

            seq.append("ecel")

        seq.append("nl")

    return seq


def _get_bbox(obj: object) -> object | None:
    bbox = getattr(obj, "bbox", None)
    if bbox is not None:
        return bbox

    rect = getattr(obj, "rect", None)
    if rect is not None and hasattr(rect, "to_bounding_box"):
        return rect.to_bounding_box()

    return None


def _bbox_height(bbox: object) -> float:
    return abs(float(getattr(bbox, "b")) - float(getattr(bbox, "t")))


def _bbox_center_y(bbox: object) -> float:
    return (float(getattr(bbox, "t")) + float(getattr(bbox, "b"))) / 2.0


def _estimate_text_row_bands(text_cells: list[object] | None) -> int:
    from statistics import median

    samples: list[tuple[float, float]] = []

    for text_cell in text_cells or []:
        text = getattr(text_cell, "text", "")
        if not str(text).strip():
            continue

        bbox = _get_bbox(text_cell)
        if bbox is None:
            continue

        height = _bbox_height(bbox)
        if height <= 0:
            continue

        samples.append((_bbox_center_y(bbox), height))

    if not samples:
        return 0

    heights = [height for _, height in samples]
    tolerance = max(median(heights) * 0.75, 1.0)

    bands: list[float] = []
    for center_y, _ in sorted(samples):
        if not bands or abs(center_y - bands[-1]) > tolerance:
            bands.append(center_y)
        else:
            bands[-1] = (bands[-1] + center_y) / 2.0

    return len(bands)


def _bbox_left(bbox: object) -> float | None:
    value = getattr(bbox, "l", None)
    if value is None:
        value = getattr(bbox, "left", None)
    return None if value is None else float(value)


def _bbox_right(bbox: object) -> float | None:
    value = getattr(bbox, "r", None)
    if value is None:
        value = getattr(bbox, "right", None)
    return None if value is None else float(value)


def _bbox_center_x(bbox: object) -> float | None:
    left = _bbox_left(bbox)
    right = _bbox_right(bbox)
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def _bbox_width(bbox: object) -> float | None:
    left = _bbox_left(bbox)
    right = _bbox_right(bbox)
    if left is None or right is None:
        return None
    return abs(right - left)


def _normalized_text(text: object) -> str:
    return " ".join(str(text or "").split())


def _make_table_cell(
    *,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    text: str,
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
) -> TableCell:
    return TableCell.model_validate(
        {
            "row_span": end_row - start_row,
            "col_span": end_col - start_col,
            "start_row_offset_idx": start_row,
            "end_row_offset_idx": end_row,
            "start_col_offset_idx": start_col,
            "end_col_offset_idx": end_col,
            "text": text,
            "column_header": column_header,
            "row_header": row_header,
            "row_section": row_section,
            "bbox": None,
        }
    )


def _cluster_axis_by_tolerance(
    values: list[tuple[float, float]],
    tolerance_scale: float = 0.75,
) -> list[float]:
    from statistics import median

    if not values:
        return []

    sizes = [size for _, size in values if size > 0]
    tolerance = max(median(sizes) * tolerance_scale, 1.0) if sizes else 1.0

    centers: list[float] = []
    for value, _ in sorted(values):
        if not centers or abs(value - centers[-1]) > tolerance:
            centers.append(value)
        else:
            centers[-1] = (centers[-1] + value) / 2.0

    return centers


def _cluster_axis_expected_count(
    values: list[tuple[float, float]],
    expected_count: int | None,
) -> list[float]:
    if not values:
        return []

    centers = [value for value, _ in sorted(values)]

    if expected_count is None or expected_count <= 0 or len(centers) < expected_count:
        return _cluster_axis_by_tolerance(values)

    if expected_count == 1:
        return [sum(centers) / len(centers)]

    min_value = min(centers)
    max_value = max(centers)

    if min_value == max_value:
        return [min_value]

    # Deterministic one-dimensional k-means. This gives stable column bands
    # without hard-coding any table content.
    current = [
        min_value + (max_value - min_value) * i / (expected_count - 1)
        for i in range(expected_count)
    ]

    for _ in range(12):
        groups: list[list[float]] = [[] for _ in current]
        for value in centers:
            group_idx = min(
                range(len(current)),
                key=lambda idx: abs(value - current[idx]),
            )
            groups[group_idx].append(value)

        updated: list[float] = []
        for idx, group in enumerate(groups):
            if group:
                updated.append(sum(group) / len(group))
            else:
                updated.append(current[idx])

        if all(abs(a - b) < 0.01 for a, b in zip(current, updated)):
            break

        current = updated

    return sorted(current)


def _nearest_band(value: float, centers: list[float]) -> int:
    return min(range(len(centers)), key=lambda idx: abs(value - centers[idx]))


def _collect_text_geometry_samples(
    text_cells: list[object] | None,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []

    for text_cell in text_cells or []:
        text = _normalized_text(getattr(text_cell, "text", ""))
        if not text:
            continue

        bbox = _get_bbox(text_cell)
        if bbox is None:
            continue

        center_x = _bbox_center_x(bbox)
        center_y = _bbox_center_y(bbox)
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)

        if center_x is None or width is None or width <= 0 or height <= 0:
            continue

        samples.append(
            {
                "text": text,
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }
        )

    return samples


def _predicted_span_priors(
    cells: list[TableCell],
    row_centers: list[float],
    col_centers: list[float],
) -> dict[str, tuple[int, int, int, int, bool, bool, bool]]:
    priors: dict[str, tuple[int, int, int, int, bool, bool, bool]] = {}

    if not row_centers or not col_centers:
        return priors

    for cell in cells:
        text = _normalized_text(_cell_text(cell))
        bbox = getattr(cell, "bbox", None)

        if not text or bbox is None:
            continue

        left = _bbox_left(bbox)
        right = _bbox_right(bbox)
        if left is None or right is None:
            continue

        top = float(getattr(bbox, "t"))
        bottom = float(getattr(bbox, "b"))
        min_y = min(top, bottom)
        max_y = max(top, bottom)
        min_x = min(left, right)
        max_x = max(left, right)

        covered_rows = [
            idx for idx, center in enumerate(row_centers) if min_y <= center <= max_y
        ]

        model_col_start = cell.start_col_offset_idx
        model_col_end = cell.end_col_offset_idx
        has_trusted_model_cols = 0 <= model_col_start < model_col_end <= len(
            col_centers
        ) and (
            cell.col_span > 1
            or bool(getattr(cell, "column_header", False))
            or bool(getattr(cell, "row_header", False))
        )

        if has_trusted_model_cols:
            covered_cols = list(range(model_col_start, model_col_end))
        else:
            covered_cols = [
                idx
                for idx, center in enumerate(col_centers)
                if min_x <= center <= max_x
            ]

        if not covered_rows or not covered_cols:
            continue

        start_row = min(covered_rows)
        end_row = max(covered_rows) + 1
        start_col = min(covered_cols)
        end_col = max(covered_cols) + 1

        priors[text] = (
            start_row,
            end_row,
            start_col,
            end_col,
            bool(getattr(cell, "column_header", False)),
            bool(getattr(cell, "row_header", False)),
            bool(getattr(cell, "row_section", False)),
        )

    return priors


def _extend_text_cells_into_empty_vertical_slots(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
) -> list[TableCell]:
    occupied_text_slots: set[tuple[int, int]] = set()
    for cell in cells:
        if _cell_text(cell):
            occupied_text_slots.update(_cell_slots(cell))

    extended_cells: list[TableCell] = []

    for cell in cells:
        should_consider = (
            not getattr(cell, "column_header", False)
            and not getattr(cell, "row_header", False)
            and not getattr(cell, "row_section", False)
            and cell.row_span == 1
            and cell.col_span == 1
            and _cell_text(cell)
        )

        if not should_consider:
            extended_cells.append(cell)
            continue

        start_row = cell.start_row_offset_idx
        end_row = cell.end_row_offset_idx
        start_col = cell.start_col_offset_idx
        end_col = cell.end_col_offset_idx

        if start_row < 0 or end_row > num_rows or start_col < 0 or end_col > num_cols:
            extended_cells.append(cell)
            continue

        extended_end_row = end_row

        while extended_end_row < num_rows:
            # Stop when another real text cell already owns this column slot.
            same_col_has_real_cell = any(
                (extended_end_row, col_idx) in occupied_text_slots
                for col_idx in range(start_col, end_col)
            )
            if same_col_has_real_cell:
                break

            # Only extend through visual rows that have other populated cells.
            row_has_other_real_cell = any(
                (extended_end_row, col_idx) in occupied_text_slots
                for col_idx in range(num_cols)
                if not (start_col <= col_idx < end_col)
            )
            if not row_has_other_real_cell:
                break

            extended_end_row += 1

        if extended_end_row > end_row:
            extended_cell = _copy_cell(cell)
            object.__setattr__(
                extended_cell,
                "end_row_offset_idx",
                extended_end_row,
            )
            object.__setattr__(
                extended_cell,
                "row_span",
                extended_end_row - start_row,
            )
            extended_cells.append(extended_cell)
        else:
            extended_cells.append(cell)

    diag = validate_table_topology(extended_cells, num_rows, num_cols)
    if diag.valid:
        return extended_cells

    return cells


def infer_table_from_text_geometry(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
    otsl_seq: list[str],
    text_cells: list[object] | None = None,
    allow_same_row_count: bool = False,
    allow_column_count_growth: bool = False,
) -> tuple[list[TableCell], int, int, list[str], TableTopologyDiagnostics]:
    samples = _collect_text_geometry_samples(text_cells)

    if not samples:
        diag = validate_table_topology(cells, num_rows, num_cols)
        diag.notes.append("No text geometry available for fallback.")
        return cells, num_rows, num_cols, otsl_seq, diag

    row_centers = _cluster_axis_by_tolerance(
        [(float(sample["center_y"]), float(sample["height"])) for sample in samples]
    )
    expected_col_count = (
        None if allow_column_count_growth else num_cols if num_cols > 0 else None
    )
    col_centers = _cluster_axis_expected_count(
        [(float(sample["center_x"]), float(sample["width"])) for sample in samples],
        expected_count=expected_col_count,
    )

    if len(col_centers) <= 0:
        diag = validate_table_topology(cells, num_rows, num_cols)
        return cells, num_rows, num_cols, otsl_seq, diag

    has_row_gain = len(row_centers) > num_rows
    has_col_gain = len(col_centers) > num_cols

    if not allow_same_row_count and not has_row_gain and not has_col_gain:
        diag = validate_table_topology(cells, num_rows, num_cols)
        return cells, num_rows, num_cols, otsl_seq, diag

    if allow_same_row_count and len(row_centers) < num_rows and not has_col_gain:
        diag = validate_table_topology(cells, num_rows, num_cols)
        return cells, num_rows, num_cols, otsl_seq, diag

    fallback_rows = len(row_centers)
    fallback_cols = len(col_centers)

    grid_text: dict[tuple[int, int], list[str]] = {}
    for sample in samples:
        row_idx = _nearest_band(float(sample["center_y"]), row_centers)
        col_idx = _nearest_band(float(sample["center_x"]), col_centers)
        grid_text.setdefault((row_idx, col_idx), []).append(str(sample["text"]))

    span_priors = _predicted_span_priors(cells, row_centers, col_centers)

    fallback_cells: list[TableCell] = []
    covered_slots: set[tuple[int, int]] = set()

    for row_idx in range(fallback_rows):
        for col_idx in range(fallback_cols):
            if (row_idx, col_idx) in covered_slots:
                continue

            parts = grid_text.get((row_idx, col_idx), [])
            text = _normalized_text(" ".join(parts))
            if not text:
                continue

            start_row = row_idx
            start_col = col_idx
            end_row = row_idx + 1
            end_col = col_idx + 1
            column_header = False
            row_header = False
            row_section = False
            used_prior_span = False

            prior = span_priors.get(text)
            if prior is not None:
                (
                    p_start_row,
                    p_end_row,
                    p_start_col,
                    p_end_col,
                    p_col_header,
                    p_row_header,
                    p_row_section,
                ) = prior

                # A merged-cell text is often geometrically centered inside the
                # merged rectangle, not at the top-left grid slot. Preserve the
                # model span when the inferred text slot falls inside it.
                if (
                    p_start_row <= row_idx < p_end_row
                    and p_start_col <= col_idx < p_end_col
                ):
                    start_row = p_start_row
                    start_col = p_start_col
                    end_row = max(end_row, p_end_row)
                    end_col = max(end_col, p_end_col)
                    column_header = p_col_header
                    row_header = p_row_header
                    row_section = p_row_section
                    used_prior_span = True

            # Generic vertical merge:
            # if the current column has empty slots below until the next text,
            # merge the current cell downward. Do not extend cells that already
            # used a model-provided prior span; that span is more trustworthy
            # than this fallback heuristic.
            while (
                not used_prior_span
                and end_row < fallback_rows
                and not grid_text.get((end_row, col_idx))
            ):
                # Do not merge past a row where other columns are completely
                # empty; that usually means geometry noise rather than a table row.
                row_has_any_text = any(
                    grid_text.get((end_row, other_col))
                    for other_col in range(fallback_cols)
                )
                if not row_has_any_text:
                    break
                end_row += 1

            end_row = min(end_row, fallback_rows)
            end_col = min(end_col, fallback_cols)

            fallback_cell = _make_table_cell(
                start_row=start_row,
                end_row=end_row,
                start_col=start_col,
                end_col=end_col,
                text=text,
                column_header=column_header,
                row_header=row_header,
                row_section=row_section,
            )
            fallback_cells.append(fallback_cell)

            anchor = (
                fallback_cell.start_row_offset_idx,
                fallback_cell.start_col_offset_idx,
            )
            for slot in _cell_slots(fallback_cell):
                if slot != anchor:
                    covered_slots.add(slot)

    def _slot_has_text(row: int, col: int) -> bool:
        return any(
            _cell_text(cell)
            and cell.start_row_offset_idx <= row < cell.end_row_offset_idx
            and cell.start_col_offset_idx <= col < cell.end_col_offset_idx
            for cell in fallback_cells
        )

    def _row_has_column_header(row: int) -> bool:
        return any(
            getattr(cell, "column_header", False)
            and cell.start_row_offset_idx <= row < cell.end_row_offset_idx
            for cell in fallback_cells
        )

    def _row_has_non_first_col_text(row: int) -> bool:
        return any(
            _cell_text(cell)
            and cell.start_col_offset_idx > 0
            and cell.start_row_offset_idx <= row < cell.end_row_offset_idx
            for cell in fallback_cells
        )

    # Recover simple parent header spans, for example a centered "Schedule"
    # above the adjacent "Begin" and "End" child headers.
    for cell in fallback_cells:
        if (
            cell.col_span != 1
            or not _cell_text(cell)
            or cell.start_col_offset_idx <= 0
            or cell.end_row_offset_idx >= fallback_rows
        ):
            continue

        if not (
            _row_has_column_header(cell.start_row_offset_idx)
            or _row_has_column_header(cell.end_row_offset_idx)
        ):
            continue

        left_col = cell.start_col_offset_idx - 1
        current_col = cell.start_col_offset_idx
        child_row = cell.end_row_offset_idx

        left_child = _slot_has_text(child_row, left_col)
        current_child = _slot_has_text(child_row, current_col)
        left_parent_slot_empty = not _slot_has_text(cell.start_row_offset_idx, left_col)

        if left_child and current_child and left_parent_slot_empty:
            cell.start_col_offset_idx = left_col
            cell.col_span = cell.end_col_offset_idx - cell.start_col_offset_idx

    # Recover first-column row labels whose text is vertically centered in a
    # merged visual cell. Move them up one empty body row if needed, then extend
    # each label until the next first-column label.
    first_col_labels = sorted(
        [
            cell
            for cell in fallback_cells
            if _cell_text(cell)
            and cell.start_col_offset_idx == 0
            and cell.end_col_offset_idx == 1
            and not getattr(cell, "column_header", False)
        ],
        key=lambda item: item.start_row_offset_idx,
    )

    for idx, cell in enumerate(first_col_labels):
        # Do not move the final first-column label upward. Without a following
        # label to bound it, moving the last label can steal the final row from
        # the previous group.
        if idx + 1 >= len(first_col_labels):
            continue

        previous_row = cell.start_row_offset_idx - 1
        if (
            previous_row >= 0
            and not _row_has_column_header(previous_row)
            and not _slot_has_text(previous_row, 0)
            and _row_has_non_first_col_text(previous_row)
        ):
            cell.start_row_offset_idx = previous_row
            cell.row_span = cell.end_row_offset_idx - cell.start_row_offset_idx

    first_col_labels = sorted(
        first_col_labels,
        key=lambda item: item.start_row_offset_idx,
    )

    for idx, cell in enumerate(first_col_labels):
        if idx + 1 >= len(first_col_labels):
            continue

        next_start = first_col_labels[idx + 1].start_row_offset_idx
        if next_start > cell.start_row_offset_idx:
            cell.end_row_offset_idx = next_start
            cell.row_span = cell.end_row_offset_idx - cell.start_row_offset_idx

    fallback_diag = validate_table_topology(
        fallback_cells,
        fallback_rows,
        fallback_cols,
    )

    if not fallback_diag.valid:
        fallback_diag.notes.append("Geometry fallback produced invalid topology.")
        return cells, num_rows, num_cols, otsl_seq, fallback_diag

    fallback_cells = _extend_text_cells_into_empty_vertical_slots(
        fallback_cells,
        fallback_rows,
        fallback_cols,
    )

    fallback_diag = validate_table_topology(
        fallback_cells,
        fallback_rows,
        fallback_cols,
    )

    fallback_otsl = cells_to_otsl(fallback_cells, fallback_rows, fallback_cols)
    return fallback_cells, fallback_rows, fallback_cols, fallback_otsl, fallback_diag


def looks_overspanned_by_text_geometry(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int | None = None,
    text_cells: list[object] | None = None,
) -> bool:
    if num_rows <= 0 or not text_cells:
        return False

    samples = _collect_text_geometry_samples(text_cells)
    if not samples:
        return False

    expected_cols = num_cols
    if expected_cols is None or expected_cols <= 0:
        expected_cols = max(
            (cell.end_col_offset_idx for cell in cells),
            default=0,
        )

    row_centers = _cluster_axis_expected_count(
        [(float(sample["center_y"]), float(sample["height"])) for sample in samples],
        expected_count=num_rows,
    )
    col_centers = _cluster_axis_expected_count(
        [(float(sample["center_x"]), float(sample["width"])) for sample in samples],
        expected_count=expected_cols if expected_cols > 0 else None,
    )

    if len(row_centers) < num_rows or not col_centers:
        return False

    for cell in cells:
        if (
            getattr(cell, "column_header", False)
            or getattr(cell, "row_header", False)
            or getattr(cell, "row_section", False)
        ):
            continue

        if cell.row_span <= 1 or not _cell_text(cell):
            continue

        matching_rows: set[int] = set()

        for sample in samples:
            row_idx = _nearest_band(float(sample["center_y"]), row_centers)
            col_idx = _nearest_band(float(sample["center_x"]), col_centers)

            if (
                cell.start_row_offset_idx <= row_idx < cell.end_row_offset_idx
                and cell.start_col_offset_idx <= col_idx < cell.end_col_offset_idx
            ):
                matching_rows.add(row_idx)

        if len(matching_rows) >= 2:
            return True

    return False


def _estimate_text_col_bands(text_cells: list[object] | None) -> int:
    from statistics import median

    centers: list[float] = []
    widths: list[float] = []

    for text_cell in text_cells or []:
        bbox = getattr(text_cell, "bbox", None)
        if bbox is None:
            continue

        width = _bbox_width(bbox)
        center = _bbox_center_x(bbox)

        if width is None or center is None:
            continue

        if width > 0:
            widths.append(width)

        centers.append(center)

    if not centers:
        return 0

    tolerance = 12.0
    if widths:
        tolerance = max(tolerance, median(widths) * 0.55)

    bands: list[list[float]] = []

    for center in sorted(centers):
        if not bands:
            bands.append([center])
            continue

        band_center = sum(bands[-1]) / len(bands[-1])
        if abs(center - band_center) <= tolerance:
            bands[-1].append(center)
        else:
            bands.append([center])

    return len(bands)


def looks_undersegmented(
    cells: list[TableCell],
    num_rows: int,
    otsl_seq: list[str],
    text_cells: list[object] | None = None,
    num_cols: int | None = None,
) -> bool:
    from statistics import median

    if num_rows <= 0 or (num_cols is not None and num_cols <= 0):
        return True

    # Generic signal 1:
    # The PDF/text geometry has clearly more column bands than the decoded table.
    # Check this before the vertical-span guard because a table can contain
    # row-span markers and still have too few decoded columns.
    text_col_bands = _estimate_text_col_bands(text_cells)
    if num_cols is not None and text_col_bands >= num_cols + 1:
        return True

    # If OTSL already contains vertical-span markers, do not classify it as
    # simple row under-segmentation just because geometry is complex.
    if any(tok in {"ucel", "xcel"} for tok in otsl_seq):
        return False

    # Generic signal 2:
    # The PDF/text geometry has clearly more row bands than the OTSL table.
    text_row_bands = _estimate_text_row_bands(text_cells)
    if text_row_bands >= num_rows + 2:
        return True

    # Generic signal 3:
    # Some body cells are much taller than typical body cells, which often
    # means multiple visual rows collapsed into one logical row.
    body_heights: list[float] = []
    for cell in cells:
        if getattr(cell, "column_header", False) or getattr(cell, "row_header", False):
            continue

        bbox = getattr(cell, "bbox", None)
        if bbox is None:
            continue

        height = _bbox_height(bbox)
        if height > 0:
            body_heights.append(height)

    if len(body_heights) < 3:
        return False

    typical_height = median(body_heights)
    if typical_height <= 0:
        return False

    tall_cells = [height for height in body_heights if height > typical_height * 1.75]
    return len(tall_cells) >= max(1, len(body_heights) // 5)
