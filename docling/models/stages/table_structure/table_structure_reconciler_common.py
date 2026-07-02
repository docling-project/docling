from __future__ import annotations

import copy


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cell_row_range(cell) -> range:
    start = _safe_int(getattr(cell, "start_row_offset_idx", None))
    end = _safe_int(getattr(cell, "end_row_offset_idx", None))
    if start is None or end is None:
        return range(0)
    return range(start, end)


def _cell_col_range(cell) -> range:
    start = _safe_int(getattr(cell, "start_col_offset_idx", None))
    end = _safe_int(getattr(cell, "end_col_offset_idx", None))
    if start is None or end is None:
        return range(0)
    return range(start, end)


def _first_float_value(*values) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _cell_row_offsets(cell: object) -> tuple[int, int] | None:
    start_row = _safe_int(getattr(cell, "start_row_offset_idx", None))
    end_row = _safe_int(getattr(cell, "end_row_offset_idx", None))

    if start_row is None or end_row is None or end_row <= start_row:
        return None

    return start_row, end_row


def _cell_column_offsets(cell: object) -> tuple[int, int] | None:
    start_col = _safe_int(getattr(cell, "start_col_offset_idx", None))
    end_col = _safe_int(getattr(cell, "end_col_offset_idx", None))

    if start_col is None or end_col is None or end_col <= start_col:
        return None

    return start_col, end_col


def _copy_cell_with_offsets(
    cell,
    *,
    start_row: int | None = None,
    end_row: int | None = None,
    start_col: int | None = None,
    end_col: int | None = None,
    text: str | None = None,
):
    copied_cell = copy.deepcopy(cell)

    current_start_row, current_end_row = _cell_row_offsets(cell)
    current_start_col, current_end_col = _cell_column_offsets(cell)

    new_start_row = current_start_row if start_row is None else start_row
    new_end_row = current_end_row if end_row is None else end_row
    new_start_col = current_start_col if start_col is None else start_col
    new_end_col = current_end_col if end_col is None else end_col

    copied_cell.start_row_offset_idx = new_start_row
    copied_cell.end_row_offset_idx = new_end_row
    copied_cell.start_col_offset_idx = new_start_col
    copied_cell.end_col_offset_idx = new_end_col
    copied_cell.row_span = new_end_row - new_start_row
    copied_cell.col_span = new_end_col - new_start_col

    if text is not None:
        copied_cell.text = text

    return copied_cell


def _joined_interval_text(intervals) -> str:
    return " ".join(
        str(getattr(interval, "text", "") or "").strip()
        for interval in intervals
        if str(getattr(interval, "text", "") or "").strip()
    )


def _set_cell_header_flags(
    cell,
    *,
    column_header: bool | None = None,
    row_header: bool | None = None,
):
    copied_cell = copy.deepcopy(cell)

    if column_header is not None:
        copied_cell.column_header = column_header

    if row_header is not None:
        copied_cell.row_header = row_header

    return copied_cell


def _row_has_column_header(model_cells, row_idx: int) -> bool:
    return any(
        getattr(cell, "column_header", False) and row_idx in _cell_row_range(cell)
        for cell in model_cells
    )


def _infer_split_upper_row_is_column_header(
    model_cells,
    new_cells,
    *,
    upper_row: int,
    lower_row: int,
) -> bool:
    if _row_has_column_header(model_cells, upper_row):
        return True

    upper_text_cells = [
        cell
        for cell in new_cells
        if _cell_row_offsets(cell) == (upper_row, upper_row + 1)
        and str(getattr(cell, "text", "") or "").strip()
    ]
    lower_text_cells = [
        cell
        for cell in new_cells
        if _cell_row_offsets(cell) == (lower_row, lower_row + 1)
        and str(getattr(cell, "text", "") or "").strip()
    ]

    if not upper_text_cells or not lower_text_cells:
        return False

    upper_cols = {_cell_column_offsets(cell) for cell in upper_text_cells}
    lower_cols = {_cell_column_offsets(cell) for cell in lower_text_cells}

    # Header rows usually label several columns and are followed by a data row.
    return len(upper_cols) >= 2 and upper_cols != lower_cols
