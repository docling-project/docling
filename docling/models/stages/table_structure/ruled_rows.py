# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections import defaultdict
from collections.abc import Callable
from unicodedata import east_asian_width

from docling_core.types.doc import BoundingBox, TableCell

_RULE_TOLERANCE = 0.25
_TABLE_EDGE_TOLERANCE = 3.0


def _has_column_edges(
    segments: list[tuple[float, float]], columns: list[BoundingBox]
) -> bool:
    for index, box in enumerate(columns):
        previous = columns[index - 1] if index else None
        following = columns[index + 1] if index + 1 < len(columns) else None
        if not any(
            left <= box.l + _RULE_TOLERANCE
            and right >= box.r - _RULE_TOLERANCE
            and (previous is None or left > previous.r)
            and (following is None or right < following.l)
            for left, right in segments
        ):
            return False
    return True


def _is_one_printed_row(
    columns: list[BoundingBox],
    rules: list[BoundingBox],
    previous_bottom: float,
    table_bottom: float,
) -> bool:
    row = BoundingBox.enclosing_bbox(columns)
    horizontal: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for rule in rules:
        if rule.height > _RULE_TOLERANCE or not row.x_overlap_with(rule):
            continue
        center = (rule.t + rule.b) / 2
        y = round(center / _RULE_TOLERANCE) * _RULE_TOLERANCE
        if row.t < y < row.b:
            return False
        if previous_bottom <= y <= table_bottom + _TABLE_EDGE_TOLERANCE:
            horizontal[y].append((rule.l, rule.r))

    return any(
        previous_bottom <= y <= row.t and _has_column_edges(segments, columns)
        for y, segments in horizontal.items()
    ) and any(
        row.b <= y
        and abs(y - table_bottom) <= _TABLE_EDGE_TOLERANCE
        and _has_column_edges(segments, columns)
        for y, segments in horizontal.items()
    )


def repair_split_final_row(
    *,
    cells: list[TableCell],
    tags: list[str],
    num_rows: int,
    num_cols: int,
    get_rules: Callable[[], list[BoundingBox] | None],
    table_box: BoundingBox,
) -> tuple[list[TableCell], list[str]] | None:
    """Recover a wrapped stub and its values within a single printed bottom row.

    The decoder can emit a row for each text line when the values align with
    the second line of a wrapped label. Require complementary predictions and
    printed edges for each column above and below the two lines. A table-wide
    rule alone does not establish that the text belongs to one row.
    """
    stride = num_cols + 1
    if num_rows < 3 or num_cols < 4 or len(tags) < 2 * stride:
        return None
    upper, lower = tags[-2 * stride : -stride], tags[-stride:]
    num_labels = 1 + upper.count("fcel")
    num_values = num_cols - num_labels
    if num_labels < 2 or num_values < 2:
        return None
    label_tags = ["rhed", *(["fcel"] * (num_labels - 1))]
    value_tags = ["fcel"] * num_values
    if upper != [*label_tags, *(["ecel"] * num_values), "nl"] or lower != [
        *(["ecel"] * (num_labels - 1)),
        "fcel",
        *value_tags,
        "nl",
    ]:
        return None

    first = num_rows - 2
    earlier = [cell for cell in cells if cell.end_row_offset_idx <= first]
    tail = [cell for cell in cells if cell.end_row_offset_idx > first]
    if any(
        cell.start_row_offset_idx not in (first, first + 1)
        or cell.end_row_offset_idx != cell.start_row_offset_idx + 1
        or cell.end_col_offset_idx != cell.start_col_offset_idx + 1
        or cell.bbox is None
        or not cell.text.strip()
        or cell.column_header
        or cell.row_section
        for cell in tail
    ):
        return None
    tail.sort(key=lambda cell: (cell.start_row_offset_idx, cell.start_col_offset_idx))
    labels, lower_cells = tail[:num_labels], tail[num_labels:]
    if [(c.start_row_offset_idx, c.start_col_offset_idx) for c in labels] != [
        (first, col) for col in range(num_labels)
    ] or len(lower_cells) != num_values + 1:
        return None
    fragment, *values = lower_cells
    if (
        fragment.start_row_offset_idx != first + 1
        or not 0 <= fragment.start_col_offset_idx < num_labels - 1
        or [(c.start_row_offset_idx, c.start_col_offset_idx) for c in values]
        != [(first + 1, col) for col in range(num_labels, num_cols)]
    ):
        return None

    label = labels[-1]
    assert label.bbox is not None and fragment.bbox is not None
    if fragment.bbox.t <= label.bbox.b or not label.bbox.x_overlap_with(fragment.bbox):
        return None
    label_box = BoundingBox.enclosing_bbox([label.bbox, fragment.bbox])
    columns = [cell.bbox for cell in [*labels, *values] if cell.bbox is not None]
    columns[num_labels - 1] = label_box
    previous_bottom = max(
        (cell.bbox.b for cell in earlier if cell.bbox is not None), default=table_box.t
    )
    rules = get_rules()
    if not rules or not _is_one_printed_row(
        columns, rules, previous_bottom, table_box.b
    ):
        return None

    left, right = label.text.rstrip(), fragment.text.lstrip()
    separator = (
        ""
        if any(east_asian_width(c) in ("W", "F") for c in (left[-1], right[0]))
        else " "
    )
    wrapped_label = label.model_copy(
        update={"text": left + separator + right, "bbox": label_box}
    )
    moved_values = [
        value.model_copy(
            update={"start_row_offset_idx": first, "end_row_offset_idx": first + 1}
        )
        for value in values
    ]
    return [*earlier, *labels[:-1], wrapped_label, *moved_values], [
        *tags[: -2 * stride],
        *label_tags,
        *value_tags,
        "nl",
    ]
