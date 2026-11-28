# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import BoundingBox, TableCell

from docling.models.stages.table_structure.ruled_rows import repair_split_final_row


def _cell(row: int, col: int, text: str, left: float, top: float) -> TableCell:
    return TableCell(
        bbox=BoundingBox(l=left, t=top, r=left + 30, b=top + 8),
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=col,
        end_col_offset_idx=col + 1,
        text=text,
        row_header=col == 0 and row != 2,
    )


def _table() -> tuple[list[TableCell], list[str]]:
    cells = [
        _cell(0, 0, "Opening", 2, 6),
        _cell(0, 1, "期初", 52, 6),
        _cell(0, 2, "12,400", 106, 6),
        _cell(0, 3, "10", 156, 6),
        _cell(1, 0, "Closing", 2, 23),
        _cell(1, 1, "於2025年", 52, 23),
        _cell(2, 0, "12月31日", 53, 37),
        _cell(2, 2, "10,500", 106, 37),
        _cell(2, 3, "8", 156, 37),
    ]
    tags = "rhed fcel fcel fcel nl rhed fcel ecel ecel nl ecel fcel fcel fcel nl"
    return cells, tags.split()


def _segmented_rule(y: float) -> list[BoundingBox]:
    return [BoundingBox(l=x, t=y, r=x + 50, b=y) for x in range(0, 200, 50)]


def test_complete_rows_do_not_request_page_rules() -> None:
    cells, tags = _table()
    tags[5:10] = "rhed fcel fcel fcel nl".split()

    def get_rules() -> list[BoundingBox]:
        pytest.fail("TableFormer already predicted a complete row")

    assert (
        repair_split_final_row(
            cells=cells,
            tags=tags,
            num_rows=3,
            num_cols=4,
            get_rules=get_rules,
            table_box=BoundingBox(l=0, t=0, r=200, b=49),
        )
        is None
    )


def test_wrapped_final_row_preserves_labels_amounts_and_boxes() -> None:
    cells, tags = _table()
    result = repair_split_final_row(
        cells=cells,
        tags=tags,
        num_rows=3,
        num_cols=4,
        get_rules=lambda: [*_segmented_rule(18), *_segmented_rule(48)],
        table_box=BoundingBox(l=0, t=0, r=200, b=49),
    )

    assert result is not None
    merged, new_tags = result
    assert new_tags == "rhed fcel fcel fcel nl rhed fcel fcel fcel nl".split()
    assert merged[:5] == cells[:5]
    assert [
        (cell.start_row_offset_idx, cell.start_col_offset_idx) for cell in merged[4:]
    ] == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert [cell.text for cell in merged[4:]] == [
        "Closing",
        "於2025年12月31日",
        "10,500",
        "8",
    ]
    assert merged[5].bbox == BoundingBox(l=52, t=23, r=83, b=45)
    assert [cell.bbox for cell in merged[6:]] == [cell.bbox for cell in cells[7:]]
    assert cells[6].start_row_offset_idx == 2


@pytest.mark.parametrize(
    "geometry", ["divider", "wide_rules", "unruled", "different_label"]
)
def test_incomplete_evidence_leaves_rows_alone(geometry: str) -> None:
    cells, tags = _table()
    rules = [*_segmented_rule(18), *_segmented_rule(48)]
    if geometry == "divider":
        rules.extend(_segmented_rule(34))
    elif geometry == "wide_rules":
        rules = [BoundingBox(l=0, t=y, r=200, b=y) for y in (18, 48)]
    elif geometry == "unruled":
        rules = []
    else:
        cells[6] = _cell(2, 0, "Adjustments", 2, 37)

    assert (
        repair_split_final_row(
            cells=cells,
            tags=tags,
            num_rows=3,
            num_cols=4,
            get_rules=lambda: rules,
            table_box=BoundingBox(l=0, t=0, r=200, b=49),
        )
        is None
    )
