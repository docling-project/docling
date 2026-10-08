# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Behavioral checks of the AcroForm keying on small synthetic pages."""

from dataclasses import dataclass, field, replace

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, PdfWidget, TextCell

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.keying import (
    Assignment,
    Scope,
    assign,
    regions,
    scope_of,
)


@dataclass
class KeyingPage:
    """The arguments of ``assign`` for one synthetic page."""

    widgets: list[PdfWidget]
    layout: list[Cluster]
    tables: dict[int, list[TableCell]] = field(default_factory=dict)
    height: float = 400
    rules: list[BoundingBox] = field(default_factory=list)


def box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def widget(
    index: int, bbox: BoundingBox, *, checkbox: bool = False, name: str | None = None
) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        widget_field_type="/Btn" if checkbox else "/Tx",
        widget_field_name=name,
        widget_field_flags=0,
        widget_appearance_state="/Off" if checkbox else None,
        widget_text="/Off" if checkbox else "retained value",
    )


def key_text(index: int, text: str, bbox: BoundingBox) -> Cluster:
    return Cluster(
        id=index,
        label=DocItemLabel.TEXT,
        bbox=bbox,
        cells=[
            TextCell(
                index=index,
                text=text,
                orig=text,
                from_ocr=False,
                rect=BoundingRectangle.from_bounding_box(bbox),
            )
        ],
    )


def table(bbox: BoundingBox) -> Cluster:
    return Cluster(id=100, label=DocItemLabel.TABLE, bbox=bbox)


def snapshot(
    widgets: list[PdfWidget],
    key_texts: list[Cluster],
    tables: dict[int, list[TableCell]] | None = None,
) -> KeyingPage:
    return KeyingPage(widgets, key_texts, tables or {})


def keyed(page: KeyingPage) -> Assignment:
    return assign(page.widgets, page.layout, page.tables, page.height, page.rules)


def chosen(page: KeyingPage) -> tuple[list[int], list[tuple[str, list[int], str]]]:
    result = keyed(page)
    assert result.solver_status == "optimal"
    fields = [
        (
            result.candidates[c].kind,
            [result.values[i].native.index for i in result.candidates[c].members],
            result.key_texts[result.candidates[c].key].text,
        )
        for c in result.selected
    ]
    return [v.native.index for v in result.values], fields


def test_table_exception_requires_one_cell_and_never_uses_other_cell_header():
    table_region = table(box(0, 40, 200, 200))
    cells = [
        TableCell(
            bbox=box(0, 40, 100, 200),
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="",
        ),
        TableCell(
            bbox=box(100, 40, 200, 200),
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="",
        ),
    ]
    page = snapshot(
        [widget(0, box(82, 80, 98, 95)), widget(1, box(110, 150, 170, 170))],
        [
            table_region,
            key_text(1, "Local name", box(10, 82, 65, 92)),
            key_text(2, "Other cell", box(101, 81, 165, 91)),
            key_text(3, "Outside table", box(80, 25, 180, 35)),
        ],
        {100: cells},
    )
    order, fields = chosen(page)
    assert order == [0, 1]
    assert ("field_key", [0], "Local name") in fields
    assert all(text != "Outside table" for _, _, text in fields)
    assert scope_of(box(95, 80, 105, 90), regions(page.layout), page.tables) == Scope(
        100
    )
    # Even near-total coverage must not move a cross-cell widget into one cell.
    assert scope_of(box(82, 80, 100.1, 95), regions(page.layout), page.tables) == Scope(
        100
    )
    assert scope_of(box(82, 80, 98, 95), regions(page.layout), page.tables) == Scope(
        100, 0
    )

    # A detected table with no usable cells stays excluded, even with an
    # attractive nearby key. Removing detection restores ordinary pairing.
    page.tables = {}
    assert chosen(page)[1] == []
    page.layout.remove(table_region)
    assert chosen(page)[1]


def test_column_reset_and_candidate_enumeration_preserve_native_order():
    page = snapshot(
        [
            widget(7, box(10, 40, 20, 50), checkbox=True),
            widget(2, box(10, 80, 20, 90), checkbox=True),
            widget(9, box(200, 40, 210, 50), checkbox=True),
            widget(3, box(200, 80, 210, 90), checkbox=True),
        ],
        [
            key_text(1, "English", box(25, 40, 90, 50)),
            key_text(2, "Spanish", box(25, 80, 90, 90)),
            key_text(3, "French", box(215, 40, 280, 50)),
            key_text(4, "Arabic", box(215, 80, 280, 90)),
        ],
    )
    order, fields = chosen(page)
    assert order == [7, 2, 9, 3]
    assert {(tuple(indices), text) for _, indices, text in fields} == {
        ((7,), "English"),
        ((2,), "Spanish"),
        ((9,), "French"),
        ((3,), "Arabic"),
    }
    page.layout.reverse()
    assert chosen(page) == (order, fields)


def test_shared_prompt_keeps_keys_and_interleaved_native_values():
    page = snapshot(
        [
            widget(0, box(10, 50, 20, 60), checkbox=True, name="same native field"),
            widget(1, box(220, 120, 300, 140)),
            widget(2, box(10, 90, 20, 100), checkbox=True, name="same native field"),
        ],
        [
            key_text(1, "Preferred language", box(10, 15, 140, 25)),
            key_text(2, "English", box(25, 50, 90, 60)),
            key_text(3, "French", box(25, 90, 90, 100)),
            key_text(4, "Your name", box(220, 103, 290, 113)),
        ],
    )
    order, fields = chosen(page)
    assert order == [0, 1, 2]
    assert ("choice_group", [0, 2], "Preferred language") in fields
    assert ("option_key", [0], "English") in fields
    assert ("option_key", [2], "French") in fields
    assert ("field_key", [1], "Your name") in fields


@pytest.mark.parametrize("scale, offset", [(1, 0), (3.5, 23)])
@pytest.mark.parametrize("text", ["Extension requested until", "期限延長"])
def test_inline_checkbox_and_date_share_clause_without_duplicate_ownership(
    scale: float, offset: float, text: str
) -> None:
    def positioned(left: float, top: float, right: float, bottom: float) -> BoundingBox:
        return box(*(v * scale + offset for v in (left, top, right, bottom)))

    page = snapshot(
        [
            widget(7, positioned(10, 60, 20, 70), checkbox=True),
            widget(2, positioned(160, 60, 230, 70)),
            widget(9, positioned(160, 67, 230, 77)),
        ],
        # Both clause controls have 80% coverage despite protruding below the
        # text bounds. The neighboring control has only 10% and must not join.
        [key_text(1, text, positioned(5, 55, 250, 68))],
    )
    page.height = 400 * scale + offset
    order, fields = chosen(page)
    assert order == [7, 2, 9]
    assert fields == [("inline_clause", [7, 2], text)]


def test_option_checkboxes_inside_a_sentence_keep_their_own_keys():
    # "(a) Business income [ ] Yes [ ] No": the layout returns the line as one
    # block, but each checkbox has its own option key printed right after
    # it. Options answer the sentence; they are not blanks within it, so each
    # takes its key instead of the whole line.
    cells = [
        ("(a) Business income", box(10, 50, 110, 58)),
        ("Yes", box(132, 50, 150, 58)),
        ("No", box(172, 50, 186, 58)),
    ]
    row = Cluster(
        id=1,
        label=DocItemLabel.TEXT,
        bbox=box(10, 50, 186, 58),
        cells=[
            TextCell(
                index=k,
                text=text,
                orig=text,
                from_ocr=False,
                rect=BoundingRectangle.from_bounding_box(bbox),
            )
            for k, (text, bbox) in enumerate(cells)
        ],
    )
    page = snapshot(
        [
            widget(0, box(120, 50, 128, 58), checkbox=True),
            widget(1, box(160, 50, 168, 58), checkbox=True),
        ],
        [row],
    )
    _, fields = chosen(page)
    assert ("option_key", [0], "Yes") in fields
    assert ("option_key", [1], "No") in fields


def test_shared_business_number_is_not_split_at_printed_component():
    page = snapshot(
        [
            widget(0, box(40, 70, 150, 85)),
            widget(1, box(168, 70, 230, 85)),
        ],
        [
            key_text(1, "Business Number", box(40, 40, 140, 50)),
            key_text(2, "RT", box(152, 72, 166, 82)),
        ],
    )
    _, fields = chosen(page)
    assert fields == [("composite_field", [0, 1], "Business Number")]


def test_row_and_column_headers_never_key_a_value_in_a_detected_table():
    def cell(row: int, column: int, text: str, bbox: BoundingBox, header=False):
        return TableCell(
            bbox=bbox,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            text=text,
            column_header=header,
        )

    # Detected cell boxes cover only their text; the values sit beside the
    # line codes, not inside any cell box.
    cells = [
        cell(0, 1, "From head office", box(120, 40, 190, 50), header=True),
        cell(0, 2, "From third parties", box(220, 40, 290, 50), header=True),
        cell(1, 0, "Financial services", box(10, 62, 90, 72)),
        cell(1, 1, "250", box(120, 62, 135, 72)),
        cell(1, 2, "251", box(220, 62, 235, 72)),
        cell(2, 0, "Taxable goods", box(10, 82, 80, 92)),
        cell(2, 1, "260", box(120, 82, 135, 92)),
        cell(2, 2, "261", box(220, 82, 235, 92)),
    ]
    page = snapshot(
        [
            widget(0, box(140, 60, 200, 74)),
            widget(1, box(240, 60, 300, 74)),
            widget(2, box(140, 80, 200, 94)),
            widget(3, box(240, 80, 300, 94)),
        ],
        [table(box(5, 35, 305, 100))],
        {100: cells},
    )
    result = keyed(page)
    # "Financial services" (row header) and "From head office" (column header)
    # sit in other cells: the table already carries that association.
    assert result.selected == []
    # Each value still gets the cell of its line code: a code has no letters,
    # so the cell has no key of its own.
    cells = {
        result.values[i].native.index: (slot.table, slot.rows, slot.columns, slot.key)
        for i, slot in result.slots.items()
    }
    assert cells == {
        0: (100, (1, 2), (1, 2), None),
        1: (100, (1, 2), (2, 3), None),
        2: (100, (2, 3), (1, 2), None),
        3: (100, (2, 3), (2, 3), None),
    }


def grid_cell(
    row: int, column: int, text: str, bbox: BoundingBox, *, columns: int = 1
) -> TableCell:
    return TableCell(
        bbox=bbox,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=column,
        end_col_offset_idx=column + columns,
        text=text,
    )


def ruled(page: KeyingPage, *lines: tuple[str, float, float, float]) -> KeyingPage:
    """The page with printed rules: ("h", y, x0, x1) or ("v", x, y0, y1), 0.5 pt thick."""
    boxes = [
        box(start, at - 0.25, end, at + 0.25)
        if kind == "h"
        else box(at - 0.25, start, at + 0.25, end)
        for kind, at, start, end in lines
    ]
    return replace(page, rules=boxes)


def test_printed_cell_moves_a_box_to_the_key_printed_over_it():
    # "Name:" and its box share one printed cell, but the detected grid puts
    # the key in its own row, and a title cell whose box runs down beside
    # it takes the box. The printed cell holds the key: the box goes to
    # the key's cell, keyed by it.
    page = snapshot(
        [widget(0, box(5, 29, 95, 39))],
        [table(box(0, 0, 200, 100))],
        {
            100: [
                grid_cell(0, 0, "Personal data", box(2, 2, 60, 30), columns=2),
                grid_cell(1, 0, "Name:", box(5, 22, 40, 28)),
                grid_cell(1, 1, "Date:", box(105, 22, 140, 28)),
                grid_cell(2, 0, "Total", box(5, 62, 40, 68)),
                grid_cell(2, 1, "Sum", box(105, 62, 140, 68)),
            ]
        },
    )

    def placed(result):
        (slot,) = result.slots.values()
        (key,) = (
            result.key_texts[c.key].text
            for c in (result.candidates[i] for i in result.selected)
            if c.kind == "table_cell"
        )
        return slot.rows, slot.columns, key

    assert placed(keyed(page)) == ((0, 1), (0, 2), "Personal data")
    printed = ruled(
        page,
        ("h", 20, 0, 200),
        ("h", 40, 0, 200),
        ("v", 0, 0, 100),
        ("v", 100, 20, 40),
        ("v", 200, 0, 100),
    )
    assert placed(keyed(printed)) == ((1, 2), (0, 1), "Name:")


def test_value_in_a_printed_column_the_grid_lost_gets_no_cell():
    # The printed table has a second column the detected grid missed: the
    # value's printed cell holds no text and misses the grid's only column, so
    # it gets no cell instead of the row number's.
    page = snapshot(
        [widget(0, box(120, 20, 190, 30))],
        [table(box(0, 0, 200, 60))],
        {
            100: [
                grid_cell(0, 0, "Name", box(5, 2, 40, 8)),
                grid_cell(1, 0, "1", box(5, 22, 15, 28)),
                grid_cell(2, 0, "2", box(5, 42, 15, 48)),
            ]
        },
    )
    assert [(s.rows, s.columns) for s in keyed(page).slots.values()] == [
        ((1, 2), (0, 1))
    ]
    printed = ruled(
        page,
        ("h", 15, 0, 200),
        ("h", 35, 0, 200),
        ("v", 0, 0, 60),
        ("v", 100, 0, 60),
        ("v", 200, 0, 60),
    )
    assert keyed(printed).slots == {}


def test_row_key_is_shared_by_like_sized_values_but_not_operand_boxes():
    page = snapshot(
        [
            widget(0, box(100, 60, 160, 72)),
            widget(1, box(170, 60, 190, 72)),
            widget(2, box(200, 60, 260, 72)),
        ],
        [key_text(1, "Line 5 total", box(10, 61, 80, 71))],
    )
    _, fields = chosen(page)
    assert sorted(fields) == [
        ("field_key", [0], "Line 5 total"),
        ("field_key", [2], "Line 5 total"),
    ]


def test_key_split_per_line_is_joined_but_option_lines_are_not():
    page = snapshot(
        [
            widget(0, box(150, 72, 220, 84)),
            widget(1, box(12, 221, 20, 229), checkbox=True),
            widget(2, box(12, 231, 20, 239), checkbox=True),
        ],
        [
            key_text(1, "Intangible personal", box(10, 60, 100, 68)),
            key_text(2, "property and services", box(10, 69, 110, 77)),
            key_text(3, "financial services", box(10, 78, 95, 86)),
            key_text(4, "a Parent group", box(10, 220, 90, 230)),
            key_text(5, "b Brother group", box(10, 230, 95, 240)),
        ],
    )
    _, fields = chosen(page)
    assert (
        "field_key",
        [0],
        "Intangible personal property and services financial services",
    ) in fields
    assert ("option_key", [1], "a Parent group") in fields
    assert ("option_key", [2], "b Brother group") in fields


def test_line_split_into_pieces_is_joined_but_neighbouring_cells_are_not():
    page = snapshot(
        [
            widget(0, box(250, 60, 320, 72)),
            widget(1, box(40, 131, 60, 143)),
            widget(2, box(66, 131, 86, 143)),
        ],
        [
            key_text(1, "Manitoba", box(10, 61, 45, 69)),
            key_text(2, "tax (line 23)", box(47, 61, 100, 69)),
            # Sub-keys over their own boxes: separate cells, never one line.
            key_text(3, "GIORNO", box(40, 122, 62, 130)),
            key_text(4, "MESE", box(66, 122, 84, 130)),
        ],
    )
    _, fields = chosen(page)
    assert ("field_key", [0], "Manitoba tax (line 23)") in fields
    assert ("field_key", [1], "GIORNO") in fields
    assert ("field_key", [2], "MESE") in fields


def test_close_call_follows_the_side_of_aligned_sibling_options():
    # The first option has a slightly closer key on its left, but its
    # sibling options below all read their keys on the right.
    page = snapshot(
        [
            widget(0, box(100, 50, 108, 58), checkbox=True),
            widget(1, box(100, 70, 108, 78), checkbox=True),
            widget(2, box(100, 90, 108, 98), checkbox=True),
        ],
        [
            key_text(1, "Applicant", box(50, 50, 98, 58)),
            key_text(2, "Type 1A", box(114, 50, 150, 58)),
            key_text(3, "Type 1B", box(114, 70, 150, 78)),
            key_text(4, "Type 2", box(114, 90, 150, 98)),
        ],
    )
    _, fields = chosen(page)
    assert ("option_key", [0], "Type 1A") in fields
    assert ("option_key", [1], "Type 1B") in fields
    assert ("option_key", [2], "Type 2") in fields


def test_layout_child_repeated_at_top_level_is_not_consumed_twice():
    key = key_text(1, "Name", box(10, 60, 45, 70))
    container = Cluster(
        id=2, label=DocItemLabel.FORM, bbox=box(0, 40, 200, 100), children=[key]
    )
    # The second box is not a like-sized sibling, so it may only take "Name"
    # if the repeated child produced a second copy of the key.
    page = snapshot(
        [widget(0, box(50, 60, 80, 70)), widget(1, box(90, 60, 160, 70))],
        [key, container],
    )
    _, fields = chosen(page)
    assert fields == [("field_key", [0], "Name")]


def test_option_run_after_a_prompt_reads_keys_on_the_right():
    # "Sexo: (o) M (o) F": each key sits nearer the option before it, but
    # more than a text line away. Texts at both ends of the run leave one
    # over; the inner key leans on the option before it, so the first text
    # is the shared prompt and every option takes the key after it.
    page = snapshot(
        [
            widget(0, box(74, 50, 82, 58), checkbox=True),
            widget(1, box(112, 50, 120, 58), checkbox=True),
        ],
        [
            key_text(1, "Sexo:", box(40, 50, 72, 58)),
            key_text(2, "M", box(91, 50, 99, 58)),
            key_text(3, "F", box(129, 50, 137, 58)),
        ],
    )
    _, fields = chosen(page)
    assert ("option_key", [0], "M") in fields
    assert ("option_key", [1], "F") in fields


def test_run_starting_with_a_key_reads_keys_on_the_left():
    # "Jua (o) Ii (o)": nothing after the last option, so each option takes the
    # key before it, even where the next key is as close.
    page = snapshot(
        [
            widget(0, box(62, 50, 70, 58), checkbox=True),
            widget(1, box(88, 50, 96, 58), checkbox=True),
        ],
        [
            key_text(1, "Jua", box(40, 50, 60, 58)),
            key_text(2, "Ii", box(72, 50, 80, 58)),
        ],
    )
    _, fields = chosen(page)
    assert ("option_key", [0], "Jua") in fields
    assert ("option_key", [1], "Ii") in fields


def test_text_boxes_between_keys_keep_their_leading_keys():
    # "Name [__] Date [__] Place": the inner key leans on the box after
    # it, so the trailing text starts the next field and keys come first.
    page = snapshot(
        [widget(0, box(42, 50, 100, 58)), widget(1, box(127, 50, 180, 58))],
        [
            key_text(1, "Name", box(10, 50, 40, 58)),
            key_text(2, "Date", box(105, 50, 125, 58)),
            key_text(3, "Place", box(185, 50, 210, 58)),
        ],
    )
    _, fields = chosen(page)
    assert ("field_key", [0], "Name") in fields
    assert ("field_key", [1], "Date") in fields


def test_stacked_boxes_read_the_key_above_when_the_next_one_touches_below():
    # Key above each box; the next box's key touches the first box
    # from below. The column run starts with a key: keys are above.
    page = snapshot(
        [widget(0, box(20, 24, 120, 38)), widget(1, box(20, 54, 120, 68))],
        [
            key_text(1, "Name", box(20, 10, 50, 18)),
            key_text(2, "Address", box(20, 40, 60, 48)),
        ],
    )
    _, fields = chosen(page)
    assert ("field_key", [0], "Name") in fields
    assert ("field_key", [1], "Address") in fields


def test_repeated_options_read_their_keys_from_one_side():
    # The first option has the list's shared prompt right above it, in its own
    # cell; the identical options below all read their keys on the
    # right. The structure reads one side: the first option follows the rest.
    page = snapshot(
        [
            widget(0, box(100, 50, 108, 58), checkbox=True),
            widget(1, box(100, 70, 108, 78), checkbox=True),
            widget(2, box(100, 90, 108, 98), checkbox=True),
        ],
        [
            key_text(1, "Types:", box(100, 40, 140, 48)),
            key_text(2, "Type 1A", box(114, 50, 150, 58)),
            key_text(3, "Type 1B", box(114, 70, 150, 78)),
            key_text(4, "Type 2", box(114, 90, 150, 98)),
        ],
    )
    _, fields = chosen(page)
    assert ("option_key", [0], "Type 1A") in fields
    assert ("option_key", [1], "Type 1B") in fields
    assert ("option_key", [2], "Type 2") in fields


def test_key_touching_both_boxes_does_not_decide_a_run():
    # Keys below their boxes, each touching the next box too: a text that
    # touches both neighbours says nothing about which it keys, so the run
    # decides nothing and each box keeps the key right under it.
    page = snapshot(
        [widget(0, box(20, 16, 120, 30)), widget(1, box(20, 40, 120, 54))],
        [
            key_text(1, "Application", box(20, 0, 80, 8)),
            key_text(2, "UFID", box(20, 32, 40, 40)),
            key_text(3, "Last Name", box(20, 56, 60, 64)),
        ],
    )
    _, fields = chosen(page)
    assert ("field_key", [0], "UFID") in fields
    assert ("field_key", [1], "Last Name") in fields
