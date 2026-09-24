# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Behavioral checks of the AcroForm keying, independent of native models."""

import json
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, Size, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell

pytest.importorskip("scipy", minversion="1.9")

from docling.models.stages.form_field.keying import (
    DetectedTable,
    NativeWidget,
    Region,
    Scope,
    Snapshot,
    Tables,
    assign,
    scope_of,
)
from scripts.replay_acroform_keying import (
    Annotation,
    FieldReview,
    OrderedValue,
    associations,
    evaluate,
    validate_field_reviews,
)


def box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def widget(
    index: int, bbox: BoundingBox, *, checkbox: bool = False, name: str | None = None
) -> NativeWidget:
    return NativeWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        widget_field_type="/Btn" if checkbox else "/Tx",
        widget_field_name=name,
        widget_field_flags=0,
        widget_appearance_state="/Off" if checkbox else None,
        widget_text="/Off" if checkbox else "retained value",
    )


def label(index: int, text: str, bbox: BoundingBox) -> Region:
    return Region(
        id=index,
        label="text",
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


def snapshot(
    widgets: list[NativeWidget], labels: list[Region], tables: Tables | None = None
) -> Snapshot:
    return Snapshot(
        page=1,
        size=Size(width=400, height=400),
        widgets=widgets,
        layout=labels,
        tables=tables or Tables(),
    )


def chosen(page: Snapshot) -> tuple[list[int], list[tuple[str, list[int], str]]]:
    result = assign(page)
    assert result.solver_status == "optimal"
    fields = [
        (
            result.candidates[c].kind,
            [result.values[i].native.index for i in result.candidates[c].members],
            result.labels[result.candidates[c].label].text,
        )
        for c in result.selected
    ]
    return [v.native.index for v in result.values], fields


def test_table_exception_requires_one_cell_and_never_uses_other_cell_header():
    table = Region(id=100, label="table", bbox=box(0, 40, 200, 200))
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
            table,
            label(1, "Local name", box(10, 82, 65, 92)),
            label(2, "Other cell", box(101, 81, 165, 91)),
            label(3, "Outside table", box(80, 25, 180, 35)),
        ],
        Tables(table_map={100: DetectedTable(table_cells=cells)}),
    )
    order, fields = chosen(page)
    assert order == [0, 1]
    assert ("field_key", [0], "Local name") in fields
    assert all(text != "Outside table" for _, _, text in fields)
    assert scope_of(box(95, 80, 105, 90), page) == Scope(100)
    # Even near-total coverage must not move a cross-cell widget into one cell.
    assert scope_of(box(82, 80, 100.1, 95), page) == Scope(100)
    assert scope_of(box(82, 80, 98, 95), page) == Scope(100, 0)

    # A detected table with no usable cells stays excluded, even with an
    # attractive nearby label. Removing detection restores ordinary pairing.
    page.tables = Tables()
    assert chosen(page)[1] == []
    page.layout.remove(table)
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
            label(1, "English", box(25, 40, 90, 50)),
            label(2, "Spanish", box(25, 80, 90, 90)),
            label(3, "French", box(215, 40, 280, 50)),
            label(4, "Arabic", box(215, 80, 280, 90)),
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


def test_shared_question_keeps_captions_and_interleaved_native_values():
    page = snapshot(
        [
            widget(0, box(10, 50, 20, 60), checkbox=True, name="same native field"),
            widget(1, box(220, 120, 300, 140)),
            widget(2, box(10, 90, 20, 100), checkbox=True, name="same native field"),
        ],
        [
            label(1, "Preferred language", box(10, 15, 140, 25)),
            label(2, "English", box(25, 50, 90, 60)),
            label(3, "French", box(25, 90, 90, 100)),
            label(4, "Your name", box(220, 103, 290, 113)),
        ],
    )
    order, fields = chosen(page)
    assert order == [0, 1, 2]
    assert ("choice_group", [0, 2], "Preferred language") in fields
    assert ("option_caption", [0], "English") in fields
    assert ("option_caption", [2], "French") in fields
    assert ("field_key", [1], "Your name") in fields
    exported, _ = associations(assign(page))
    restored = [OrderedValue.model_validate_json(v.model_dump_json()) for v in exported]
    assert [v.native.model_dump() for v in restored] == [
        v.model_dump() for v in page.widgets
    ]
    assert len(set(restored[0].field_refs) & set(restored[2].field_refs)) == 1
    # Snapshot and output round-trips both retain the interleaved native order.
    assert chosen(Snapshot.model_validate_json(page.model_dump_json())) == (
        order,
        fields,
    )


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
        [label(1, text, positioned(5, 55, 250, 68))],
    )
    page.size = Size(width=400 * scale + offset, height=400 * scale + offset)
    order, fields = chosen(page)
    assert order == [7, 2, 9]
    assert fields == [("inline_clause", [7, 2], text)]


def test_shared_business_number_is_not_split_at_printed_component():
    page = snapshot(
        [
            widget(0, box(40, 70, 150, 85)),
            widget(1, box(168, 70, 230, 85)),
        ],
        [
            label(1, "Business Number", box(40, 40, 140, 50)),
            label(2, "RT", box(152, 72, 166, 82)),
        ],
    )
    _, fields = chosen(page)
    assert fields == [("composite_field", [0, 1], "Business Number")]


def test_detected_table_keys_values_by_row_caption_with_column_context():
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
        [Region(id=100, label="table", bbox=box(5, 35, 305, 100))],
        Tables(table_map={100: DetectedTable(table_cells=cells)}),
    )
    result = assign(page)
    keys = {
        result.values[c.members[0]].native.index: (
            result.labels[c.label].text,
            result.labels[c.context].text if c.context is not None else None,
        )
        for c in (result.candidates[i] for i in result.selected)
        if c.kind == "table_cell"
    }
    assert keys == {
        0: ("Financial services", "From head office"),
        1: ("Financial services", "From third parties"),
        2: ("Taxable goods", "From head office"),
        3: ("Taxable goods", "From third parties"),
    }


def test_row_caption_is_shared_by_like_sized_values_but_not_operand_boxes():
    page = snapshot(
        [
            widget(0, box(100, 60, 160, 72)),
            widget(1, box(170, 60, 190, 72)),
            widget(2, box(200, 60, 260, 72)),
        ],
        [label(1, "Line 5 total", box(10, 61, 80, 71))],
    )
    _, fields = chosen(page)
    assert sorted(fields) == [
        ("field_key", [0], "Line 5 total"),
        ("field_key", [2], "Line 5 total"),
    ]


def test_caption_split_per_line_is_joined_but_option_lines_are_not():
    page = snapshot(
        [
            widget(0, box(150, 72, 220, 84)),
            widget(1, box(12, 221, 20, 229), checkbox=True),
            widget(2, box(12, 231, 20, 239), checkbox=True),
        ],
        [
            label(1, "Intangible personal", box(10, 60, 100, 68)),
            label(2, "property and services", box(10, 69, 110, 77)),
            label(3, "financial services", box(10, 78, 95, 86)),
            label(4, "a Parent group", box(10, 220, 90, 230)),
            label(5, "b Brother group", box(10, 230, 95, 240)),
        ],
    )
    _, fields = chosen(page)
    assert (
        "field_key",
        [0],
        "Intangible personal property and services financial services",
    ) in fields
    assert ("option_caption", [1], "a Parent group") in fields
    assert ("option_caption", [2], "b Brother group") in fields


def test_line_split_into_pieces_is_joined_but_neighbouring_cells_are_not():
    page = snapshot(
        [
            widget(0, box(250, 60, 320, 72)),
            widget(1, box(40, 131, 60, 143)),
            widget(2, box(66, 131, 86, 143)),
        ],
        [
            label(1, "Manitoba", box(10, 61, 45, 69)),
            label(2, "tax (line 23)", box(47, 61, 100, 69)),
            # Sub-captions over their own boxes: separate cells, never one line.
            label(3, "GIORNO", box(40, 122, 62, 130)),
            label(4, "MESE", box(66, 122, 84, 130)),
        ],
    )
    _, fields = chosen(page)
    assert ("field_key", [0], "Manitoba tax (line 23)") in fields
    assert ("field_key", [1], "GIORNO") in fields
    assert ("field_key", [2], "MESE") in fields


def test_close_call_follows_the_side_of_aligned_sibling_options():
    # The first option has a slightly closer caption on its left, but its
    # sibling options below all read their captions on the right.
    page = snapshot(
        [
            widget(0, box(100, 50, 108, 58), checkbox=True),
            widget(1, box(100, 70, 108, 78), checkbox=True),
            widget(2, box(100, 90, 108, 98), checkbox=True),
        ],
        [
            label(1, "Applicant", box(50, 50, 98, 58)),
            label(2, "Type 1A", box(114, 50, 150, 58)),
            label(3, "Type 1B", box(114, 70, 150, 78)),
            label(4, "Type 2", box(114, 90, 150, 98)),
        ],
    )
    _, fields = chosen(page)
    assert ("option_caption", [0], "Type 1A") in fields
    assert ("option_caption", [1], "Type 1B") in fields
    assert ("option_caption", [2], "Type 2") in fields


def test_duplicate_widget_identity_is_rejected():
    page = snapshot(
        [widget(1, box(10, 10, 20, 20)), widget(1, box(30, 10, 40, 20))], []
    )
    with pytest.raises(ValueError, match="Duplicate native widget"):
        assign(page)


def test_layout_child_repeated_at_top_level_is_not_consumed_twice():
    caption = label(1, "Name", box(10, 60, 45, 70))
    container = Region(
        id=2, label="form", bbox=box(0, 40, 200, 100), children=[caption]
    )
    # The second box is not a like-sized sibling, so it may only take "Name"
    # if the repeated child produced a second copy of the caption.
    page = snapshot(
        [widget(0, box(50, 60, 80, 70)), widget(1, box(90, 60, 160, 70))],
        [caption, container],
    )
    _, fields = chosen(page)
    assert fields == [("field_key", [0], "Name")]


def test_real_language_form_recovers_all_captions_and_one_question():
    data = Path(__file__).parent / "data/acroform_keying_replay"
    page = Snapshot.model_validate_json(
        (data / "f1040lep.json").read_text(encoding="utf-8")
    )
    truth = Path(__file__).parent / "data/groundtruth/acroform_keying/annotations.jsonl"
    expected = [
        json.loads(line) for line in truth.read_text(encoding="utf-8").splitlines()
    ]
    annotations = [
        Annotation.model_validate(a)
        for a in expected
        if a["fixture"] == "usa_cluster011_partial_page_prefilled__f1040lep"
    ]
    result = assign(page)
    reviews, groups = evaluate(page, result, annotations)
    assert result.solver_status == "optimal"
    assert [v.native.index for v in result.values] == list(range(23))
    assert all(r.status == "correct" for r in reviews)
    assert groups == {"correct": 1}


def test_review_distinguishes_table_exclusion_keyless_and_ambiguous():
    page = snapshot(
        [
            widget(0, box(50, 30, 80, 40)),
            widget(1, box(50, 90, 80, 100)),
            widget(2, box(50, 150, 80, 160)),
        ],
        [
            label(1, "Name", box(10, 30, 45, 40)),
            Region(id=20, label="table", bbox=box(0, 130, 100, 200)),
            Region(id=30, label="form", bbox=box(0, 0, 100, 200)),
        ],
    )
    result = assign(page)
    references = [
        FieldReview(
            fixture="test",
            page=1,
            widget_index=i,
            disposition=disposition,
            reason="Explicit visual judgment",
            detected_form_ids=[30],
            table=20 if i == 2 else None,
            cell=None,
        )
        for i, disposition in enumerate(
            ["no_visible_label", "ambiguous", "excluded_table"]
        )
    ]
    reviews, _ = evaluate(page, result, [], references)
    assert [r.status for r in reviews] == [
        "wrong: expected no label",
        "ambiguous",
        "excluded by table rule",
    ]
    # Table exclusion must hold even without any semantic annotation.
    historical, _ = evaluate(page, result, [])
    assert historical[2].status == "excluded by table rule"
    # Explicit no-label decisions reward abstention; ambiguous remains unscored.
    result.selected = []
    reviews, _ = evaluate(page, result, [], references)
    assert [r.status for r in reviews] == [
        "correct abstention",
        "ambiguous",
        "excluded by table rule",
    ]
    with pytest.raises(ValueError, match="every retained widget"):
        validate_field_reviews(page, references[:-1])
    with pytest.raises(ValueError, match="every retained widget"):
        validate_field_reviews(page, references + references[:1])
    with pytest.raises(ValueError, match="Stale detection scope"):
        validate_field_reviews(
            page,
            [
                references[0].model_copy(update={"detected_form_ids": []}),
                *references[1:],
            ],
        )


def test_complete_visual_references_cover_real_language_form():
    data = Path(__file__).parent / "data"
    page = Snapshot.model_validate_json(
        (data / "acroform_keying_replay/f1040lep.json").read_text(encoding="utf-8")
    )
    references = [
        FieldReview.model_validate_json(line)
        for line in (data / "groundtruth/acroform_keying/field_reviews.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    references = [
        r
        for r in references
        if r.fixture == "usa_cluster011_partial_page_prefilled__f1040lep"
    ]
    reviews, _ = evaluate(page, assign(page), [], references)
    assert len(reviews) == 23
    assert all(r.status == "correct" for r in reviews)
