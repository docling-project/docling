# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Behavioral checks for the offline experiment, independent of native models."""

import json
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, Size, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell

pytest.importorskip("scipy", minversion="1.9")

from scripts.acroform_keying import (
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
    page = snapshot(
        [widget(0, box(50, 60, 80, 70)), widget(1, box(90, 60, 120, 70))],
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
