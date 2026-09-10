"""Structural checks independent of country names and field-label meanings."""

import json
from pathlib import Path

import pytest
from run_experiment import module
from uncertainty import audit_assignment

from scripts.replay_acroform_keying import label_matches
from tests.test_acroform_keying_prototype import box, label, snapshot, widget


def selected(result):
    return [
        (
            result.candidates[j].kind,
            tuple(result.values[i].native.index for i in result.candidates[j].members),
            result.labels[result.candidates[j].label].text,
        )
        for j in result.selected
    ]


@pytest.mark.parametrize("position", ["left", "above"])
@pytest.mark.parametrize("prompt", ["Group heading", "問", "123"])
def test_question_configuration_keeps_option_captions(position, prompt):
    q = box(10, 60, 100, 80) if position == "left" else box(100, 35, 190, 45)
    page = snapshot(
        [
            widget(i, box(105, 60 + 30 * i, 115, 70 + 30 * i), checkbox=True)
            for i in range(3)
        ],
        [label(0, prompt, q)]
        + [
            label(i + 1, f"Option {i}", box(120, 60 + 30 * i, 180, 70 + 30 * i))
            for i in range(3)
        ],
    )
    result = module.assign(page)
    assert ("choice_group", (0, 1, 2), prompt) in selected(result)
    assert all(
        ("option_caption", (i,), f"Option {i}") in selected(result) for i in range(3)
    )


def test_independent_choices_do_not_inherit_a_side_penalty():
    path = Path(
        "output/acroform-keying-coverage-20260910/independent-checkbox-counterexample.json"
    )
    page = module.Snapshot.model_validate(json.loads(path.read_text())["snapshot"])
    result, audit = audit_assignment(page)
    assert selected(result) == [
        ("option_caption", (0,), "AAAA"),
        ("option_caption", (1,), "BBBB"),
    ]
    assert audit["withheld_widgets"] == []


def test_native_interleaving_and_left_hand_captions():
    page = snapshot(
        [
            widget(7, box(200, 60, 210, 70), checkbox=True),
            widget(2, box(300, 200, 350, 210)),
            widget(9, box(200, 100, 210, 110), checkbox=True),
        ],
        [
            label(0, "問", box(130, 35, 250, 45)),
            label(1, "甲", box(140, 60, 195, 70)),
            label(2, "乙", box(140, 100, 195, 110)),
            label(3, "Name", box(300, 180, 350, 190)),
        ],
    )
    result = module.assign(page)
    assert [v.native.index for v in result.values] == [7, 2, 9]
    assert ("choice_group", (7, 9), "問") in selected(result)
    assert ("option_caption", (7,), "甲") in selected(result)
    assert ("option_caption", (9,), "乙") in selected(result)


def test_equal_prompt_interpretations_are_withheld():
    page = snapshot(
        [
            widget(0, box(100, 60, 110, 70), checkbox=True),
            widget(1, box(100, 90, 110, 100), checkbox=True),
        ],
        [
            label(0, "Above", box(110, 40, 200, 60)),
            label(1, "Beside", box(10, 60, 100, 70)),
            label(2, "Alpha", box(115, 60, 170, 70)),
            label(3, "Beta", box(115, 90, 170, 100)),
        ],
    )
    result, audit = audit_assignment(page, margin=1e-6)
    assert any(
        g["uncertain"] and abs(g["objective_gap"]) < 1e-6 for g in audit["groups"]
    )
    assert all(kind != "choice_group" for kind, _, _ in selected(result))


def test_two_side_prompts_define_separate_groups_in_one_column():
    page = snapshot(
        [
            widget(i, box(105, 60 + 30 * i, 115, 70 + 30 * i), checkbox=True)
            for i in range(4)
        ],
        [
            label(0, "First", box(10, 60, 100, 70)),
            label(1, "Second", box(10, 120, 100, 130)),
        ]
        + [
            label(i + 2, f"Option {i}", box(120, 60 + 30 * i, 180, 70 + 30 * i))
            for i in range(4)
        ],
    )
    result = module.assign(page)
    assert {
        (indices, text)
        for kind, indices, text in selected(result)
        if kind == "choice_group"
    } == {((0, 1), "First"), ((2, 3), "Second")}


def test_same_question_can_have_ambiguous_caption_sides():
    page = snapshot(
        [
            widget(i, box(150, 60 + 30 * i, 160, 70 + 30 * i), checkbox=True)
            for i in range(3)
        ],
        [label(0, "Question", box(80, 30, 210, 40))]
        + [
            label(1 + i, "Left", box(100, 60 + 30 * i, 145, 70 + 30 * i))
            for i in range(3)
        ]
        + [
            label(4 + i, "Right", box(165, 60 + 30 * i, 210, 70 + 30 * i))
            for i in range(3)
        ],
    )
    result, audit = audit_assignment(page, margin=1e-6)
    assert audit["withheld_widgets"] == [0, 1, 2]
    assert any(
        row["uncertain"] and abs(row["objective_gap"]) < 1e-6
        for row in audit["caption_checks"]
    )
    assert result.selected == []


def saved_page(fixture):
    path = Path("output/acroform-keying-review-20260908/snapshots") / fixture / "1.json"
    return module.Snapshot.model_validate_json(path.read_text())


def test_language_fragments_do_not_seed_spurious_questions():
    result = module.assign(
        saved_page("usa_cluster011_partial_page_prefilled__f1040lep")
    )
    groups = [
        result.candidates[i]
        for i in result.selected
        if result.candidates[i].kind == "choice_group"
    ]
    assert result.solver_status == "optimal"
    assert len(groups) == 1 and groups[0].members == tuple(range(2, 23))
    assert result.labels[groups[0].label].bbox.t < result.values[2].bbox.t


def test_sami_left_caption_arrangement_is_available_without_distance_cutoff():
    page = saved_page("norway__rf-1125s")
    values, labels, h = module.inputs(page)
    proposals = module.choice_configurations(values, labels, h)
    reviews = [
        json.loads(line)
        for line in Path("tests/data/groundtruth/acroform_keying/field_reviews.jsonl")
        .read_text()
        .splitlines()
    ]
    expected = [
        next(
            r["expected_label"]["bbox"]
            for r in reviews
            if r["fixture"] == "norway__rf-1125s"
            and r["page"] == 1
            and r["widget_index"] == i
        )
        for i in (10, 11, 12)
    ]

    assert any(
        c.members == (10, 11, 12)
        and all(
            any(label_matches(box(*expected[k]), labels[li].bbox) for li in choices)
            for k, choices in enumerate(c.option_choices)
        )
        for c in proposals
    )


def test_nested_yes_caption_is_owned_once_and_referenced_by_children():
    page = saved_page("usa_cluster019_singlecolumn__f1120so")
    result = module.assign(page)
    assert result.solver_status == "optimal"
    assert [v.native.index for v in result.values] == [w.index for w in page.widgets]
    assert any(
        c.kind == "choice_group" and c.members == (28, 37) for c in result.candidates
    )
    child = next(
        result.candidates[j]
        for j in result.selected
        if result.candidates[j].kind == "choice_group"
        and result.candidates[j].members == (29, 32)
    )
    assert child.parent == 28
    owner = next(
        result.candidates[j]
        for j in result.selected
        if result.candidates[j].kind != "choice_group"
        and 28 in result.candidates[j].members
    )
    assert owner.label == child.label
    assert "Yes" in result.labels[owner.label].text


def test_group_reward_can_support_captions_costlier_than_abstention():
    page = snapshot(
        [
            widget(0, box(100, 20, 110, 30), checkbox=True),
            widget(1, box(130, 50, 140, 60), checkbox=True),
            widget(2, box(130, 80, 140, 90), checkbox=True),
        ],
        [
            label(0, "Parent", box(120, 20, 180, 30)),
            label(1, "Alpha", box(150, 50, 200, 60)),
            label(2, "Beta", box(150, 80, 200, 90)),
        ],
    )
    _, labels, _ = module.inputs(page)
    ids = {x.text: i for i, x in enumerate(labels)}
    # Controlled solver counterexample: each child alone costs more than null,
    # but the compatible complete assignment costs 5.2 rather than 6.
    proposed = [
        module.Candidate((0,), ids["Parent"], "option_caption", {"cost": 0}),
        module.Candidate((1,), ids["Alpha"], "option_caption", {"cost": 3.1}),
        module.Candidate((2,), ids["Beta"], "option_caption", {"cost": 3.1}),
        module.Candidate(
            (1, 2),
            ids["Parent"],
            "choice_group",
            {"support": -1},
            ((ids["Alpha"],), (ids["Beta"],)),
            0,
        ),
    ]
    result = module.assign(page, _candidates=proposed)
    assert result.solver_status == "optimal" and result.selected == [0, 1, 2, 3]
    assert result.objective == pytest.approx(5.2)
    ungrouped = module.assign(
        page, _candidates=proposed, forbidden_groups=frozenset({3})
    )
    assert ungrouped.selected == [0] and ungrouped.objective == pytest.approx(6)


@pytest.mark.parametrize("mirror", [False, True])
def test_nested_reference_supports_both_indentation_directions(mirror):
    def rect(left, top, right, bottom):
        return (
            box(300 - right, top, 300 - left, bottom)
            if mirror
            else box(left, top, right, bottom)
        )

    page = snapshot(
        [
            widget(0, rect(100, 60, 110, 70), checkbox=True),
            widget(1, rect(120, 80, 130, 90), checkbox=True),
            widget(2, rect(120, 100, 130, 110), checkbox=True),
            widget(3, rect(100, 130, 110, 140), checkbox=True),
        ],
        [
            label(0, "Question", rect(90, 30, 185, 40)),
            label(1, "Parent", rect(95, 60, 185, 70)),
            label(2, "Alpha", rect(115, 80, 185, 90)),
            label(3, "Beta", rect(115, 100, 185, 110)),
            label(4, "Sibling", rect(95, 130, 185, 140)),
        ],
    )
    result = module.assign(page)
    assert result.solver_status == "optimal"
    assert any(
        result.candidates[j].members == (1, 2) and result.candidates[j].parent == 0
        for j in result.selected
    )
    assert any(
        c.kind == "choice_group" and c.members == (0, 3) for c in result.candidates
    )
