"""Structural checks independent of country names and field-label meanings."""

import json
from pathlib import Path

import pytest
from run_experiment import module
from uncertainty import audit_assignment

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
