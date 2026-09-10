"""Reproduce question-role failures without changing the experimental optimizer.

Run from the repository root with PYTHONPATH=.:output/acroform-keying-grouping-20260910.
Fixture IDs below select diagnostic cases; they never enter proposal/scoring code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from run_experiment import module as m


def force_group(snapshot, result, index):
    """Require an existing candidate, keeping every original objective term."""
    active = [
        i
        for i, c in enumerate(result.candidates)
        if c.cost < (0 if c.kind == "choice_group" else 3 * len(c.members))
    ]
    column = active.index(index)
    original = m.milp

    def solve(costs, **kwargs):
        lower = np.zeros(len(costs))
        lower[column] = 1
        kwargs["bounds"] = m.Bounds(lower, np.ones(len(costs)))
        return original(costs, **kwargs)

    with patch.object(m, "milp", solve):
        forced = m.assign(snapshot)
    assert forced.solver_status == "optimal" and index in forced.selected
    assert forced.objective >= result.objective - 1e-8
    return forced


def groups(result):
    return [
        {
            "candidate": i,
            "native_members": [result.values[j].native.index for j in c.members],
            "question": result.labels[c.label].text,
            "question_bbox": result.labels[c.label].bbox.model_dump(mode="json"),
            "question_atoms": sorted(result.labels[c.label].atoms),
            "features": c.features,
            "captions": [result.labels[j].text for j in c.option_labels],
        }
        for i in result.selected
        if (c := result.candidates[i]).kind == "choice_group"
    ]


def main() -> None:
    folder = Path(__file__).parent
    source = folder / "experimental_algorithm.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshots = sorted(
        Path("output/acroform-keying-review-20260908/snapshots").glob("*/*.json")
    )
    evidence = {
        "experimental_source_sha256": digest,
        "text_invariance_pages": [],
        "cases": {},
    }
    for path in snapshots:
        s = m.Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
        values, labels, h = m.inputs(s)
        proposals = m.choice_configurations(values, labels, h)
        # Vary words, alphabet, punctuation, and length, preserving geometry/atoms.
        assert proposals == m.choice_configurations(
            values, [replace(x, text="?") for x in labels], h
        )
        slug = f"{path.parent.name}-p{s.page}"
        evidence["text_invariance_pages"].append(slug)
        if s.page != 1 or not any(
            key in slug for key in ("f1040lep", "f1120so", "gst111", "rf-1125s")
        ):
            continue
        result = m.assign(s)
        assert result.solver_status == "optimal"
        case = {
            "h": h,
            "objective": result.objective,
            "selected_groups": groups(result),
        }
        if "f1040lep" in slug:
            full = next(
                i
                for i, c in enumerate(result.candidates)
                if c.kind == "choice_group"
                and c.label == 11
                and c.members == tuple(range(2, 23))
            )
            forced = force_group(s, result, full)
            case.update(
                forced_full_group_objective=forced.objective,
                forced_full_group_gap=forced.objective - result.objective,
                forced_full_groups=groups(forced),
            )
            arabic = next(
                g
                for g in case["selected_groups"]
                if g["native_members"] == [8, 9, 10, 11, 12]
            )
            q = result.labels[result.candidates[arabic["candidate"]].label]
            case["arabic_gap_from_own_checkbox_in_h"] = (
                q.bbox.l - values[7].bbox.r
            ) / h
        if "f1120so" in slug:
            assert not any(c.members == (28, 37) for c in proposals)
            child = next(
                i
                for i in result.selected
                if result.candidates[i].kind == "choice_group"
                and result.candidates[i].members == (29, 32)
            )
            c = result.candidates[child]
            assert labels[c.label].atoms & labels[123].atoms
            case["outer_yes_no_candidate_present"] = False
            case["child_question_conflicts_with_yes_caption_atoms"] = sorted(
                labels[c.label].atoms & labels[123].atoms
            )
            without = m.assign(
                s,
                forbidden_groups=frozenset(
                    i
                    for i, x in enumerate(result.candidates)
                    if x.kind == "choice_group"
                    and labels[x.label].atoms & labels[123].atoms
                ),
            )
            assert without.solver_status == "optimal"
            case["yes_caption_without_conflicting_child_question"] = [
                without.labels[without.candidates[i].label].text
                for i in without.selected
                if 28 in without.candidates[i].members
                and without.candidates[i].kind != "choice_group"
            ]
        if "gst111" in slug:
            assert not any(c.label == 37 for c in proposals)
            first = next(v for v in values if v.native.index == 15)
            case["true_question_candidate_present"] = False
            case["true_question_gap_in_h"] = (first.bbox.t - labels[37].bbox.b) / h
            case["first_caption_gap_in_h"] = (first.bbox.l - labels[50].bbox.r) / h
        if "rf-1125s" in slug:
            case["left_caption_gaps_in_h"] = [
                (values[j].bbox.l - labels[k].bbox.r) / h
                for j, k in [(10, 91), (11, 114), (12, 92)]
            ]
            actual = next(
                i
                for i, c in enumerate(result.candidates)
                if c.kind == "choice_group"
                and c.label == 0
                and c.members == (10, 11, 12)
            )
            forced = force_group(s, result, actual)
            case["forced_actual_question_gap"] = forced.objective - result.objective
            case["forced_actual_question_groups"] = groups(forced)
        evidence["cases"][slug] = case
        print(slug, flush=True)
    assert len(snapshots) == 19
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert (
        Path("scripts/acroform_keying.py").read_bytes()
        == (folder / "baseline_algorithm.py").read_bytes()
    )
    (folder / "role-investigation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "All 19 proposal sets unchanged by text replacement; diagnostic assertions passed."
    )


if __name__ == "__main__":
    main()
