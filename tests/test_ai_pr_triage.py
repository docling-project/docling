# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "ai_pr_triage", SCRIPTS_DIR / "ai_pr_triage.py"
)
assert SPEC is not None and SPEC.loader is not None
triage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = triage
SPEC.loader.exec_module(triage)

REPO = "docling-project/docling"


def make_context(**overrides: object) -> object:
    data = {
        "pr": {
            "number": 10,
            "title": "fix: x",
            "body": "",
            "author": "someone",
            "author_association": "CONTRIBUTOR",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "merge_base_sha": "c" * 40,
            "changed_files": ["docling/x.py"],
            "additions": 1,
            "deletions": 1,
        },
        "issue_refs": [],
        "candidates": [
            {"number": 7, "title": "fix x too", "state": "open", "url": ""},
        ],
        "risk": {"forced": [], "hints": []},
        "groundtruth_markdown": "",
        "diff_truncated": False,
        "topic_labels": [],
    }
    data.update(overrides)
    return triage.TriageContext.from_dict(data)


def answer(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "summary": "The PR fixes x.",
        "duplicates": [{"pr": 7, "verdict": "duplicate", "reason": "Same fix."}],
        "ci_safety": {"verdict": "safe", "concerns": []},
        "groundtruth": None,
        "topics": [],
    }
    data.update(overrides)
    return data


def test_issue_refs_ignore_own_number_other_repos_and_anchors() -> None:
    text = (
        "Fixes #12 and #10. See https://github.com/docling-project/docling/issues/34,"
        " https://github.com/other/repo/issues/56 and docs/page#78. Again #12."
    )
    assert triage.extract_issue_refs(text, REPO, own_number=10) == [12, 34]


def test_sensitive_paths_force_care_and_hints_only_flag_added_lines() -> None:
    files = [
        {"filename": ".github/workflows/ci.yml", "status": "modified"},
        {"filename": "tests/sub/conftest.py", "status": "added"},
        {
            "filename": "docling/new.py",
            "status": "renamed",
            "previous_filename": "AGENTS.md",
        },
        {
            "filename": "docling/backend/x.py",
            "status": "modified",
            "patch": "@@ -1,2 +1,2 @@\n-import subprocess\n+x = eval(data)\n context",
        },
        {
            "filename": "tests/data/groundtruth/doc.md",
            "status": "modified",
            "patch": "@@ -1 +1 @@\n+os.system('rm')",
        },
    ]
    report = triage.assess_risk(files)
    assert [f.path for f in report.forced] == [
        ".github/workflows/ci.yml",
        "tests/sub/conftest.py",
        "AGENTS.md",
    ]
    assert [(h.path, h.kind) for h in report.hints] == [
        ("docling/backend/x.py", "dynamic code execution")
    ]


def test_overlap_needs_intersecting_hunks_or_many_shared_files() -> None:
    ours = [
        {"filename": "docling/a.py", "patch": "@@ -10,5 +10,6 @@\n"},
        {"filename": "uv.lock", "patch": "@@ -1,3 +1,3 @@\n"},
    ]
    near = [{"filename": "docling/a.py", "patch": "@@ -17,2 +17,3 @@\n"}]
    far = [
        {"filename": "docling/a.py", "patch": "@@ -200,2 +200,3 @@\n"},
        {"filename": "uv.lock", "patch": "@@ -1,3 +1,3 @@\n"},
    ]
    assert triage.compare_files(ours, near) == (["docling/a.py"], 1)
    shared, overlaps = triage.compare_files(ours, far)
    assert (shared, overlaps) == (["docling/a.py"], 0)
    assert not triage.is_overlap_candidate(shared, overlaps)


def test_answer_with_unknown_pr_or_verdict_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown PR"):
        triage.parse_triage_result(
            answer(duplicates=[{"pr": 99, "verdict": "duplicate"}]), {7}
        )
    with pytest.raises(ValueError, match="ci_safety"):
        triage.parse_triage_result(
            answer(ci_safety={"verdict": "approve", "concerns": []}), {7}
        )


def test_answer_is_read_from_fenced_final_message() -> None:
    message = "Done.\n```json\n" + json.dumps(answer()) + "\n```"
    result = triage.parse_triage_result(triage.extract_answer(message), {7})
    assert result.ci_verdict == "safe"
    assert result.duplicates[0].verdict == "duplicate"


def test_model_text_cannot_mention_users_or_inject_markup() -> None:
    result = triage.parse_triage_result(
        answer(summary="Ping @maintainers <img src=x> | col\nnext"), {7}
    )
    assert "@​maintainers" in result.summary
    assert "<img" not in result.summary
    assert "\\|" in result.summary
    assert "\n" not in result.summary


def test_deterministic_findings_override_a_safe_model_verdict() -> None:
    context = make_context(
        risk={"forced": [{"path": "uv.lock", "reason": "deps"}], "hints": []}
    )
    result = triage.parse_triage_result(answer(), {7})
    assert triage.decide_labels(context, result) == {
        triage.LABEL_CI_NEEDS_CARE,
        triage.LABEL_DUPLICATE,
    }
    assert triage.decide_labels(make_context(), None) == set()


def test_stored_result_round_trips_through_validation_in_publish() -> None:
    result = triage.parse_triage_result(
        answer(
            groundtruth={"verdict": "expected", "reason": "Matches the fix."},
            topics=["table structure"],
        ),
        {7},
    )
    stored = json.loads(json.dumps(triage.asdict(result)))
    assert triage.parse_triage_result(triage._result_to_answer(stored), {7}) == result


def test_comment_shows_duplicates_collapses_related_and_hides_unrelated() -> None:
    candidates = [
        {"number": n, "title": "", "state": "open", "url": ""} for n in (7, 8, 9)
    ]
    context = make_context(candidates=candidates)
    result = triage.parse_triage_result(
        answer(
            duplicates=[
                {"pr": 7, "verdict": "duplicate", "reason": "Same fix."},
                {"pr": 8, "verdict": "related", "reason": "Same file."},
                {"pr": 9, "verdict": "unrelated", "reason": "No."},
            ]
        ),
        {7, 8, 9},
    )
    comment = triage.render_comment(context, result, {triage.LABEL_CI_SAFE})
    assert comment.startswith(triage.COMMENT_MARKER)
    visible, collapsed = comment.split("<details>")
    assert "| #7 |" in visible
    assert "| #8 |" in collapsed
    assert "#9" not in comment


def test_comment_states_failed_analysis() -> None:
    context = make_context()

    comment = triage.render_comment(context, None, set())
    assert "did not finish" in comment
    assert "| #7 | open |" in comment


def test_model_topics_must_come_from_the_repository_label_set() -> None:
    with pytest.raises(ValueError, match="unknown topic"):
        triage.parse_triage_result(answer(topics=["priority:high"]), {7})
