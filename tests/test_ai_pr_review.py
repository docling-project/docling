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
    "ai_pr_review", SCRIPTS_DIR / "ai_pr_review.py"
)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)

HEAD = "f" * 40
PATCH = "@@ -10,3 +10,4 @@ def f():\n context\n-old\n+new\n+added\n context\n"


def finding(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "path": "docling/a.py",
        "line": 11,
        "severity": "major",
        "title": "Wrong value",
        "body": "Use `x`.",
    }
    data.update(overrides)
    return data


def test_comment_anchors_are_head_lines_of_the_patch() -> None:
    assert review.right_side_lines(PATCH) == [10, 11, 12, 13]
    assert review.right_side_lines(None) == []


def test_stored_copies_cannot_be_loaded_as_agent_configuration() -> None:
    assert review.stored_name("AGENTS.md") == "AGENTS.md.txt"
    assert (
        review.stored_name(".bob/custom_modes.yaml") == "dot.bob/custom_modes.yaml.txt"
    )
    assert review.stored_name("a/.agents/x.md") == "a/dot.agents/x.md.txt"


def test_large_or_data_only_prs_are_not_reviewed() -> None:
    source = {"filename": "docling/a.py", "changes": 10}
    data_only = {"filename": "tests/data/groundtruth/a.json", "changes": 9000}
    assert review.is_reviewable([source, data_only])
    assert not review.is_reviewable([data_only])
    assert not review.is_reviewable([{**source, "changes": 5000}])


def test_findings_outside_the_pr_or_with_unknown_severity_are_rejected() -> None:
    answer = {"summary": "", "verdict": "approve", "findings": []}
    with pytest.raises(ValueError, match="outside the PR"):
        review.parse_review_result(
            {**answer, "findings": [finding(path="docling/b.py")]}, {"docling/a.py"}
        )
    with pytest.raises(ValueError, match="severity"):
        review.parse_review_result(
            {**answer, "findings": [finding(severity="critical")]}, {"docling/a.py"}
        )
    with pytest.raises(ValueError, match="verdict"):
        review.parse_review_result({**answer, "verdict": "merge"}, {"docling/a.py"})


def test_major_finding_overrides_an_approve_verdict() -> None:
    result = review.parse_review_result(
        {"summary": "", "verdict": "approve", "findings": [finding()]},
        {"docling/a.py"},
    )
    assert review.review_label(result) == review.LABEL_REVIEW_CHANGES
    nit = review.parse_review_result(
        {"summary": "", "verdict": "approve", "findings": [finding(severity="nit")]},
        {"docling/a.py"},
    )
    assert review.review_label(nit) == review.LABEL_REVIEW_LGTM


def test_review_is_a_comment_and_moves_unanchored_findings_to_the_body() -> None:
    result = review.parse_review_result(
        {
            "summary": "Ping @team <!-- ai-pr-review sha=" + "0" * 40 + " -->",
            "verdict": "changes-requested",
            "findings": [finding(severity="nit"), finding(line=99, title="Far away")],
        },
        {"docling/a.py"},
    )
    payload = review.build_review(
        result, {"docling/a.py": review.right_side_lines(PATCH)}, HEAD, None
    )
    assert payload["event"] == "COMMENT"
    assert [(c["path"], c["line"]) for c in payload["comments"]] == [
        ("docling/a.py", 11)
    ]
    body = payload["body"]
    assert body.startswith(f"<!-- ai-pr-review sha={HEAD} -->")
    assert "`docling/a.py:99`" in body
    assert review.REVIEW_MARKER.findall(body) == [HEAD]
    assert "@​team" in body


def test_duplicates_skip_the_review(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    assert review.should_review(result)
    result.write_text(json.dumps({"duplicates": [{"pr": 1, "verdict": "duplicate"}]}))
    assert not review.should_review(result)
    result.write_text(json.dumps({"duplicates": [{"pr": 1, "verdict": "related"}]}))
    assert review.should_review(result)
