# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import ai_issue_triage as triage  # noqa: E402

REPO = "docling-project/docling"


@pytest.fixture
def issue() -> dict[str, Any]:
    return {
        "number": 42,
        "title": "PDF conversion loses text",
        "body": "Text is missing on the second page.",
        "state": "open",
        "labels": [{"name": "priority:low"}, {"name": "bug"}],
    }


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, issue: dict[str, Any]):
    comments: list[dict[str, Any]] = []
    writes: list[tuple[str, str, dict[str, Any]]] = []

    def read(path: str, **_: object) -> object:
        if path.endswith("/comments?per_page=100"):
            return comments
        assert path == f"repos/{REPO}/issues/42"
        return issue

    def write(method: str, path: str, payload: dict[str, Any]) -> None:
        writes.append((method, path, payload))
        if path.endswith("/comments"):
            comments.append({"user": {"login": triage.BOT_LOGIN}, **payload})

    monkeypatch.setattr(triage, "gh_api", read)
    monkeypatch.setattr(triage, "gh_write", write)
    monkeypatch.setattr(
        triage,
        "ensure_classification_labels",
        lambda _: {"priority:high", "priority:low", "bug", "enhancement", "pdf"},
    )
    return comments, writes


def result_file(tmp_path: Path) -> Path:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {"severity": "high", "application": "enhancement", "formats": ["pdf"]}
        ),
        encoding="utf-8",
    )
    return path


def test_publish_preserves_groups_and_does_not_reapply_removed_labels(
    tmp_path: Path,
    api,
    issue: dict[str, Any],
) -> None:
    comments, writes = api
    assert triage.prepare(REPO, 42, tmp_path)
    context = tmp_path / "context.json"
    result = result_file(tmp_path)
    triage.publish(REPO, 42, context, result)
    assert writes[0] == ("POST", f"repos/{REPO}/issues/42/labels", {"labels": ["pdf"]})
    assert len(comments) == 1
    assert triage.COMMENT_MARKER in comments[0]["body"]
    # The maintainer removes every label. Neither a new event nor an old
    # artifact is allowed to restore them.
    issue["labels"] = []
    assert not triage.prepare(REPO, 42, tmp_path)
    triage.publish(REPO, 42, context, result)
    assert len(writes) == 2


def test_unclassified_issue_receives_priority_application_and_format(
    tmp_path: Path,
    api,
    issue: dict[str, Any],
) -> None:
    issue["labels"] = []
    assert triage.prepare(REPO, 42, tmp_path)
    triage.publish(REPO, 42, tmp_path / "context.json", result_file(tmp_path))
    assert api[1][0] == (
        "POST",
        f"repos/{REPO}/issues/42/labels",
        {"labels": ["enhancement", "priority:high", "pdf"]},
    )


@pytest.mark.parametrize(
    "change", [{"body": "Updated report"}, {"state": "closed"}, {"pull_request": {}}]
)
def test_changed_closed_or_pr_issue_discards_analysis(
    tmp_path: Path,
    api,
    issue: dict[str, Any],
    change: dict[str, Any],
) -> None:
    assert triage.prepare(REPO, 42, tmp_path)
    issue.update(change)
    triage.publish(REPO, 42, tmp_path / "context.json", result_file(tmp_path))
    assert api[1] == []


def test_publish_revalidates_artifact_before_any_write(tmp_path: Path, api) -> None:
    assert triage.prepare(REPO, 42, tmp_path)
    result = result_file(tmp_path)
    result.write_text(
        '{"severity": "high", "application": "bug", "formats": ["tests:full"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="formats"):
        triage.publish(REPO, 42, tmp_path / "context.json", result)
    assert api[1] == []


def test_marker_from_reporter_does_not_block_classification(
    tmp_path: Path, api
) -> None:
    api[0].append({"user": {"login": "reporter"}, "body": triage.COMMENT_MARKER})
    assert triage.prepare(REPO, 42, tmp_path)


@pytest.mark.parametrize(
    ("association", "command", "is_pr", "expected"),
    [
        ("MEMBER", "/ai triage", False, 42),
        ("OWNER", "/ai triage", False, 42),
        ("CONTRIBUTOR", "/ai triage", False, None),
        ("MEMBER", "/ai review", False, None),
        ("MEMBER", "/ai triage", True, None),
    ],
)
def test_issue_commands_require_maintainer_and_ignore_prs(
    issue: dict[str, Any],
    association: str,
    command: str,
    is_pr: bool,
    expected: int | None,
) -> None:
    if is_pr:
        issue["pull_request"] = {}
    event = {
        "issue": issue,
        "comment": {"author_association": association, "body": command},
    }
    assert triage.event_issue("issue_comment", event) == expected
