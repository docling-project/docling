# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "pr_approval_reminder",
    Path(__file__).resolve().parents[1] / ".github/scripts/pr_approval_reminder.py",
)
assert SPEC is not None and SPEC.loader is not None
reminder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reminder
SPEC.loader.exec_module(reminder)

NOW = datetime(2026, 10, 8, 6, 0, tzinfo=timezone.utc)


def run(sha: str, name: str, created: str, **overrides: Any) -> dict[str, Any]:
    data = {
        "head_sha": sha,
        "name": name,
        "event": "pull_request",
        "conclusion": "action_required",
        "created_at": created,
    }
    data.update(overrides)
    return data


def pull(number: int, sha: str, labels: tuple[str, ...] = (), **overrides: Any):
    data = {
        "number": number,
        "title": f"fix: change {number}",
        "html_url": f"https://github.com/o/r/pull/{number}",
        "user": {"login": f"user{number}"},
        "head": {"sha": sha},
        "draft": False,
        "labels": [{"name": label} for label in labels],
    }
    data.update(overrides)
    return data


def test_only_current_head_commits_of_ready_prs_count() -> None:
    runs = [
        run("aaa", "Run CI", "2026-10-06T06:00:00Z"),
        run("aaa", "Run Docs CI", "2026-10-05T06:00:00Z"),
        run("old", "Run CI", "2026-10-01T06:00:00Z"),
        run("bbb", "Run CI", "2026-10-07T06:00:00Z", conclusion="success"),
        run("ccc", "Run CI", "2026-10-07T06:00:00Z"),
    ]
    pulls = [
        pull(1, "aaa", ("ai:ci-safe", "docx")),
        pull(2, "new"),  # its waiting run belongs to an older commit
        pull(3, "bbb"),
        pull(4, "ccc", draft=True),
    ]
    prs = reminder.find_waiting_prs(pulls, reminder.waiting_runs_by_sha(runs))
    assert [pr.number for pr in prs] == [1]
    assert prs[0].workflows == ["Run CI", "Run Docs CI"]
    assert prs[0].waiting_since == datetime(2026, 10, 5, 6, tzinfo=timezone.utc)
    assert prs[0].ai_labels == ["ai:ci-safe"]


def test_message_groups_by_triage_label_and_escapes_titles() -> None:
    safe = reminder.WaitingPr(
        1, "fix: a <!channel> & b|c", "u1", "alice", NOW, ["Run CI"], ["ai:ci-safe"]
    )
    care = reminder.WaitingPr(
        2,
        "feat: x",
        "u2",
        "bob",
        datetime(2026, 10, 5, 6, tzinfo=timezone.utc),
        ["Run CI"],
        ["ai:ci-needs-care", "ai:possible-duplicate"],
    )
    plain = reminder.WaitingPr(3, "docs: y", "u3", "carol", NOW, ["Run CI"], [])
    payload = reminder.build_message([care, safe, plain], "o/r", NOW)
    sections = [b["text"]["text"] for b in payload["blocks"] if b["type"] == "section"]
    assert len(sections) == 3
    assert sections[0].startswith(
        "*:white_check_mark: AI triage found no CI concern* (1)"
    )
    assert "&lt;!channel&gt; &amp; b¦c" in sections[0]
    assert sections[1].startswith("*:warning: AI triage asks for care* (1)")
    assert "waiting 3 d · `ai:possible-duplicate`" in sections[1]
    assert sections[2].startswith("*:grey_question: Not triaged* (1)")
    assert payload["text"] == "3 PR(s) in o/r wait for a maintainer to approve CI"


def test_long_lists_are_split_and_capped() -> None:
    prs = [
        reminder.WaitingPr(n, "t" * 90, f"https://x/{n}", "u", NOW, ["CI"], [])
        for n in range(60)
    ]
    payload = reminder.build_message(prs, "o/r", NOW)
    sections = [b for b in payload["blocks"] if b["type"] == "section"]
    assert len(sections) > 1
    assert all(len(b["text"]["text"]) <= reminder.MAX_SECTION_CHARS for b in sections)
    assert "… and 20 more" in payload["blocks"][-2]["elements"][0]["text"]
