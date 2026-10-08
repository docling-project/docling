# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ai_pr_event",
    Path(__file__).resolve().parents[1] / ".github/scripts/ai_pr_event.py",
)
assert SPEC is not None and SPEC.loader is not None
event = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = event
SPEC.loader.exec_module(event)


def comment_event(body: str, association: str = "MEMBER", **issue: Any):
    issue_data = {"number": 7, "state": "open", "pull_request": {}}
    issue_data.update(issue)
    return {
        "issue": issue_data,
        "comment": {"body": body, "author_association": association},
    }


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("/ai", "review"),
        ("/ai review", "review"),
        ("  /ai triage  \nplease", "triage"),
        ("/ai please review", None),
        ("/aireview", None),
        ("please run /ai review", None),
        ("", None),
    ],
)
def test_only_exact_commands_on_the_first_line_count(body, expected) -> None:
    assert event.parse_command(body) == expected


def test_push_events_skip_drafts_and_bots() -> None:
    def push(draft: bool, user_type: str) -> object:
        pull = {"number": 3, "draft": draft, "user": {"type": user_type}}
        return event.resolve("pull_request_target", {"pull_request": pull})

    assert push(False, "User") == event.Decision(3, True, False, "push")
    assert not push(True, "User").run
    assert not push(False, "Bot").run


def test_commands_need_a_maintainer_and_an_open_pr() -> None:
    review = event.resolve("issue_comment", comment_event("/ai review"))
    assert (review.pr, review.run, review.force_review) == (7, True, True)

    triage = event.resolve("issue_comment", comment_event("/ai triage", "OWNER"))
    assert (triage.run, triage.force_review) == (True, False)

    assert not event.resolve("issue_comment", comment_event("/ai", "CONTRIBUTOR")).run
    assert not event.resolve("issue_comment", comment_event("/ai", state="closed")).run
    issue_only = comment_event("/ai")
    del issue_only["issue"]["pull_request"]
    assert not event.resolve("issue_comment", issue_only).run


def test_maintainer_activity_ignores_bots_the_author_and_commands() -> None:
    def item(login: str, association: str, body: str = "Looks off.", kind="User"):
        return {
            "user": {"login": login, "type": kind},
            "author_association": association,
            "body": body,
        }

    assert event.is_maintainer_activity(item("maint", "MEMBER"), "contrib")
    assert not event.is_maintainer_activity(item("contrib", "CONTRIBUTOR"), "contrib")
    assert not event.is_maintainer_activity(
        item("github-actions[bot]", "NONE", kind="Bot"), "contrib"
    )
    # A maintainer's own PR: their replies are not a review by someone else.
    assert not event.is_maintainer_activity(item("maint", "MEMBER"), "maint")
    assert not event.is_maintainer_activity(
        item("maint", "MEMBER", body="/ai review"), "contrib"
    )
