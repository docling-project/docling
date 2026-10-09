# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Resolve which PR the AI triage handles and in which mode.

The workflow starts on ``pull_request_target`` (each push) and on
``issue_comment`` (``/ai`` commands from maintainers). This module turns the
event payload into a small, validated decision, so the jobs never read
untrusted comment text themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAINTAINER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
COMMAND = re.compile(r"^/ai(?:\s+(?P<name>review|triage))?\s*$")


@dataclass(slots=True)
class Decision:
    pr: int
    run: bool
    force_review: bool
    reason: str


def parse_command(body: str) -> str | None:
    """Return 'review' or 'triage' for an /ai command on the first line."""
    lines = body.strip().splitlines()
    match = COMMAND.match(lines[0].strip()) if lines else None
    if match is None:
        return None
    return match.group("name") or "review"


def resolve(event_name: str, event: dict[str, Any]) -> Decision:
    if event_name == "pull_request_target":
        pull = event["pull_request"]
        number = int(pull["number"])
        if pull.get("draft"):
            return Decision(number, False, False, "draft PR")
        if pull["user"]["type"] == "Bot":
            return Decision(number, False, False, "PR from a bot")
        return Decision(number, True, False, "push")

    if event_name == "issue_comment":
        issue = event["issue"]
        number = int(issue["number"])
        comment = event["comment"]
        if "pull_request" not in issue or issue.get("state") != "open":
            return Decision(number, False, False, "not an open PR")
        if comment.get("author_association") not in MAINTAINER_ASSOCIATIONS:
            return Decision(number, False, False, "comment is not from a maintainer")
        command = parse_command(comment.get("body") or "")
        if command is None:
            return Decision(number, False, False, "no /ai command")
        return Decision(number, True, command == "review", f"/ai {command}")

    raise ValueError(f"unsupported event: {event_name}")


def is_maintainer_activity(item: dict[str, Any], pr_author: str) -> bool:
    """True for a comment or review that a maintainer wrote by hand."""
    user = item.get("user") or {}
    body = (item.get("body") or "").strip()
    return (
        item.get("author_association") in MAINTAINER_ASSOCIATIONS
        and user.get("type") != "Bot"
        and user.get("login") != pr_author
        and parse_command(body) is None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--event-path", type=Path, required=True)
    args = parser.parse_args(argv)

    event = json.loads(args.event_path.read_text("utf-8"))
    decision = resolve(args.event_name, event)
    print(
        f"PR #{decision.pr}: run={decision.run},"
        f" force_review={decision.force_review} ({decision.reason})"
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"pr={decision.pr}\n")
            handle.write(f"run={'true' if decision.run else 'false'}\n")
            handle.write(
                f"force_review={'true' if decision.force_review else 'false'}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
