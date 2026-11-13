# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Prepare, validate, and publish advisory issue classifications.

Reuse the PR model runner, output parser, GitHub helpers, and format labels.
Only the publish command needs a GitHub write token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_pr_event import MAINTAINER_ASSOCIATIONS, parse_command
from ai_pr_triage import (
    BOT_LOGIN,
    answer_from_output,
    ensure_classification_labels,
    gh_api,
    gh_write,
    read_bob_output,
)
from pr_topic_labels import FORMAT_LABELS
from triage_classification import classification_labels, parse_classification

COMMENT_MARKER = "<!-- ai-issue-classification -->"
MAX_BODY_CHARS = 12_000


@dataclass(slots=True)
class IssueContext:
    number: int
    title: str
    body: str
    fingerprint: str
    format_labels: list[str]


def issue_fingerprint(issue: dict[str, Any]) -> str:
    """Bind the analysis to the full title and body, even when input is truncated."""
    content = json.dumps([issue["title"], issue.get("body") or ""])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def has_classification(repo: str, number: int) -> bool:
    comments = gh_api(
        f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True
    )
    return any(
        comment["user"]["login"] == BOT_LOGIN
        and COMMENT_MARKER in (comment.get("body") or "")
        for comment in comments
    )


def event_issue(event_name: str, event: dict[str, Any]) -> int | None:
    """Ignore PR comments and commands from users who are not maintainers."""
    if event_name == "workflow_dispatch":
        number = int(event["inputs"]["issue"])
        if number < 1:
            raise ValueError("issue must be a positive number")
        return number
    issue = event["issue"]
    if "pull_request" in issue or issue.get("state") != "open":
        return None
    if event_name == "issue_comment":
        comment = event["comment"]
        if (
            comment.get("author_association") not in MAINTAINER_ASSOCIATIONS
            or parse_command(comment.get("body") or "") != "triage"
        ):
            return None
    elif event_name != "issues":
        raise ValueError(f"unsupported event: {event_name}")
    return int(issue["number"])


def prepare(repo: str, number: int, out_dir: Path) -> bool:
    issue = gh_api(f"repos/{repo}/issues/{number}")
    if "pull_request" in issue or issue["state"] != "open":
        return False
    if has_classification(repo, number):
        print("The issue was classified. Maintainer label changes are preserved.")
        return False
    context = IssueContext(
        number=number,
        title=issue["title"],
        body=(issue.get("body") or "")[:MAX_BODY_CHARS],
        fingerprint=issue_fingerprint(issue),
        format_labels=list(FORMAT_LABELS),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "context.json").write_text(
        json.dumps(asdict(context), indent=2), encoding="utf-8"
    )
    return True


def extract(bob_output: Path, out: Path) -> None:
    answer = answer_from_output(read_bob_output(bob_output), ("classification",))
    classification = parse_classification(answer["classification"])
    out.write_text(json.dumps(asdict(classification), indent=2), encoding="utf-8")


def publish(repo: str, number: int, context_path: Path, result_path: Path) -> None:
    context = IssueContext(**json.loads(context_path.read_text(encoding="utf-8")))
    classification = parse_classification(
        json.loads(result_path.read_text(encoding="utf-8"))
    )
    issue = gh_api(f"repos/{repo}/issues/{number}")
    if (
        context.number != number
        or "pull_request" in issue
        or issue["state"] != "open"
        or issue_fingerprint(issue) != context.fingerprint
    ):
        print("The issue changed or closed. The classification is discarded.")
        return
    if has_classification(repo, number):
        return
    labels = classification_labels(
        classification,
        [],
        {label["name"] for label in issue["labels"]},
        ensure_classification_labels(repo),
    )
    if labels:
        gh_write("POST", f"repos/{repo}/issues/{number}/labels", {"labels": labels})
    names = ", ".join(f"`{label}`" for label in labels)
    body = (
        f"{COMMENT_MARKER}\n"
        + (
            f"Classification labels added: {names}."
            if labels
            else "No classification labels added."
        )
        + "\n\nThese labels are advisory. Maintainers can correct them. "
        "The workflow preserves these corrections."
    )
    gh_write("POST", f"repos/{repo}/issues/{number}/comments", {"body": body})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", required=True)
    prep.add_argument("--event-name", required=True)
    prep.add_argument("--event-path", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--bob-output", type=Path, required=True)
    ext.add_argument("--out", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--repo", required=True)
    pub.add_argument("--issue", type=int, required=True)
    pub.add_argument("--context", type=Path, required=True)
    pub.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        number = event_issue(args.event_name, event)
        run = number is not None and prepare(args.repo, number, args.out)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(
                    f"run={'true' if run else 'false'}\nissue={number or ''}\n"
                )
    elif args.command == "extract":
        extract(args.bob_output, args.out)
    elif args.command == "publish":
        publish(args.repo, args.issue, args.context, args.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
