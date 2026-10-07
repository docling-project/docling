# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Post a Slack summary of open PRs whose CI waits for maintainer approval.

GitHub holds ``pull_request`` runs from new contributors' forks with the
conclusion ``action_required`` until a maintainer approves them. These runs
have an empty ``pull_requests`` list, so they are matched to open PRs by the
head commit. The message groups the PRs by their AI triage label.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

APPROVAL_CONCLUSION = "action_required"
AI_LABEL_PREFIX = "ai:"
MAX_LISTED_PRS = 40
MAX_SECTION_CHARS = 2_900
MAX_TITLE_CHARS = 90
# (heading, label that selects the group); None collects the rest.
GROUPS: tuple[tuple[str, str | None], ...] = (
    (":white_check_mark: AI triage found no CI concern", "ai:ci-safe"),
    (":warning: AI triage asks for care", "ai:ci-needs-care"),
    (":grey_question: Not triaged", None),
)


@dataclass(slots=True)
class WaitingPr:
    number: int
    title: str
    url: str
    author: str
    waiting_since: datetime
    workflows: list[str]
    ai_labels: list[str]


def gh_api(path: str) -> list[dict[str, Any]]:
    command = ["gh", "api", "--paginate", "--slurp", path]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    pages = json.loads(output)
    items: list[dict[str, Any]] = []
    for page in pages:
        # List endpoints return arrays; the runs endpoint wraps them.
        items.extend(page["workflow_runs"] if isinstance(page, dict) else page)
    return items


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def waiting_runs_by_sha(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    waiting: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if (
            run.get("event") == "pull_request"
            and run.get("conclusion") == APPROVAL_CONCLUSION
        ):
            waiting.setdefault(run["head_sha"], []).append(run)
    return waiting


def find_waiting_prs(
    pulls: list[dict[str, Any]], waiting: dict[str, list[dict[str, Any]]]
) -> list[WaitingPr]:
    prs: list[WaitingPr] = []
    for pull in pulls:
        runs = waiting.get(pull["head"]["sha"])
        if not runs or pull.get("draft"):
            continue
        prs.append(
            WaitingPr(
                number=int(pull["number"]),
                title=str(pull["title"]),
                url=str(pull["html_url"]),
                author=str(pull["user"]["login"]),
                waiting_since=min(_parse_time(run["created_at"]) for run in runs),
                workflows=sorted({str(run["name"]) for run in runs}),
                ai_labels=sorted(
                    label["name"]
                    for label in pull.get("labels", [])
                    if label["name"].startswith(AI_LABEL_PREFIX)
                ),
            )
        )
    return sorted(prs, key=lambda pr: pr.waiting_since)


def slack_escape(text: str) -> str:
    """Escape text for Slack mrkdwn, so it cannot form links or mentions."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _age(since: datetime, now: datetime) -> str:
    hours = max(int((now - since).total_seconds() // 3600), 0)
    return f"{hours // 24} d" if hours >= 48 else f"{hours} h"


def format_pr(pr: WaitingPr, now: datetime) -> str:
    title = pr.title if len(pr.title) <= MAX_TITLE_CHARS else pr.title[:89] + "…"
    title = slack_escape(title).replace("|", "¦")
    labels = " ".join(
        f"`{slack_escape(label)}`"
        for label in pr.ai_labels
        if label not in {"ai:ci-safe", "ai:ci-needs-care"}
    )
    details = [
        f"by {slack_escape(pr.author)}",
        f"waiting {_age(pr.waiting_since, now)}",
    ]
    if labels:
        details.append(labels)
    return f"• <{pr.url}|#{pr.number} {title}>\n      {' · '.join(details)}"


def _sections(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    chunk = ""
    for line in lines:
        if chunk and len(chunk) + len(line) + 1 > MAX_SECTION_CHARS:
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
            )
            chunk = ""
        chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
    return blocks


def build_message(prs: list[WaitingPr], repo: str, now: datetime) -> dict[str, Any]:
    count = len(prs)
    summary = f"{count} PR(s) in {repo} wait for a maintainer to approve CI"
    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🚦 {summary}"}},
    ]
    listed = prs[:MAX_LISTED_PRS]
    for heading, label in GROUPS:
        if label is None:
            known = {group_label for _, group_label in GROUPS if group_label}
            members = [pr for pr in listed if not known & set(pr.ai_labels)]
        else:
            members = [pr for pr in listed if label in pr.ai_labels]
        if not members:
            continue
        lines = [f"*{heading}* ({len(members)})"]
        lines.extend(format_pr(pr, now) for pr in members)
        blocks.extend(_sections(lines))
    if count > len(listed):
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"… and {count - len(listed)} more. Oldest first.",
                    }
                ],
            }
        )
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "The `ai:*` labels are advisory. Read the PR before you"
                        f" approve CI. <https://github.com/{repo}/pulls|Open PRs>"
                    ),
                }
            ],
        }
    )
    return {"text": summary, "blocks": blocks}


def post_to_slack(webhook_url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Slack returned HTTP {response.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the message, do not post."
    )
    args = parser.parse_args(argv)

    runs = gh_api(
        f"repos/{args.repo}/actions/runs?status={APPROVAL_CONCLUSION}&per_page=100"
    )
    pulls = gh_api(f"repos/{args.repo}/pulls?state=open&per_page=100")
    prs = find_waiting_prs(pulls, waiting_runs_by_sha(runs))
    print(f"{len(prs)} open PR(s) wait for CI approval.")
    if not prs:
        return 0

    payload = build_message(prs, args.repo, datetime.now(timezone.utc))
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if args.dry_run or not webhook_url:
        if not webhook_url and not args.dry_run:
            print("::warning::SLACK_WEBHOOK_URL is not set. Printing the message only.")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    post_to_slack(webhook_url, payload)
    print("Posted the reminder to Slack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
