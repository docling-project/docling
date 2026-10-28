# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Post a Slack summary of open PRs whose CI waits for maintainer approval.

GitHub holds ``pull_request`` runs from new contributors' forks with the
conclusion ``action_required`` until a maintainer approves them. These runs
have an empty ``pull_requests`` list, so they are matched to open PRs by the
head commit. The Slack message shows how many PRs wait in each AI triage
group. The full list goes to the GitHub Actions job summary, and the message
links to it.
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
from pathlib import Path
from typing import Any

APPROVAL_CONCLUSION = "action_required"
AI_LABEL_PREFIX = "ai:"
# (heading, label that selects the group); None collects the rest.
# The group that needs the most attention comes first.
GROUPS: tuple[tuple[str, str | None], ...] = (
    (":warning: Needs care", "ai:ci-needs-care"),
    (":grey_question: Not triaged", None),
    (":white_check_mark: No CI concern", "ai:ci-safe"),
)
GROUP_LABELS = {label for _, label in GROUPS if label}


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


def markdown_escape(text: str) -> str:
    """Escape text for a GitHub Markdown table cell."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace("|", "\\|")


def _age(since: datetime, now: datetime) -> str:
    hours = max(int((now - since).total_seconds() // 3600), 0)
    return f"{hours // 24} d" if hours >= 48 else f"{hours} h"


def group_prs(prs: list[WaitingPr]) -> list[tuple[str, list[WaitingPr]]]:
    groups: list[tuple[str, list[WaitingPr]]] = []
    for heading, label in GROUPS:
        if label is None:
            members = [pr for pr in prs if not GROUP_LABELS & set(pr.ai_labels)]
        else:
            members = [pr for pr in prs if label in pr.ai_labels]
        if members:
            groups.append((heading, members))
    return groups


def build_message(
    prs: list[WaitingPr], repo: str, now: datetime, summary_url: str | None = None
) -> dict[str, Any]:
    count = len(prs)
    oldest = _age(min(pr.waiting_since for pr in prs), now)
    counts = "\n".join(
        f"{heading}: *{len(members)}*" for heading, members in group_prs(prs)
    )
    if summary_url:
        link = f"*<{summary_url}|:point_right: Open the full list of {count} PRs>*"
    else:
        link = f"*<https://github.com/{repo}/pulls|:point_right: Open the PRs>*"
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🚦 {count} PR(s) wait for CI approval · oldest {oldest}",
            },
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": counts}},
        {"type": "section", "text": {"type": "mrkdwn", "text": link}},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "The `ai:*` labels are advisory."
                        " Read the PR before you approve CI."
                    ),
                }
            ],
        },
    ]
    summary = f"{count} PR(s) in {repo} wait for a maintainer to approve CI"
    return {"text": summary, "blocks": blocks}


def build_summary(prs: list[WaitingPr], repo: str, now: datetime) -> str:
    """Render the full list as Markdown for the GitHub Actions job summary."""
    lines = [f"# {len(prs)} PR(s) in {repo} wait for a maintainer to approve CI", ""]
    for heading, members in group_prs(prs):
        lines.extend(
            [
                f"## {heading} ({len(members)})",
                "",
                "| PR | Title | Author | Waiting | Workflows | Labels |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for pr in members:
            labels = " ".join(
                f"`{label}`" for label in pr.ai_labels if label not in GROUP_LABELS
            )
            lines.append(
                f"| [#{pr.number}]({pr.url}) | {markdown_escape(pr.title)}"
                f" | @{pr.author} | {_age(pr.waiting_since, now)}"
                f" | {markdown_escape(', '.join(pr.workflows))} | {labels} |"
            )
        lines.append("")
    lines.append("The `ai:*` labels are advisory. Read the PR before you approve CI.")
    return "\n".join(lines) + "\n"


def run_url() -> str | None:
    """Return the URL of the current GitHub Actions run, if there is one."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not run_id or not repo:
        return None
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{repo}/actions/runs/{run_id}"


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

    now = datetime.now(timezone.utc)
    summary = build_summary(prs, args.repo, now)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    summary_url = None
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(summary)
        summary_url = run_url()

    payload = build_message(prs, args.repo, now, summary_url)
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if args.dry_run or not webhook_url:
        if not webhook_url and not args.dry_run:
            print("::warning::SLACK_WEBHOOK_URL is not set. Printing the message only.")
        if not summary_path:
            print(summary)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    post_to_slack(webhook_url, payload)
    print("Posted the reminder to Slack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
