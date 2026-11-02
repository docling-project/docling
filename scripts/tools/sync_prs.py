# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Rank the open PRs by merge readiness, and act on them with your gh login.

The script saves the metadata of each PR to ``<cache-dir>/<number>/pr.json``
(title, author, labels, reviews, checks, and the computed rank). It does not
download code. It then prints these lists:

1. Ready to merge: CI is green, the PR has all the approvals it needs, and
   GitHub reports the merge state ``CLEAN``.
2. Needs approval: CI is green, but the PR needs at least one more approval.
3. Merge conflict: the PR conflicts with its base branch.
4. Other: drafts, failing or pending CI, CI that waits for maintainer
   approval, and PRs that GitHub blocks for another reason.

"CI is green" means: every required status check of the base branch is
present, and every check is complete with ``SUCCESS``, ``SKIPPED``, or
``NEUTRAL``. The ``Mergify Merge Protections`` check is not part of CI,
because it also counts approvals.

The number of approvals that a PR needs follows the repository rules: the
branch ruleset (1), Mergify for changes in ``tests/data`` (2), and Mergify for
breaking changes with ``!:`` in the title (3). Only approvals from owners,
members, and collaborators count.

Actions (all of them are off by default):

``--comment-conflicts``
    Post a comment that asks the author to resolve the merge conflict. The
    script posts one comment per PR head commit, so a rerun does not post the
    same comment again.

``--approve-ci``
    Approve the ``pull_request`` runs that wait for a maintainer (conclusion
    ``action_required``) when all of these conditions are true: the PR is not a
    draft, it has the label ``ai:ci-safe`` and not ``ai:ci-needs-care``, the
    run is for the current head commit, and the label was added after GitHub
    created the run (so the triage saw that commit).

Examples::

    uv run python scripts/tools/sync_prs.py
    uv run python scripts/tools/sync_prs.py --comment-conflicts --approve-ci
    uv run python scripts/tools/sync_prs.py --pr 5115 --pr 5122
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_REPO = "docling-project/docling"
SAFE_LABEL = "ai:ci-safe"
NEEDS_CARE_LABEL = "ai:ci-needs-care"
APPROVAL_CONCLUSION = "action_required"
MERGIFY_CHECK = "Mergify Merge Protections"
GREEN_RESULTS = {"SUCCESS", "SKIPPED", "NEUTRAL"}
FAILED_RESULTS = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}
MAINTAINER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR"}
READY_MERGE_STATES = {"CLEAN", "HAS_HOOKS"}
CONFLICT_MARKER = "<!-- sync-prs:merge-conflict head={sha} -->"
PR_FRAGMENT = """
fragment PrFields on PullRequest {
  number title url isDraft headRefOid baseRefName updatedAt
  mergeable mergeStateStatus reviewDecision
  author { login }
  labels(first: 50) { nodes { name } }
  latestReviews(first: 50) {
    nodes { state authorAssociation author { login } }
  }
  files(first: 100) { nodes { path } }
  commits(last: 1) {
    nodes { commit { statusCheckRollup { contexts(first: 100) { nodes {
      __typename
      ... on CheckRun { name status conclusion }
      ... on StatusContext { context state }
    } } } } }
  }
}
"""
LIST_QUERY = (
    PR_FRAGMENT
    + """
query($owner: String!, $name: String!, $size: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: $size, after: $after,
                 orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { ...PrFields }
    }
  }
}
"""
)
VIEW_QUERY = (
    PR_FRAGMENT
    + """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { ...PrFields }
  }
}
"""
)
# Status checks and files make the query heavy; large pages time out (502).
PAGE_SIZE = 10
GRAPHQL_ATTEMPTS = 3
# GitHub computes mergeability on demand; ask again for UNKNOWN results.
UNKNOWN_RETRY_DELAY_S = 5


class Rank(str, Enum):
    READY = "ready to merge"
    NEEDS_APPROVAL = "CI green, needs approval"
    CONFLICT = "merge conflict"
    OTHER = "other"


@dataclass(slots=True)
class Check:
    name: str
    result: str  # conclusion if complete, else status ("IN_PROGRESS", ...)


@dataclass(slots=True)
class WaitingRun:
    id: int
    name: str
    created_at: datetime
    url: str


@dataclass(slots=True)
class PullRequest:
    number: int
    title: str
    url: str
    author: str
    draft: bool
    labels: list[str]
    head_sha: str
    base_ref: str
    updated_at: str
    mergeable: str
    merge_state: str
    review_decision: str
    approvers: list[str]
    changed_files: list[str]
    checks: list[Check]
    required_approvals: int = 1
    rank: Rank = Rank.OTHER
    reason: str = ""
    waiting_runs: list[WaitingRun] = field(default_factory=list)

    @classmethod
    def from_graphql(cls, data: dict[str, Any]) -> PullRequest:
        commits = data["commits"]["nodes"]
        rollup = commits[0]["commit"]["statusCheckRollup"] if commits else None
        contexts = rollup["contexts"]["nodes"] if rollup else []
        return cls(
            number=int(data["number"]),
            title=str(data["title"]),
            url=str(data["url"]),
            # The author is None for deleted accounts.
            author=str((data["author"] or {"login": "ghost"})["login"]),
            draft=bool(data["isDraft"]),
            labels=sorted(label["name"] for label in data["labels"]["nodes"]),
            head_sha=str(data["headRefOid"]),
            base_ref=str(data["baseRefName"]),
            updated_at=str(data["updatedAt"]),
            mergeable=str(data["mergeable"]),
            merge_state=str(data["mergeStateStatus"]),
            review_decision=str(data["reviewDecision"] or ""),
            approvers=sorted(
                review["author"]["login"]
                for review in data["latestReviews"]["nodes"]
                if review["state"] == "APPROVED"
                and review["author"] is not None
                and review["authorAssociation"] in MAINTAINER_ROLES
            ),
            changed_files=[item["path"] for item in data["files"]["nodes"]],
            checks=[parse_check(item) for item in contexts],
        )


def parse_check(item: dict[str, Any]) -> Check:
    # The rollup mixes check runs (name/status/conclusion) and commit
    # statuses (context/state).
    if item["__typename"] == "StatusContext":
        return Check(name=str(item["context"]), result=str(item["state"]))
    result = item["conclusion"] if item["status"] == "COMPLETED" else item["status"]
    return Check(name=str(item["name"]), result=str(result))


# --- GitHub access ---------------------------------------------------------


def run(command: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        command, input=stdin, check=True, capture_output=True, text=True
    )
    return result.stdout


def gh_json(args: list[str]) -> Any:
    return json.loads(run(["gh", *args]))


def gh_api_list(path: str, key: str | None = None) -> list[dict[str, Any]]:
    pages = gh_json(["api", "--paginate", "--slurp", path])
    items: list[dict[str, Any]] = []
    for page in pages:
        items.extend(page[key] if key else page)
    return items


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def graphql(query: str, **variables: str | int) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        flag = "-F" if isinstance(value, int) else "-f"
        command += [flag, f"{key}={value}"]
    for attempt in range(1, GRAPHQL_ATTEMPTS + 1):
        try:
            return gh_json(command[1:])["data"]
        except subprocess.CalledProcessError as exc:
            # Heavy queries sometimes time out on GitHub (HTTP 502).
            if attempt == GRAPHQL_ATTEMPTS or "HTTP 50" not in exc.stderr:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def view_pr(repo: str, number: int) -> PullRequest:
    owner, name = repo.split("/")
    data = graphql(VIEW_QUERY, owner=owner, name=name, number=number)
    return PullRequest.from_graphql(data["repository"]["pullRequest"])


def list_prs(repo: str, limit: int, numbers: list[int]) -> list[PullRequest]:
    if numbers:
        prs = [view_pr(repo, number) for number in numbers]
    else:
        owner, name = repo.split("/")
        prs = []
        after: dict[str, str | int] = {}
        while len(prs) < limit:
            size = min(PAGE_SIZE, limit - len(prs))
            data = graphql(LIST_QUERY, owner=owner, name=name, size=size, **after)
            page = data["repository"]["pullRequests"]
            prs += [PullRequest.from_graphql(node) for node in page["nodes"]]
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = {"after": str(page["pageInfo"]["endCursor"])}
    unknown = [pr.number for pr in prs if pr.mergeable == "UNKNOWN"]
    if unknown:
        # The first query starts the computation on GitHub.
        time.sleep(UNKNOWN_RETRY_DELAY_S)
        fresh = {number: view_pr(repo, number) for number in unknown}
        prs = [fresh.get(pr.number, pr) for pr in prs]
    return prs


def branch_rules(repo: str, branch: str) -> tuple[set[str], int]:
    """Return the required status checks and approvals of a branch."""
    rules = gh_api_list(f"repos/{repo}/rules/branches/{branch}?per_page=100")
    checks: set[str] = set()
    approvals = 0
    for rule in rules:
        if rule["type"] == "required_status_checks":
            for check in rule["parameters"]["required_status_checks"]:
                checks.add(str(check["context"]))
        elif rule["type"] == "pull_request":
            count = int(rule["parameters"]["required_approving_review_count"])
            approvals = max(approvals, count)
    return checks, approvals


def waiting_runs_by_sha(repo: str) -> dict[str, list[WaitingRun]]:
    runs = gh_api_list(
        f"repos/{repo}/actions/runs?status={APPROVAL_CONCLUSION}"
        "&event=pull_request&per_page=100",
        key="workflow_runs",
    )
    waiting: dict[str, list[WaitingRun]] = {}
    for item in runs:
        waiting.setdefault(item["head_sha"], []).append(
            WaitingRun(
                id=int(item["id"]),
                name=str(item["name"]),
                created_at=parse_time(item["created_at"]),
                url=str(item["html_url"]),
            )
        )
    return waiting


# --- Ranking ---------------------------------------------------------------


def mergify_approvals(pr: PullRequest) -> int:
    """Return the approvals that ``.github/mergify.yml`` requires."""
    if "!:" in pr.title:
        return 3
    if any(path.startswith("tests/data") for path in pr.changed_files):
        # Also matches tests/data_scanned, like the Mergify rule.
        return 2
    return 0


def ci_problem(pr: PullRequest, required: set[str]) -> str | None:
    """Return why CI is not green, or None if it is green."""
    checks = [check for check in pr.checks if check.name != MERGIFY_CHECK]
    failed = sorted({c.name for c in checks if c.result in FAILED_RESULTS})
    if failed:
        return "CI failed: " + ", ".join(failed)
    done = GREEN_RESULTS | FAILED_RESULTS
    pending = sorted({c.name for c in checks if c.result not in done})
    if pending:
        return "CI running: " + ", ".join(pending)
    missing = sorted(required - {MERGIFY_CHECK} - {c.name for c in checks})
    if missing:
        if pr.waiting_runs:
            return "CI waits for maintainer approval"
        return "CI missing: " + ", ".join(missing)
    if not checks:
        return "no CI checks"
    return None


def rank_pr(pr: PullRequest, required: set[str], base_approvals: int) -> None:
    pr.required_approvals = max(base_approvals, mergify_approvals(pr))
    approvals = f"{len(pr.approvers)}/{pr.required_approvals} approvals"
    missing_approvals = pr.required_approvals - len(pr.approvers)
    if pr.review_decision == "REVIEW_REQUIRED":
        missing_approvals = max(missing_approvals, 1)

    if pr.draft:
        pr.rank, pr.reason = Rank.OTHER, "draft"
    elif pr.mergeable == "CONFLICTING":
        pr.rank, pr.reason = Rank.CONFLICT, f"conflicts with {pr.base_ref}"
    elif (problem := ci_problem(pr, required)) is not None:
        pr.rank, pr.reason = Rank.OTHER, problem
    elif pr.review_decision == "CHANGES_REQUESTED":
        pr.rank, pr.reason = Rank.OTHER, f"changes requested, {approvals}"
    elif missing_approvals > 0:
        pr.rank = Rank.NEEDS_APPROVAL
        pr.reason = f"{approvals}, needs {missing_approvals} more"
    elif pr.merge_state not in READY_MERGE_STATES:
        # For example BLOCKED: unresolved review threads or a Mergify rule.
        pr.rank, pr.reason = Rank.OTHER, f"{approvals}, merge state {pr.merge_state}"
    else:
        pr.rank, pr.reason = Rank.READY, approvals


# --- Actions ---------------------------------------------------------------


def conflict_comment(pr: PullRequest) -> str:
    return (
        f"{CONFLICT_MARKER.format(sha=pr.head_sha)}\n"
        f"Hi @{pr.author}, thank you for this pull request.\n\n"
        f"It has a merge conflict with `{pr.base_ref}`. Please rebase your"
        f" branch on the latest `{pr.base_ref}` and resolve the conflict:\n\n"
        "```bash\n"
        "git fetch upstream\n"
        f"git rebase upstream/{pr.base_ref}\n"
        "# resolve the conflicts, then:\n"
        "git push --force-with-lease\n"
        "```\n\n"
        "We can continue with the review and CI after that."
    )


def comment_conflict(repo: str, pr: PullRequest) -> str:
    marker = CONFLICT_MARKER.format(sha=pr.head_sha)
    comments = gh_api_list(f"repos/{repo}/issues/{pr.number}/comments?per_page=100")
    if any(marker in str(comment["body"]) for comment in comments):
        return "already commented for this commit"
    run(
        ["gh", "pr", "comment", str(pr.number), "-R", repo, "--body-file", "-"],
        stdin=conflict_comment(pr),
    )
    return "commented"


def safe_label_added_at(repo: str, number: int) -> datetime | None:
    """Return when ``ai:ci-safe`` was last added, or None if it was removed."""
    events = gh_api_list(f"repos/{repo}/issues/{number}/events?per_page=100")
    added: datetime | None = None
    for event in events:
        if event.get("label", {}).get("name") != SAFE_LABEL:
            continue
        if event["event"] == "labeled":
            added = parse_time(event["created_at"])
        elif event["event"] == "unlabeled":
            added = None
    return added


def ci_approval_blocker(repo: str, pr: PullRequest) -> str | None:
    """Return why the waiting runs must not be approved, or None."""
    if pr.draft:
        return "draft"
    if NEEDS_CARE_LABEL in pr.labels:
        return f"has {NEEDS_CARE_LABEL}"
    if SAFE_LABEL not in pr.labels:
        return f"no {SAFE_LABEL}"
    added = safe_label_added_at(repo, pr.number)
    if added is None or added < max(item.created_at for item in pr.waiting_runs):
        return f"{SAFE_LABEL} is older than the last push"
    return None


def approve_runs(repo: str, runs: list[WaitingRun]) -> list[str]:
    errors: list[str] = []
    for item in runs:
        path = f"repos/{repo}/actions/runs/{item.id}/approve"
        try:
            run(["gh", "api", "--method", "POST", path])
        except subprocess.CalledProcessError as exc:
            errors.append(f"{item.name} ({item.url}): {exc.stderr.strip()}")
    return errors


# --- Output ----------------------------------------------------------------


def save_metadata(cache_dir: Path, pr: PullRequest) -> None:
    pr_dir = cache_dir / str(pr.number)
    pr_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(pr) | {"rank": pr.rank.value}
    (pr_dir / "pr.json").write_text(
        json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
    )


def print_lists(prs: list[PullRequest]) -> None:
    for index, rank in enumerate(Rank, start=1):
        members = [pr for pr in prs if pr.rank is rank]
        title = rank.value[0].upper() + rank.value[1:]
        print(f"\n## {index}. {title} ({len(members)})\n")
        for pr in members:
            print(f"- #{pr.number} {pr.title}")
            print(f"  @{pr.author} · {pr.reason} · {pr.url}")


def handle_conflicts(repo: str, prs: list[PullRequest]) -> bool:
    """Comment on the conflicting PRs. Return True if all comments worked."""
    conflicts = [pr for pr in prs if pr.rank is Rank.CONFLICT]
    if not conflicts:
        return True
    ok = True
    print("\n## Merge conflict comments\n")
    for pr in conflicts:
        try:
            print(f"- #{pr.number}: {comment_conflict(repo, pr)}")
        except subprocess.CalledProcessError as exc:
            ok = False
            print(f"- #{pr.number}: error: {exc.stderr.strip()}", file=sys.stderr)
    return ok


def handle_waiting_ci(repo: str, prs: list[PullRequest], approve: bool) -> bool:
    """Approve or list the waiting CI runs. Return True if no approval failed."""
    waiting = [pr for pr in prs if pr.waiting_runs]
    if not waiting:
        return True
    ok = True
    mode = "approve" if approve else "dry run, use --approve-ci"
    print(f"\n## CI that waits for maintainer approval ({mode})\n")
    for pr in waiting:
        blocker = ci_approval_blocker(repo, pr)
        if blocker is not None:
            print(f"- #{pr.number}: skip ({blocker})")
        elif not approve:
            print(f"- #{pr.number}: would approve {len(pr.waiting_runs)} run(s)")
        else:
            errors = approve_runs(repo, pr.waiting_runs)
            for error in errors:
                ok = False
                print(f"- #{pr.number}: error: {error}", file=sys.stderr)
            done = len(pr.waiting_runs) - len(errors)
            print(f"- #{pr.number}: approved {done} run(s)")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of recently updated open PRs."
    )
    parser.add_argument(
        "--pr",
        dest="numbers",
        type=int,
        action="append",
        default=[],
        help="Only this PR (repeatable). Ignores --limit.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/prs"))
    parser.add_argument(
        "--comment-conflicts",
        action="store_true",
        help="Ask the authors of conflicting PRs to resolve the conflict.",
    )
    parser.add_argument(
        "--approve-ci",
        action="store_true",
        help=f"Approve the waiting CI runs of PRs with {SAFE_LABEL}.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(run(["git", "rev-parse", "--show-toplevel"]).strip())
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir

    prs = list_prs(args.repo, args.limit, args.numbers)
    waiting = waiting_runs_by_sha(args.repo)
    rules: dict[str, tuple[set[str], int]] = {}
    for pr in prs:
        if pr.base_ref not in rules:
            rules[pr.base_ref] = branch_rules(args.repo, pr.base_ref)
        pr.waiting_runs = waiting.get(pr.head_sha, [])
        rank_pr(pr, *rules[pr.base_ref])
        save_metadata(cache_dir, pr)

    print(f"# {len(prs)} open PR(s) in {args.repo}")
    print(f"Metadata: {cache_dir}/<number>/pr.json")
    print_lists(prs)

    ok = True
    if args.comment_conflicts:
        ok = handle_conflicts(args.repo, prs) and ok
    ok = handle_waiting_ci(args.repo, prs, args.approve_ci) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: {' '.join(exc.cmd[:3])} failed: {exc.stderr.strip()}")
