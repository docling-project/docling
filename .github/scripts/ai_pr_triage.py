# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Advisory AI triage for pull requests before a maintainer approves CI.

The workflow runs three steps, each in its own job:

* ``prepare`` (read-only token): collects PR data with the GitHub API, finds
  possible duplicate PRs, applies deterministic CI-safety rules, and summarizes
  groundtruth changes. It never checks out or runs PR code.
* ``extract`` (no GitHub token): validates the JSON answer of the Bob Shell run.
* ``publish`` (write token, no LLM): applies an allowlist of labels and updates
  one sticky comment.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from groundtruth_diff import is_groundtruth_path, render_markdown, summarize_file

COMMENT_MARKER = "<!-- ai-pr-triage -->"
LABEL_DUPLICATE = "ai:possible-duplicate"
LABEL_CI_SAFE = "ai:ci-safe"
LABEL_CI_NEEDS_CARE = "ai:ci-needs-care"
MANAGED_LABELS = (LABEL_DUPLICATE, LABEL_CI_SAFE, LABEL_CI_NEEDS_CARE)

MAX_ISSUE_REFS = 5
MAX_OPEN_PRS_SCANNED = 50
MAX_CANDIDATES = 5
MAX_PR_DIFF_CHARS = 120_000
MAX_CANDIDATE_DIFF_CHARS = 20_000
MAX_GROUNDTRUTH_FILES = 80
MAX_RISK_HINTS = 30
MAX_BODY_CHARS = 6_000
MAX_MODEL_TEXT_CHARS = 600
HUNK_SLACK_LINES = 3

# Changes in these paths can alter what CI executes. A match always requires
# maintainer care, whatever the model says.
SENSITIVE_PATHS: tuple[tuple[str, str], ...] = (
    (".github/*", "changes CI workflows, actions, or CI scripts"),
    ("conftest.py", "pytest runs this file at collection"),
    ("*/conftest.py", "pytest runs this file at collection"),
    ("pyproject.toml", "changes build, dependency, or tool configuration"),
    ("packages/*/pyproject.toml", "changes package build configuration"),
    ("uv.lock", "changes locked dependencies"),
    ("setup.py", "runs at install time"),
    ("setup.cfg", "changes install configuration"),
    ("Makefile", "changes local and CI commands"),
    (".pre-commit-config.yaml", "changes hooks that CI runs"),
    ("tach.toml", "changes CI test selection"),
    ("scripts/*", "changes maintenance scripts"),
    ("*.sh", "adds or changes a shell script"),
    ("*.pth", "Python runs .pth files at startup"),
    ("sitecustomize.py", "Python runs this file at startup"),
    ("*/sitecustomize.py", "Python runs this file at startup"),
    ("AGENTS.md", "changes instructions for AI agents"),
    ("CLAUDE.md", "changes instructions for AI agents"),
    (".agents/*", "changes instructions for AI agents"),
    (".claude/*", "changes instructions for AI agents"),
    (".codex/*", "changes instructions for AI agents"),
    (".bob/*", "changes instructions for AI agents"),
)
BINARY_SUFFIXES = frozenset(
    {".so", ".dll", ".dylib", ".exe", ".whl", ".pyc", ".pyd", ".bin", ".jar"}
)
# Added lines that match these patterns are shown to the model as hints. They
# do not force a verdict, because Docling code legitimately uses some of them.
RISK_HINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "process execution",
        re.compile(r"\b(subprocess|os\.system|os\.popen|os\.exec\w*|pty\.spawn)\b"),
    ),
    ("dynamic code execution", re.compile(r"\b(eval|exec|__import__)\s*\(")),
    (
        "network access",
        re.compile(
            r"\b(urllib\.request|requests\.(get|post|put)|httpx\.|socket\.)|\b(curl|wget)\s"
        ),
    ),
    (
        "encoded payload",
        re.compile(r"base64\.b64decode|bytes\.fromhex|[A-Za-z0-9+/]{200,}={0,2}"),
    ),
    (
        "environment or secret access",
        re.compile(
            r"(environ|getenv).*(TOKEN|SECRET|KEY|PASSWORD)|\bGITHUB_|\bACTIONS_"
        ),
    ),
    (
        "unsafe deserialization",
        re.compile(r"\b(pickle|marshal|dill)\.loads?\b|yaml\.load\("),
    ),
)
# These files change in many unrelated PRs, so they do not count as overlap.
OVERLAP_NOISE_PATHS = ("uv.lock", "pyproject.toml", "CHANGELOG.md", "*/__init__.py")
DIFF_EXCLUDED_PATHS = ("uv.lock",)
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data models shared by the jobs


@dataclass(slots=True)
class PullRequestInfo:
    number: int
    title: str
    body: str
    author: str
    author_association: str
    base_sha: str
    head_sha: str
    merge_base_sha: str
    changed_files: list[str]
    additions: int
    deletions: int


@dataclass(slots=True)
class Candidate:
    number: int
    title: str
    state: str
    url: str
    signals: list[str] = field(default_factory=list)
    shared_files: list[str] = field(default_factory=list)
    overlapping_hunks: int = 0
    score: int = 0


@dataclass(slots=True)
class RiskFinding:
    path: str
    reason: str


@dataclass(slots=True)
class RiskHint:
    path: str
    kind: str
    line: str


@dataclass(slots=True)
class RiskReport:
    forced: list[RiskFinding] = field(default_factory=list)
    hints: list[RiskHint] = field(default_factory=list)


@dataclass(slots=True)
class TriageContext:
    pr: PullRequestInfo
    issue_refs: list[int]
    candidates: list[Candidate]
    risk: RiskReport
    groundtruth_markdown: str
    diff_truncated: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriageContext:
        risk = data["risk"]
        return cls(
            pr=PullRequestInfo(**data["pr"]),
            issue_refs=list(data["issue_refs"]),
            candidates=[Candidate(**item) for item in data["candidates"]],
            risk=RiskReport(
                forced=[RiskFinding(**item) for item in risk["forced"]],
                hints=[RiskHint(**item) for item in risk["hints"]],
            ),
            groundtruth_markdown=data["groundtruth_markdown"],
            diff_truncated=data["diff_truncated"],
        )


@dataclass(slots=True)
class DuplicateVerdict:
    pr: int
    verdict: str
    reason: str


@dataclass(slots=True)
class SafetyConcern:
    path: str
    reason: str


@dataclass(slots=True)
class TriageResult:
    summary: str
    duplicates: list[DuplicateVerdict]
    ci_verdict: str
    ci_concerns: list[SafetyConcern]
    groundtruth_verdict: str | None
    groundtruth_reason: str


DUPLICATE_VERDICTS = frozenset({"duplicate", "related", "unrelated"})
CI_VERDICTS = frozenset({"safe", "needs-care"})
GROUNDTRUTH_VERDICTS = frozenset({"expected", "unexpected", "unclear"})


def parse_triage_result(data: Any, allowed_prs: set[int]) -> TriageResult:
    """Validate the model answer. Unknown values are errors, not defaults."""
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
    duplicates: list[DuplicateVerdict] = []
    for item in _require_list(data, "duplicates"):
        pr = item.get("pr") if isinstance(item, dict) else None
        verdict = item.get("verdict") if isinstance(item, dict) else None
        if not isinstance(pr, int) or pr not in allowed_prs:
            raise ValueError(f"duplicate entry names an unknown PR: {pr!r}")
        if verdict not in DUPLICATE_VERDICTS:
            raise ValueError(f"invalid duplicate verdict: {verdict!r}")
        duplicates.append(DuplicateVerdict(pr, verdict, _model_text(item, "reason")))

    safety = data.get("ci_safety")
    if not isinstance(safety, dict) or safety.get("verdict") not in CI_VERDICTS:
        raise ValueError("ci_safety.verdict must be 'safe' or 'needs-care'")
    concerns = [
        SafetyConcern(_model_text(item, "path"), _model_text(item, "reason"))
        for item in _require_list(safety, "concerns")
        if isinstance(item, dict)
    ]

    groundtruth = data.get("groundtruth")
    groundtruth_verdict: str | None = None
    groundtruth_reason = ""
    if groundtruth is not None:
        if (
            not isinstance(groundtruth, dict)
            or groundtruth.get("verdict") not in GROUNDTRUTH_VERDICTS
        ):
            raise ValueError("invalid groundtruth verdict")
        groundtruth_verdict = groundtruth["verdict"]
        groundtruth_reason = _model_text(groundtruth, "reason")

    return TriageResult(
        summary=_model_text(data, "summary"),
        duplicates=duplicates,
        ci_verdict=safety["verdict"],
        ci_concerns=concerns,
        groundtruth_verdict=groundtruth_verdict,
        groundtruth_reason=groundtruth_reason,
    )


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _model_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return sanitize_text(value if isinstance(value, str) else str(value))


def sanitize_text(text: str, limit: int = MAX_MODEL_TEXT_CHARS) -> str:
    """Make untrusted text safe to embed in one Markdown table cell or bullet."""
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = " ".join(text.split())
    text = text.replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")
    # Prevent notifications to users and teams.
    text = re.sub(r"@(?=[\w-])", "@​", text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def extract_answer(last_message: str) -> Any:
    """Return the JSON object in the final model message."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", last_message, re.DOTALL)
    candidate = fenced.group(1) if fenced else last_message
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("final message contains no JSON object")
    return json.loads(candidate[start : end + 1])


# ---------------------------------------------------------------------------
# Deterministic analysis


def extract_issue_refs(text: str, repo: str, own_number: int) -> list[int]:
    repo_url = re.escape(f"github.com/{repo}/")
    pattern = re.compile(rf"(?:{repo_url}(?:issues|pull)/|(?<![\w/])#)(\d+)\b")
    refs: list[int] = []
    for match in pattern.finditer(text):
        number = int(match.group(1))
        if number != own_number and number not in refs:
            refs.append(number)
    return refs[:MAX_ISSUE_REFS]


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def assess_risk(files: list[dict[str, Any]]) -> RiskReport:
    report = RiskReport()
    for item in files:
        paths = [item["filename"]]
        if item.get("previous_filename"):
            paths.append(item["previous_filename"])
        for path in paths:
            for pattern, reason in SENSITIVE_PATHS:
                if fnmatch.fnmatchcase(path, pattern):
                    report.forced.append(RiskFinding(path, reason))
                    break
            if Path(path).suffix.lower() in BINARY_SUFFIXES and not path.startswith(
                "tests/data/"
            ):
                report.forced.append(RiskFinding(path, "adds or changes a binary file"))

        path = item["filename"]
        if path.startswith("tests/data/") or len(report.hints) >= MAX_RISK_HINTS:
            continue
        for line in (item.get("patch") or "").splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for kind, pattern in RISK_HINT_PATTERNS:
                if pattern.search(line):
                    report.hints.append(RiskHint(path, kind, line[1:].strip()[:200]))
                    break
            if len(report.hints) >= MAX_RISK_HINTS:
                break
    return report


def hunk_ranges(patch: str | None) -> list[tuple[int, int]]:
    """Return the changed line ranges in the old file of a unified patch."""
    ranges: list[tuple[int, int]] = []
    for match in HUNK_HEADER.finditer(patch or ""):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        ranges.append((start, start + max(count, 1) - 1))
    return ranges


def count_overlapping_hunks(
    ours: list[tuple[int, int]], theirs: list[tuple[int, int]]
) -> int:
    return sum(
        1
        for a_start, a_end in ours
        if any(
            a_start <= b_end + HUNK_SLACK_LINES and b_start <= a_end + HUNK_SLACK_LINES
            for b_start, b_end in theirs
        )
    )


def compare_files(
    ours: list[dict[str, Any]], theirs: list[dict[str, Any]]
) -> tuple[list[str], int]:
    """Return the shared relevant files and the count of overlapping hunks."""
    their_patches = {item["filename"]: item.get("patch") for item in theirs}
    shared: list[str] = []
    overlaps = 0
    for item in ours:
        path = item["filename"]
        if (
            path not in their_patches
            or matches_any(path, OVERLAP_NOISE_PATHS)
            or is_groundtruth_path(path)
        ):
            continue
        shared.append(path)
        overlaps += count_overlapping_hunks(
            hunk_ranges(item.get("patch")), hunk_ranges(their_patches[path])
        )
    return shared, overlaps


def is_overlap_candidate(shared_files: list[str], overlapping_hunks: int) -> bool:
    return overlapping_hunks > 0 or len(shared_files) >= 3


def build_diff(
    files: list[dict[str, Any]], max_chars: int, only: set[str] | None = None
) -> tuple[str, bool]:
    parts: list[str] = []
    size = 0
    truncated = False
    for item in files:
        path = item["filename"]
        if only is not None and path not in only:
            continue
        if is_groundtruth_path(path) or matches_any(path, DIFF_EXCLUDED_PATHS):
            body = "(content omitted: reference data or lock file)\n"
        else:
            body = item.get("patch") or "(no text patch: binary or too large)\n"
        part = f"diff --git a/{path} b/{path}\nstatus: {item.get('status')}\n{body}\n"
        if size + len(part) > max_chars:
            truncated = True
            break
        parts.append(part)
        size += len(part)
    return "".join(parts), truncated


# ---------------------------------------------------------------------------
# GitHub and git access


def gh_api(path: str, *, paginate: bool = False, jq: str | None = None) -> Any:
    command = ["gh", "api", "-H", "Accept: application/vnd.github+json", path]
    if paginate:
        command[2:2] = ["--paginate", "--slurp"]
    if jq is not None:
        command += ["--jq", jq]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    if jq is not None:
        return output.strip()
    data = json.loads(output)
    if paginate:
        return [item for page in data for item in page]
    return data


def gh_write(method: str, path: str, payload: dict[str, Any] | None = None) -> None:
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command += ["--input", "-"]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        input=json.dumps(payload) if payload is not None else None,
    )


def pr_files(repo: str, number: int) -> list[dict[str, Any]]:
    return gh_api(f"repos/{repo}/pulls/{number}/files?per_page=100", paginate=True)


def git(git_dir: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(git_dir), *args], capture_output=True)


def fetch_pr_commits(git_dir: Path, pr_number: int, merge_base: str) -> bool:
    """Fetch the merge base and the PR head without blobs and without checkout.

    `git show` then loads each blob on demand, so the job never downloads the
    full test data. Nothing from the PR is checked out or run.
    """
    fetch = git(
        git_dir,
        "fetch",
        "--no-tags",
        "--depth=1",
        "--filter=blob:none",
        "origin",
        merge_base,
        f"pull/{pr_number}/head",
    )
    return fetch.returncode == 0


def read_blob(git_dir: Path, commit: str, path: str) -> bytes | None:
    result = git(git_dir, "show", f"{commit}:{path}")
    return result.stdout if result.returncode == 0 else None


# ---------------------------------------------------------------------------
# Jobs


def find_candidates(
    repo: str,
    pr_number: int,
    refs: list[int],
    files: list[dict[str, Any]],
) -> list[tuple[Candidate, list[dict[str, Any]]]]:
    found: dict[int, Candidate] = {}

    def add(item: dict[str, Any], signal: str) -> Candidate:
        number = int(item["number"])
        candidate = found.get(number)
        if candidate is None:
            merged = (item.get("pull_request") or {}).get("merged_at") or item.get(
                "merged_at"
            )
            state = "merged" if merged else str(item.get("state", "unknown"))
            candidate = Candidate(
                number=number,
                title=str(item.get("title", "")),
                state=state,
                url=str(item.get("html_url", "")),
            )
            found[number] = candidate
        if signal not in candidate.signals:
            candidate.signals.append(signal)
            candidate.score += 10
        return candidate

    for ref in refs:
        issue = gh_api(f"repos/{repo}/issues/{ref}")
        if "pull_request" in issue:
            add(issue, f"referenced as #{ref}")
            continue
        events = gh_api(
            f"repos/{repo}/issues/{ref}/timeline?per_page=100", paginate=True
        )
        for event in events:
            source = (event.get("source") or {}).get("issue") or {}
            source_repo = (source.get("repository") or {}).get("full_name")
            if (
                event.get("event") == "cross-referenced"
                and "pull_request" in source
                and source_repo == repo
                and source.get("number") != pr_number
            ):
                add(source, f"also references issue #{ref}")

    open_prs = gh_api(
        f"repos/{repo}/pulls?state=open&sort=updated&direction=desc"
        f"&per_page={MAX_OPEN_PRS_SCANNED}"
    )
    files_by_pr: dict[int, list[dict[str, Any]]] = {}
    for other in open_prs:
        number = int(other["number"])
        if number == pr_number:
            continue
        other_files = pr_files(repo, number)
        files_by_pr[number] = other_files
        shared, overlaps = compare_files(files, other_files)
        if is_overlap_candidate(shared, overlaps):
            candidate = add(other, "changes the same code")
        elif number in found:
            candidate = found[number]
        else:
            continue
        candidate.shared_files = shared
        candidate.overlapping_hunks = overlaps
        candidate.score += 2 * overlaps + len(shared)

    ranked = sorted(found.values(), key=lambda c: (-c.score, -c.number))
    selected = ranked[:MAX_CANDIDATES]
    result: list[tuple[Candidate, list[dict[str, Any]]]] = []
    for candidate in selected:
        other_files = files_by_pr.get(candidate.number)
        if other_files is None:
            other_files = pr_files(repo, candidate.number)
            candidate.shared_files, candidate.overlapping_hunks = compare_files(
                files, other_files
            )
        result.append((candidate, other_files))
    return result


def summarize_groundtruth(
    git_dir: Path,
    pr_number: int,
    merge_base: str,
    head: str,
    files: list[dict[str, Any]],
) -> str:
    changed = [item for item in files if is_groundtruth_path(item["filename"])]
    if not changed:
        return ""
    if not fetch_pr_commits(git_dir, pr_number, merge_base):
        return "Could not fetch the PR commits to compare the reference data."
    changes = []
    for item in changed[:MAX_GROUNDTRUTH_FILES]:
        path = item["filename"]
        old_path = item.get("previous_filename") or path
        old = (
            None
            if item["status"] == "added"
            else read_blob(git_dir, merge_base, old_path)
        )
        new = None if item["status"] == "removed" else read_blob(git_dir, head, path)
        changes.append(summarize_file(path, old, new))
    markdown = render_markdown(changes)
    if len(changed) > MAX_GROUNDTRUTH_FILES:
        markdown += (
            f"\n\nOnly the first {MAX_GROUNDTRUTH_FILES} of {len(changed)}"
            " reference files were compared."
        )
    return markdown


def prepare(repo: str, pr_number: int, out_dir: Path, git_dir: Path) -> bool:
    pr = gh_api(f"repos/{repo}/pulls/{pr_number}")
    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]
    merge_base = gh_api(
        f"repos/{repo}/compare/{base_sha}...{head_sha}", jq=".merge_base_commit.sha"
    )
    files = pr_files(repo, pr_number)
    body = pr.get("body") or ""
    info = PullRequestInfo(
        number=pr_number,
        title=pr["title"],
        body=body[:MAX_BODY_CHARS],
        author=pr["user"]["login"],
        author_association=pr.get("author_association", ""),
        base_sha=base_sha,
        head_sha=head_sha,
        merge_base_sha=merge_base,
        changed_files=[item["filename"] for item in files],
        additions=int(pr.get("additions", 0)),
        deletions=int(pr.get("deletions", 0)),
    )
    refs = extract_issue_refs(f"{pr['title']}\n{body}", repo, pr_number)
    candidates = find_candidates(repo, pr_number, refs, files)
    diff, truncated = build_diff(files, MAX_PR_DIFF_CHARS)
    context = TriageContext(
        pr=info,
        issue_refs=refs,
        candidates=[candidate for candidate, _ in candidates],
        risk=assess_risk(files),
        groundtruth_markdown=summarize_groundtruth(
            git_dir, pr_number, merge_base, head_sha, files
        ),
        diff_truncated=truncated,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "context.json").write_text(
        json.dumps(asdict(context), indent=2), encoding="utf-8"
    )
    (out_dir / "pr.diff").write_text(diff, encoding="utf-8")
    (out_dir / "groundtruth.md").write_text(
        context.groundtruth_markdown, encoding="utf-8"
    )
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for candidate, other_files in candidates:
        only = set(candidate.shared_files) or None
        candidate_diff, _ = build_diff(other_files, MAX_CANDIDATE_DIFF_CHARS, only)
        (candidates_dir / f"{candidate.number}.diff").write_text(
            candidate_diff, encoding="utf-8"
        )
    return bool(files)


def extract(bob_output: Path, context_path: Path, out_path: Path) -> None:
    context = TriageContext.from_dict(json.loads(context_path.read_text("utf-8")))
    output = json.loads(bob_output.read_text("utf-8"))
    stats = output.get("stats")
    print(f"Bob Shell status: {output.get('status')}; stats: {json.dumps(stats)}")
    last_message = output.get("last_message")
    if not isinstance(last_message, str):
        raise ValueError("Bob Shell output has no final message")
    allowed = {candidate.number for candidate in context.candidates}
    result = parse_triage_result(extract_answer(last_message), allowed)
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def decide_labels(context: TriageContext, result: TriageResult | None) -> set[str]:
    labels: set[str] = set()
    if context.risk.forced or (result is not None and result.ci_verdict != "safe"):
        labels.add(LABEL_CI_NEEDS_CARE)
    elif result is not None:
        labels.add(LABEL_CI_SAFE)
    if result is not None and any(d.verdict == "duplicate" for d in result.duplicates):
        labels.add(LABEL_DUPLICATE)
    return labels


def render_comment(
    context: TriageContext, result: TriageResult | None, labels: set[str]
) -> str:
    head = context.pr.head_sha[:10]
    lines = [
        COMMENT_MARKER,
        "### AI triage (advisory)",
        "",
        f"Checked commit `{head}`. The analysis read the diff only."
        " It did not run code from this pull request.",
        "",
    ]
    if result is None:
        lines += [
            "> [!WARNING]",
            "> The AI analysis did not finish. Only the deterministic checks are shown.",
            "",
        ]
    elif result.summary:
        lines += [result.summary, ""]

    if LABEL_CI_NEEDS_CARE in labels:
        lines.append("**CI safety: needs maintainer care** before you approve CI.")
    elif LABEL_CI_SAFE in labels:
        lines.append(
            "**CI safety: no concern found.** A maintainer must still approve CI."
        )
    else:
        lines.append("**CI safety: not assessed.**")
    for finding in context.risk.forced:
        lines.append(f"- `{sanitize_text(finding.path)}`: {finding.reason}")
    if result is not None:
        for concern in result.ci_concerns:
            path = f"`{concern.path}`: " if concern.path else ""
            lines.append(f"- {path}{concern.reason}")
    lines.append("")

    verdicts = {d.pr: d.verdict for d in result.duplicates} if result else {}
    reasons = {d.pr: d.reason for d in result.duplicates} if result else {}
    # Duplicates and unassessed candidates stay visible. Related PRs are
    # collapsed, and unrelated PRs are hidden.
    main = [c for c in context.candidates if verdicts.get(c.number) in (None, "duplicate")]
    related = [c for c in context.candidates if verdicts.get(c.number) == "related"]
    if main:
        title = "Possible duplicates" if result else "Possibly related pull requests"
        lines += [f"**{title}**", ""]
        lines += _candidate_table(main, verdicts, reasons)
        lines.append("")
    if related:
        lines += [
            "<details>",
            f"<summary>{len(related)} related pull request(s) in the same code</summary>",
            "",
        ]
        lines += _candidate_table(related, verdicts, reasons)
        lines += ["", "</details>", ""]

    if context.groundtruth_markdown:
        lines += ["**Reference data changes**", "", context.groundtruth_markdown, ""]
        if result is not None and result.groundtruth_verdict is not None:
            lines += [
                f"Assessment: **{result.groundtruth_verdict}**. {result.groundtruth_reason}",
                "",
            ]

    lines.append(
        "<sub>The `ai:*` labels are advisory. They do not approve CI and they do not"
        " add `tests:full`. A new push runs this check again.</sub>"
    )
    return "\n".join(lines)


def _candidate_table(
    candidates: list[Candidate], verdicts: dict[int, str], reasons: dict[int, str]
) -> list[str]:
    rows = ["| PR | State | Signal | Assessment |", "|---|---|---|---|"]
    for candidate in candidates:
        verdict = verdicts.get(candidate.number)
        assessment = (
            f"{verdict}: {reasons[candidate.number]}" if verdict else "not assessed"
        )
        signals = "; ".join(candidate.signals)
        rows.append(
            f"| #{candidate.number} | {candidate.state} | {signals} | {assessment} |"
        )
    return rows


def publish(
    repo: str, pr_number: int, context_path: Path, result_path: Path | None
) -> None:
    context = TriageContext.from_dict(json.loads(context_path.read_text("utf-8")))
    pr = gh_api(f"repos/{repo}/pulls/{pr_number}")
    if pr["head"]["sha"] != context.pr.head_sha:
        print("The PR has a newer commit. The newer run publishes the result.")
        return

    result: TriageResult | None = None
    if result_path is not None and result_path.is_file():
        allowed = {candidate.number for candidate in context.candidates}
        try:
            data = json.loads(result_path.read_text("utf-8"))
            result = parse_triage_result(_result_to_answer(data), allowed)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Ignoring an invalid analysis result: {exc}")

    labels = decide_labels(context, result)
    current = {label["name"] for label in pr["labels"]}
    for label in sorted((current & set(MANAGED_LABELS)) - labels):
        gh_write("DELETE", f"repos/{repo}/issues/{pr_number}/labels/{label}")
    if labels - current:
        gh_write(
            "POST",
            f"repos/{repo}/issues/{pr_number}/labels",
            {"labels": sorted(labels - current)},
        )

    body = render_comment(context, result, labels)
    comments = gh_api(
        f"repos/{repo}/issues/{pr_number}/comments?per_page=100", paginate=True
    )
    existing = next(
        (
            comment
            for comment in comments
            if comment["user"]["login"] == "github-actions[bot]"
            and COMMENT_MARKER in comment["body"]
        ),
        None,
    )
    if existing is None:
        gh_write("POST", f"repos/{repo}/issues/{pr_number}/comments", {"body": body})
    else:
        gh_write(
            "PATCH", f"repos/{repo}/issues/comments/{existing['id']}", {"body": body}
        )


def _result_to_answer(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a stored TriageResult back to the answer shape for re-validation."""
    groundtruth = None
    if data["groundtruth_verdict"] is not None:
        groundtruth = {
            "verdict": data["groundtruth_verdict"],
            "reason": data["groundtruth_reason"],
        }
    return {
        "summary": data["summary"],
        "duplicates": data["duplicates"],
        "ci_safety": {"verdict": data["ci_verdict"], "concerns": data["ci_concerns"]},
        "groundtruth": groundtruth,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", required=True)
    prep.add_argument("--pr", type=int, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--git-dir", type=Path, default=Path.cwd())

    ext = sub.add_parser("extract")
    ext.add_argument("--bob-output", type=Path, required=True)
    ext.add_argument("--context", type=Path, required=True)
    ext.add_argument("--out", type=Path, required=True)

    pub = sub.add_parser("publish")
    pub.add_argument("--repo", required=True)
    pub.add_argument("--pr", type=int, required=True)
    pub.add_argument("--context", type=Path, required=True)
    pub.add_argument("--result", type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        has_changes = prepare(args.repo, args.pr, args.out, args.git_dir)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"run_llm={'true' if has_changes else 'false'}\n")
    elif args.command == "extract":
        try:
            extract(args.bob_output, args.context, args.out)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"::warning::The Bob Shell answer is not valid: {exc}")
            return 1
    else:
        publish(args.repo, args.pr, args.context, args.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
