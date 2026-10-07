# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Advisory first review of a pull request with inline comments.

This extends the AI triage workflow. The same trust rules apply:

* ``prepare`` (read-only token): writes the PR head versions of changed source
  files as plain text, the valid inline-comment anchors, and, after an earlier
  AI review, the diff since that review.
* ``extract`` (no GitHub token): validates the JSON answer of Bob Shell.
* ``publish`` (write token, no LLM): posts one ``COMMENT`` review. It never
  approves or requests changes. Findings on lines outside the diff go into the
  review body.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ai_pr_triage import (
    DIFF_EXCLUDED_PATHS,
    TriageContext,
    build_diff,
    extract_answer,
    fetch_pr_commits,
    matches_any,
    gh_api,
    gh_write,
    pr_files,
    read_blob,
    sanitize_text,
)

REVIEW_MARKER_PREFIX = "<!-- ai-pr-review sha="
REVIEW_MARKER = re.compile(r"<!-- ai-pr-review sha=([0-9a-f]{40}) -->")
LABEL_REVIEW_LGTM = "ai:review-lgtm"
LABEL_REVIEW_CHANGES = "ai:review-changes"
REVIEW_LABELS = (LABEL_REVIEW_LGTM, LABEL_REVIEW_CHANGES)

MAX_REVIEW_CHANGED_LINES = 2_000
MAX_REVIEW_FILES = 60
MAX_HEAD_FILE_BYTES = 200_000
MAX_INCREMENTAL_DIFF_CHARS = 80_000
MAX_PREVIOUS_REVIEW_CHARS = 10_000
MAX_FINDINGS = 15
MAX_FINDING_CHARS = 2_000
SEVERITIES = ("blocker", "major", "minor", "nit")
BLOCKING_SEVERITIES = frozenset({"blocker", "major"})
REVIEW_VERDICTS = frozenset({"approve", "changes-requested"})
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(slots=True)
class Finding:
    path: str
    line: int
    severity: str
    title: str
    body: str


@dataclass(slots=True)
class ReviewResult:
    summary: str
    verdict: str
    findings: list[Finding]


def right_side_lines(patch: str | None) -> list[int]:
    """Return the head-side line numbers that GitHub accepts for a comment."""
    lines: list[int] = []
    current = 0
    for line in (patch or "").splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            current = int(header.group(1))
            continue
        if line.startswith("-") or line.startswith("\\"):
            continue
        lines.append(current)
        current += 1
    return lines


def stored_name(path: str) -> str:
    """Map a repository path to a name that no tool loads as configuration.

    Leading-dot segments (``.bob``, ``.github``) are renamed, and every file
    gets a ``.txt`` suffix, so ``AGENTS.md`` or ``.bob/custom_modes.yaml`` from
    the PR are never discovered as agent instructions.
    """
    parts = [
        f"dot{part}" if part.startswith(".") else part
        for part in PurePosixPath(path).parts
    ]
    return str(PurePosixPath(*parts)) + ".txt"


def review_scope(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in files
        if not item["filename"].startswith("tests/data/")
        and not matches_any(item["filename"], DIFF_EXCLUDED_PATHS)
    ]


def is_reviewable(files: list[dict[str, Any]]) -> bool:
    scope = review_scope(files)
    changed = sum(int(item.get("changes", 0)) for item in scope)
    return (
        bool(scope)
        and len(scope) <= MAX_REVIEW_FILES
        and (changed <= MAX_REVIEW_CHANGED_LINES)
    )


def _model_text(data: dict[str, Any], key: str, limit: int) -> str:
    value = data.get(key, "")
    return sanitize_text(value if isinstance(value, str) else str(value), limit)


def sanitize_markdown(text: str, limit: int = MAX_FINDING_CHARS) -> str:
    """Keep line breaks and code, but drop HTML, comments, and mentions."""
    text = "".join(ch if ch.isprintable() or ch == "\n" else " " for ch in text)
    text = re.sub(r"<(?=[A-Za-z/!?])", "&lt;", text)
    text = re.sub(r"@(?=[\w-])", "@​", text)
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def parse_review_result(data: Any, changed_paths: set[str]) -> ReviewResult:
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
    verdict = data.get("verdict")
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"invalid review verdict: {verdict!r}")
    raw_findings = data.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("findings must be a list")
    findings: list[Finding] = []
    for item in raw_findings[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            raise ValueError("each finding must be an object")
        path, line, severity = item.get("path"), item.get("line"), item.get("severity")
        if path not in changed_paths:
            raise ValueError(f"finding names a file outside the PR: {path!r}")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise ValueError(f"invalid line for {path}: {line!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"invalid severity: {severity!r}")
        findings.append(
            Finding(
                path=path,
                line=line,
                severity=severity,
                title=_model_text(item, "title", 200),
                body=sanitize_markdown(str(item.get("body", ""))),
            )
        )
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return ReviewResult(
        summary=sanitize_markdown(str(data.get("summary", "")), 1_500),
        verdict=verdict,
        findings=findings,
    )


def review_label(result: ReviewResult) -> str:
    blocking = any(f.severity in BLOCKING_SEVERITIES for f in result.findings)
    if result.verdict == "approve" and not blocking:
        return LABEL_REVIEW_LGTM
    return LABEL_REVIEW_CHANGES


def build_review(
    result: ReviewResult,
    anchors: dict[str, list[int]],
    head_sha: str,
    incremental_base: str | None,
) -> dict[str, Any]:
    comments: list[dict[str, Any]] = []
    unanchored: list[Finding] = []
    for finding in result.findings:
        if finding.line in anchors.get(finding.path, []):
            comments.append(
                {
                    "path": finding.path,
                    "line": finding.line,
                    "side": "RIGHT",
                    "body": f"**{finding.severity}: {finding.title}**\n\n{finding.body}",
                }
            )
        else:
            unanchored.append(finding)

    label = review_label(result)
    outcome = (
        "no blocking issue found"
        if label == LABEL_REVIEW_LGTM
        else "changes are suggested"
    )
    scope = (
        f"the changes since `{incremental_base[:10]}`"
        if incremental_base
        else "the full diff"
    )
    lines = [
        f"{REVIEW_MARKER_PREFIX}{head_sha} -->",
        f"### AI first review (advisory): {outcome}",
        "",
        f"This review covers {scope} at `{head_sha[:10]}`. It read the code only."
        " It did not run tests or code from this pull request.",
        "",
    ]
    if result.summary:
        lines += [result.summary, ""]
    if unanchored:
        lines += ["**Findings outside the diff lines**", ""]
        for finding in unanchored:
            body = finding.body.replace("\n", "\n  ")
            lines.append(
                f"- **{finding.severity}** `{finding.path}:{finding.line}`:"
                f" **{finding.title}**\n  {body}"
            )
        lines.append("")
    lines.append(
        "<sub>A maintainer makes the merge decision. This review never approves"
        " the pull request.</sub>"
    )
    return {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": "\n".join(lines),
        "comments": comments,
    }


# ---------------------------------------------------------------------------
# Jobs


def last_reviewed_sha(repo: str, pr_number: int) -> str | None:
    reviews = gh_api(
        f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100", paginate=True
    )
    for review in reversed(reviews):
        if review["user"]["login"] != "github-actions[bot]":
            continue
        match = REVIEW_MARKER.search(review.get("body") or "")
        if match:
            return match.group(1)
    return None


def previous_review_text(repo: str, pr_number: int) -> str:
    comments = gh_api(
        f"repos/{repo}/pulls/{pr_number}/comments?per_page=100", paginate=True
    )
    parts = [
        f"- `{comment['path']}:{comment.get('line') or comment.get('original_line')}`:"
        f" {sanitize_text(comment['body'], 400)}"
        for comment in comments
        if comment["user"]["login"] == "github-actions[bot]"
    ]
    return "\n".join(parts)[:MAX_PREVIOUS_REVIEW_CHARS]


def prepare(repo: str, context_dir: Path, git_dir: Path) -> bool:
    context = TriageContext.from_dict(
        json.loads((context_dir / "context.json").read_text("utf-8"))
    )
    pr = context.pr
    files = pr_files(repo, pr.number)
    if not is_reviewable(files):
        print("The PR is too large or has no source changes. No AI review.")
        return False

    review_dir = context_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    anchors = {item["filename"]: right_side_lines(item.get("patch")) for item in files}
    (review_dir / "anchors.json").write_text(json.dumps(anchors), encoding="utf-8")

    reviewed = last_reviewed_sha(repo, pr.number)
    if reviewed == pr.head_sha:
        print("The AI already reviewed this commit.")
        return False
    incremental_base: str | None = None
    if reviewed is not None:
        compare = gh_api(f"repos/{repo}/compare/{reviewed}...{pr.head_sha}")
        if compare.get("status") == "ahead":
            incremental_base = reviewed
            diff, _ = build_diff(compare.get("files") or [], MAX_INCREMENTAL_DIFF_CHARS)
            (review_dir / "incremental.diff").write_text(diff, encoding="utf-8")
            (review_dir / "previous-review.md").write_text(
                previous_review_text(repo, pr.number), encoding="utf-8"
            )

    stored: dict[str, str] = {}
    if fetch_pr_commits(git_dir, pr.number, pr.merge_base_sha):
        for item in review_scope(files):
            if item["status"] == "removed":
                continue
            content = read_blob(git_dir, pr.head_sha, item["filename"])
            if content is None or len(content) > MAX_HEAD_FILE_BYTES:
                continue
            name = stored_name(item["filename"])
            target = review_dir / "files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            stored[item["filename"]] = f".ai-triage/review/files/{name}"

    meta = {"incremental_base": incremental_base, "head_files": stored}
    (review_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def should_review(result_path: Path) -> bool:
    """Skip the review when the triage rated the PR as a duplicate."""
    if not result_path.is_file():
        return True
    data = json.loads(result_path.read_text("utf-8"))
    return not any(d.get("verdict") == "duplicate" for d in data.get("duplicates", []))


def extract(bob_output: Path, context_dir: Path, out_path: Path) -> None:
    anchors = json.loads((context_dir / "review/anchors.json").read_text("utf-8"))
    output = json.loads(bob_output.read_text("utf-8"))
    print(
        f"Bob Shell status: {output.get('status')};"
        f" stats: {json.dumps(output.get('stats'))}"
    )
    last_message = output.get("last_message")
    if not isinstance(last_message, str):
        raise ValueError("Bob Shell output has no final message")
    result = parse_review_result(extract_answer(last_message), set(anchors))
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def publish(repo: str, pr_number: int, context_dir: Path, result_path: Path) -> None:
    context = TriageContext.from_dict(
        json.loads((context_dir / "context.json").read_text("utf-8"))
    )
    pr = gh_api(f"repos/{repo}/pulls/{pr_number}")
    if pr["head"]["sha"] != context.pr.head_sha:
        print("The PR has a newer commit. The newer run publishes the review.")
        return
    if not result_path.is_file():
        print("No valid review result. Nothing to publish.")
        return

    anchors = json.loads((context_dir / "review/anchors.json").read_text("utf-8"))
    meta = json.loads((context_dir / "review/meta.json").read_text("utf-8"))
    data = json.loads(result_path.read_text("utf-8"))
    # Validate again: the result artifact comes from the job that ran the model.
    answer = {
        "summary": data["summary"],
        "verdict": data["verdict"],
        "findings": data["findings"],
    }
    result = parse_review_result(answer, set(anchors))

    review = build_review(
        result, anchors, context.pr.head_sha, meta["incremental_base"]
    )
    gh_write("POST", f"repos/{repo}/pulls/{pr_number}/reviews", review)

    label = review_label(result)
    current = {item["name"] for item in pr["labels"]}
    for stale in sorted((current & set(REVIEW_LABELS)) - {label}):
        gh_write("DELETE", f"repos/{repo}/issues/{pr_number}/labels/{stale}")
    if label not in current:
        gh_write("POST", f"repos/{repo}/issues/{pr_number}/labels", {"labels": [label]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", required=True)
    prep.add_argument("--context-dir", type=Path, required=True)
    prep.add_argument("--git-dir", type=Path, default=Path.cwd())

    gate = sub.add_parser("gate")
    gate.add_argument("--triage-result", type=Path, required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--bob-output", type=Path, required=True)
    ext.add_argument("--context-dir", type=Path, required=True)
    ext.add_argument("--out", type=Path, required=True)

    pub = sub.add_parser("publish")
    pub.add_argument("--repo", required=True)
    pub.add_argument("--pr", type=int, required=True)
    pub.add_argument("--context-dir", type=Path, required=True)
    pub.add_argument("--result", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        run_review = prepare(args.repo, args.context_dir, args.git_dir)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"run_review={'true' if run_review else 'false'}\n")
    elif args.command == "gate":
        return 0 if should_review(args.triage_result) else 1
    elif args.command == "extract":
        try:
            extract(args.bob_output, args.context_dir, args.out)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"::warning::The Bob Shell review answer is not valid: {exc}")
            return 1
    else:
        publish(args.repo, args.pr, args.context_dir, args.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
