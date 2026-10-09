# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Advisory first review of a pull request with inline comments.

This extends the AI triage workflow. The same trust rules apply:

* ``prepare`` (read-only token): writes the PR head versions of changed source
  files as plain text, the valid inline-comment anchors, the discussion on the
  PR, and, after an earlier AI review, the diff since that review.
* ``extract`` (no GitHub token): validates the JSON answer of Bob Shell and
  moves each finding to the line that its quote names.
* ``publish`` (write token, no LLM): posts one ``COMMENT`` review. It never
  approves or requests changes. Findings on lines outside the diff go into the
  review body.

Without the ``/ai review`` command, a PR gets at most
``MAX_AUTOMATIC_REVIEWS`` AI reviews: the first review and one review of the
next push. More rounds on the same lines found smaller and smaller points, and
contributors spent time on each of them.
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

from ai_pr_event import is_maintainer_activity, parse_command
from ai_pr_triage import (
    COMMENT_MARKER,
    DIFF_EXCLUDED_PATHS,
    ZERO_WIDTH_SPACE,
    TriageContext,
    build_diff,
    clean_text,
    extract_answer,
    fetch_pr_commits,
    gh_api,
    gh_write,
    markdown_code,
    markdown_text,
    matches_any,
    pr_files,
    read_blob,
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
MAX_DISCUSSION_CHARS = 30_000
MAX_DISCUSSION_ITEM_CHARS = 800
MAX_AUTOMATIC_REVIEWS = 2
MAX_FINDINGS = 8
MAX_EARLIER_FINDINGS = 20
MAX_FINDING_CHARS = 2_000
MAX_QUOTE_CHARS = 300
# The severity terms of `.agents/skills/review/SKILL.md`, in report order.
SEVERITIES = ("blocker", "question", "suggestion")
EARLIER_STATUSES = ("open", "fixed", "answered")
BOT_LOGIN = "github-actions[bot]"
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(slots=True)
class Finding:
    path: str
    line: int
    severity: str
    title: str
    body: str
    quote: str = ""


@dataclass(slots=True)
class EarlierFinding:
    title: str
    severity: str
    status: str


@dataclass(slots=True)
class ReviewResult:
    summary: str
    findings: list[Finding]
    earlier_findings: list[EarlierFinding]


def right_side_lines(patch: str | None) -> list[int]:
    """Return the head-side line numbers that GitHub accepts for a comment."""
    lines: list[int] = []
    current = 0
    for line in (patch or "").splitlines():
        header = HUNK_HEADER.match(line)
        if header:
            current = int(header.group(1))
            continue
        if line.startswith(("-", "\\")):
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
    return clean_text(value if isinstance(value, str) else str(value), limit)


def _closing_run(text: str, start: int, stop: int, length: int) -> int | None:
    """Return the end of the next run of exactly ``length`` backticks."""
    index = start
    while index < stop:
        if text[index] != "`":
            index += 1
            continue
        end = index
        while end < stop and text[end] == "`":
            end += 1
        if end - index == length:
            return end
        index = end
    return None


def code_segments(text: str) -> list[tuple[bool, str]]:
    """Split Markdown into text and code, as ``(is_code, part)`` pairs.

    The split is conservative. A part is code only when CommonMark also reads
    it as code: a fenced block (open to the end when it is not closed), or a
    code span that closes on the same line. Text that CommonMark reads as code
    but this function does not only gets escaped once too often.
    """
    segments: list[tuple[bool, str]] = []
    start = index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char != "`":
            index += 1
            continue
        run_end = index
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        run = run_end - index
        line_start = text.rfind("\n", 0, index) + 1
        line_end = text.find("\n", run_end)
        line_end = len(text) if line_end < 0 else line_end
        is_fence = (
            run >= 3
            and index - line_start <= 3
            and not text[line_start:index].strip(" ")
            and "`" not in text[run_end:line_end]
        )
        end: int | None
        if is_fence:
            closing = re.compile(rf"^ {{0,3}}`{{{run},}}[ \t]*$", re.MULTILINE)
            match = closing.search(text, line_end)
            end = match.end() if match else len(text)
        else:
            end = _closing_run(text, run_end, line_end, run)
        if end is None:
            index = run_end
            continue
        segments.append((False, text[start:index]))
        segments.append((True, text[index:end]))
        start = index = end
    segments.append((False, text[start:]))
    return [(is_code, part) for is_code, part in segments if part]


def sanitize_markdown(text: str, limit: int = MAX_FINDING_CHARS) -> str:
    """Keep line breaks and code, but drop HTML, comments, and mentions.

    A finding body keeps its Markdown, so that it can quote code in a fenced
    block or a suggestion block. So it can also contain a link. This is
    accepted: the body is an inline comment on a changed line, or a list item
    in the review body. The list item ends before the closing note, and so does
    a code block that the body leaves open. Code stays as it is: GitHub shows
    HTML and mentions in code as text, and an escape there would change the
    code of a suggestion.
    """
    text = "".join(ch if ch.isprintable() or ch == "\n" else " " for ch in text)
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    parts = []
    for is_code, part in code_segments(text):
        if not is_code:
            part = re.sub(r"<(?=[A-Za-z/!?])", "&lt;", part)
            part = re.sub(r"@(?=[\w-])", f"@{ZERO_WIDTH_SPACE}", part)
        parts.append(part)
    return "".join(parts)


def _markdown_text_part(part: str) -> str:
    """Escape a text part and keep its spaces next to the code spans."""
    stripped = part.strip()
    if not stripped:
        return part
    lead = part[: len(part) - len(part.lstrip())]
    trail = part[len(part.rstrip()) :]
    return f"{lead}{markdown_text(stripped)}{trail}"


def markdown_inline(text: str) -> str:
    """Render one line of untrusted text as literal text with code spans."""
    return "".join(
        markdown_code(part.strip("`").strip()) if is_code else _markdown_text_part(part)
        for is_code, part in code_segments(clean_text(text, limit=len(text)))
    )


def resolve_line(
    source: list[str], quote: str, hint: int, anchors: list[int]
) -> int | None:
    """Return the line of ``source`` that ``quote`` names, or None.

    The model often gives a wrong line number, but it can copy the code line.
    An exact line wins over a line that only contains the quote. Among equal
    matches, a line in the diff wins, then the line closest to ``hint``.
    """
    needle = next((line.strip() for line in quote.splitlines() if line.strip()), "")
    if not needle:
        return None
    exact = [n for n, line in enumerate(source, 1) if line.strip() == needle]
    matches = exact or [n for n, line in enumerate(source, 1) if needle in line]
    if not matches:
        return None
    anchored = set(anchors)
    return min(matches, key=lambda n: (n not in anchored, abs(n - hint)))


def parse_review_result(data: Any, changed_paths: set[str]) -> ReviewResult:
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")
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
                quote=str(item.get("quote", ""))[:MAX_QUOTE_CHARS],
            )
        )
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    raw_earlier = data.get("earlier_findings", [])
    if not isinstance(raw_earlier, list):
        raise ValueError("earlier_findings must be a list")
    earlier: list[EarlierFinding] = []
    for item in raw_earlier[:MAX_EARLIER_FINDINGS]:
        if not isinstance(item, dict):
            raise ValueError("each earlier finding must be an object")
        severity, status = item.get("severity"), item.get("status")
        if severity not in SEVERITIES:
            raise ValueError(f"invalid earlier severity: {severity!r}")
        if status not in EARLIER_STATUSES:
            raise ValueError(f"invalid earlier status: {status!r}")
        earlier.append(
            EarlierFinding(_model_text(item, "title", 200), severity, status)
        )
    return ReviewResult(
        summary=clean_text(str(data.get("summary", "")), 1_000),
        findings=findings,
        earlier_findings=earlier,
    )


def resolve_anchors(
    result: ReviewResult, sources: dict[str, list[str]], anchors: dict[str, list[int]]
) -> ReviewResult:
    """Move each finding to the line of its quote. Drop a finding whose quote
    is not in the file: it describes code that the PR does not have.

    A finding keeps its line when no copy of the file exists (a large file).
    """
    kept: list[Finding] = []
    for finding in result.findings:
        source = sources.get(finding.path)
        if source is None:
            kept.append(finding)
            continue
        line = resolve_line(
            source, finding.quote, finding.line, anchors.get(finding.path, [])
        )
        if line is None:
            print(
                f"::warning::Dropped a finding: its quote is not in {finding.path}:"
                f" {clean_text(finding.title, 120)}"
            )
            continue
        finding.line = line
        kept.append(finding)
    result.findings = kept
    return result


def review_label(result: ReviewResult) -> str:
    """Only a blocker sets the changes label. A suggestion or a question never
    does, also in a later round, so the label does not flip on small points."""
    blocking = any(f.severity == "blocker" for f in result.findings) or any(
        f.severity == "blocker" and f.status == "open" for f in result.earlier_findings
    )
    return LABEL_REVIEW_CHANGES if blocking else LABEL_REVIEW_LGTM


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
                    "body": (
                        f"**{finding.severity}: {markdown_inline(finding.title)}**"
                        f"\n\n{finding.body}"
                    ),
                }
            )
        else:
            unanchored.append(finding)

    label = review_label(result)
    outcome = "no blocker found" if label == LABEL_REVIEW_LGTM else "blocker found"
    scope = (
        f"the changes since `{incremental_base[:10]}`"
        if incremental_base
        else "the full diff"
    )
    lines = [
        f"{REVIEW_MARKER_PREFIX}{head_sha} -->",
        f"### AI first review (advisory): {outcome}",
        "",
        f"This review covers {scope} at `{head_sha[:10]}`. It read the code and"
        " did not run it.",
        "",
    ]
    if result.summary:
        lines += [markdown_inline(result.summary), ""]
    if result.earlier_findings:
        lines += ["**Earlier findings**", ""]
        lines += [
            f"- {item.status} ({item.severity}): {markdown_inline(item.title)}"
            for item in result.earlier_findings
        ]
        lines.append("")
    if unanchored:
        lines += ["**Findings outside the diff lines**", ""]
        for finding in unanchored:
            body = finding.body.replace("\n", "\n  ")
            lines.append(
                f"- **{finding.severity}**"
                f" {markdown_code(f'{finding.path}:{finding.line}')}:"
                f" **{markdown_inline(finding.title)}**\n  {body}"
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


def ai_reviewed_shas(reviews: list[dict[str, Any]]) -> list[str]:
    """Return the commits of the earlier AI reviews, oldest first."""
    shas: list[str] = []
    for review in reviews:
        if review["user"]["login"] != BOT_LOGIN:
            continue
        match = REVIEW_MARKER.search(review.get("body") or "")
        if match:
            shas.append(match.group(1))
    return shas


def _role(item: dict[str, Any], pr_author: str) -> str:
    login = item["user"]["login"]
    if login == BOT_LOGIN:
        return "AI review"
    if login == pr_author:
        return "PR author"
    association = str(item.get("author_association", "NONE")).lower()
    return "maintainer" if is_maintainer_activity(item, pr_author) else association


def _entry(item: dict[str, Any], pr_author: str) -> str:
    login = clean_text(item["user"]["login"], 60)
    body = clean_text(item.get("body") or "", MAX_DISCUSSION_ITEM_CHARS)
    return f"- {login} ({_role(item, pr_author)}): {body}"


def discussion_text(
    pr_author: str,
    issue_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
) -> str:
    """Write the PR discussion as Markdown for the model: each inline thread
    with its replies, then the review bodies and the PR comments.

    The model needs the replies to know which earlier findings the author
    fixed or answered, and which points a maintainer already made.
    """
    threads: dict[int, list[dict[str, Any]]] = {}
    for comment in sorted(review_comments, key=lambda c: c["id"]):
        root = comment.get("in_reply_to_id") or comment["id"]
        threads.setdefault(root, []).append(comment)
    lines = ["# Inline review threads", ""]
    for thread in threads.values():
        first = thread[0]
        location = f"{first['path']}:{first.get('line') or first.get('original_line')}"
        lines.append(f"## `{clean_text(location, 200)}`")
        lines += [_entry(comment, pr_author) for comment in thread]
        lines.append("")
    lines += ["# Reviews and PR comments", ""]
    others = [r for r in reviews if (r.get("body") or "").strip()] + [
        c
        for c in issue_comments
        if COMMENT_MARKER not in (c.get("body") or "")
        and parse_command(c.get("body") or "") is None
    ]
    others.sort(
        key=lambda item: item.get("submitted_at") or item.get("created_at") or ""
    )
    lines += [_entry(item, pr_author) for item in others]
    return "\n".join(lines)[:MAX_DISCUSSION_CHARS]


def prepare(repo: str, context_dir: Path, git_dir: Path, force: bool) -> bool:
    """Write the review inputs. Return False when no review should run.

    Without ``force`` (a push), the review is skipped after a maintainer got
    involved, for a commit that the AI already reviewed, and after
    ``MAX_AUTOMATIC_REVIEWS`` AI reviews. ``force`` (the ``/ai review``
    command) always runs a full review of the current commit.
    """
    context = TriageContext.from_dict(
        json.loads((context_dir / "context.json").read_text("utf-8"))
    )
    pr = context.pr
    issue_comments = gh_api(
        f"repos/{repo}/issues/{pr.number}/comments?per_page=100", paginate=True
    )
    reviews = gh_api(
        f"repos/{repo}/pulls/{pr.number}/reviews?per_page=100", paginate=True
    )
    review_comments = gh_api(
        f"repos/{repo}/pulls/{pr.number}/comments?per_page=100", paginate=True
    )
    if not force and any(
        is_maintainer_activity(item, pr.author)
        for item in (*issue_comments, *reviews, *review_comments)
    ):
        print("A maintainer is involved. Use the /ai review command for a review.")
        return False
    files = pr_files(repo, pr.number)
    if not is_reviewable(files):
        print("The PR is too large or has no source changes. No AI review.")
        return False

    reviewed_shas = ai_reviewed_shas(reviews)
    reviewed = None if force else (reviewed_shas[-1] if reviewed_shas else None)
    if reviewed == pr.head_sha:
        print("The AI already reviewed this commit.")
        return False
    if not force and len(reviewed_shas) >= MAX_AUTOMATIC_REVIEWS:
        print("The PR has enough AI reviews. Use the /ai review command for more.")
        return False

    review_dir = context_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    anchors = {item["filename"]: right_side_lines(item.get("patch")) for item in files}
    (review_dir / "anchors.json").write_text(json.dumps(anchors), encoding="utf-8")
    (review_dir / "discussion.md").write_text(
        discussion_text(pr.author, issue_comments, reviews, review_comments),
        encoding="utf-8",
    )

    incremental_base: str | None = None
    if reviewed is not None:
        compare = gh_api(f"repos/{repo}/compare/{reviewed}...{pr.head_sha}")
        if compare.get("status") == "ahead":
            incremental_base = reviewed
            diff, _ = build_diff(compare.get("files") or [], MAX_INCREMENTAL_DIFF_CHARS)
            (review_dir / "incremental.diff").write_text(diff, encoding="utf-8")

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


def should_review(result_path: Path, context_path: Path) -> bool:
    """Skip the review when the triage rated the PR as a duplicate of an open
    PR from another author.

    A closed PR is not a reason to skip. The same author often opens a new PR
    that replaces an earlier one and then closes the earlier one.
    """
    if not result_path.is_file():
        return True
    data = json.loads(result_path.read_text("utf-8"))
    duplicates = {
        d.get("pr")
        for d in data.get("duplicates", [])
        if d.get("verdict") == "duplicate"
    }
    if not duplicates:
        return True
    if not context_path.is_file():
        return False
    context = TriageContext.from_dict(json.loads(context_path.read_text("utf-8")))
    return not any(
        candidate.number in duplicates
        and candidate.state == "open"
        and candidate.author != context.pr.author
        for candidate in context.candidates
    )


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
    meta = json.loads((context_dir / "review/meta.json").read_text("utf-8"))
    sources = {
        path: (context_dir / "review/files" / stored_name(path))
        .read_text("utf-8", errors="replace")
        .splitlines()
        for path in meta["head_files"]
    }
    result = resolve_anchors(result, sources, anchors)
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
        "findings": data["findings"],
        "earlier_findings": data["earlier_findings"],
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
    prep.add_argument(
        "--force", action="store_true", help="Full review (the /ai review command)."
    )

    gate = sub.add_parser("gate")
    gate.add_argument("--triage-result", type=Path, required=True)
    gate.add_argument("--context", type=Path, required=True)

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
        run_review = prepare(args.repo, args.context_dir, args.git_dir, args.force)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"run_review={'true' if run_review else 'false'}\n")
    elif args.command == "gate":
        return 0 if should_review(args.triage_result, args.context) else 1
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
