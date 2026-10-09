# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "ai_pr_review", SCRIPTS_DIR / "ai_pr_review.py"
)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)

HEAD = "f" * 40
PATCH = "@@ -10,3 +10,4 @@ def f():\n context\n-old\n+new\n+added\n context\n"


def finding(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "path": "docling/a.py",
        "line": 11,
        "quote": "new",
        "severity": "blocker",
        "title": "Wrong value",
        "body": "Use `x`.",
    }
    data.update(overrides)
    return data


def answer(*findings: dict[str, object], **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "summary": "",
        "findings": list(findings),
        "earlier_findings": [],
    }
    data.update(overrides)
    return data


def test_comment_anchors_are_head_lines_of_the_patch() -> None:
    assert review.right_side_lines(PATCH) == [10, 11, 12, 13]
    assert review.right_side_lines(None) == []


def test_stored_copies_cannot_be_loaded_as_agent_configuration() -> None:
    assert review.stored_name("AGENTS.md") == "AGENTS.md.txt"
    assert (
        review.stored_name(".bob/custom_modes.yaml") == "dot.bob/custom_modes.yaml.txt"
    )
    assert review.stored_name("a/.agents/x.md") == "a/dot.agents/x.md.txt"


def test_large_or_data_only_prs_are_not_reviewed() -> None:
    source = {"filename": "docling/a.py", "changes": 10}
    data_only = {"filename": "tests/data/groundtruth/a.json", "changes": 9000}
    assert review.is_reviewable([source, data_only])
    assert not review.is_reviewable([data_only])
    assert not review.is_reviewable([{**source, "changes": 5000}])


def test_findings_outside_the_pr_or_with_unknown_severity_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside the PR"):
        review.parse_review_result(
            answer(finding(path="docling/b.py")), {"docling/a.py"}
        )
    with pytest.raises(ValueError, match="severity"):
        review.parse_review_result(answer(finding(severity="major")), {"docling/a.py"})
    with pytest.raises(ValueError, match="status"):
        review.parse_review_result(
            answer(
                earlier_findings=[
                    {"title": "x", "severity": "blocker", "status": "maybe"}
                ]
            ),
            {"docling/a.py"},
        )


def test_only_an_open_blocker_sets_the_changes_label() -> None:
    def label(data: dict[str, object]) -> str:
        return review.review_label(review.parse_review_result(data, {"docling/a.py"}))

    assert label(answer(finding())) == review.LABEL_REVIEW_CHANGES
    assert (
        label(answer(finding(severity="question"), finding(severity="suggestion")))
        == review.LABEL_REVIEW_LGTM
    )
    earlier = {"title": "Old", "severity": "blocker", "status": "open"}
    assert label(answer(earlier_findings=[earlier])) == review.LABEL_REVIEW_CHANGES
    for status in ("fixed", "answered"):
        assert (
            label(answer(earlier_findings=[{**earlier, "status": status}]))
            == review.LABEL_REVIEW_LGTM
        )


def test_quote_moves_a_finding_to_its_line_or_drops_it() -> None:
    source = ["def f():", "    x = 1", "    return int(attr)", "    y = int(attr)"]
    result = review.parse_review_result(
        answer(
            finding(line=40, quote="  return int(attr)  "),
            finding(line=1, quote="int(attr)"),
            finding(line=2, quote="int(other)", title="Invented"),
            finding(path="docling/big.py", line=7, quote="anything"),
        ),
        {"docling/a.py", "docling/big.py"},
    )
    resolved = review.resolve_anchors(
        result, {"docling/a.py": source}, {"docling/a.py": [4]}
    )
    # Exact line first, then a line in the diff, then the line nearest the hint.
    # A file without a stored copy keeps the line of the model.
    assert [(f.path, f.line) for f in resolved.findings] == [
        ("docling/a.py", 3),
        ("docling/a.py", 4),
        ("docling/big.py", 7),
    ]


def test_code_keeps_html_and_mentions_and_text_does_not() -> None:
    body = review.sanitize_markdown(
        "Use `<ol start>` here, @team <b>x</b>.\n"
        "```suggestion\n@staticmethod\nhtml = b'<html>'\n```\n"
        "``<img src=x>`"
    )
    assert "`<ol start>`" in body
    assert "@staticmethod\nhtml = b'<html>'" in body
    assert "@\u200bteam &lt;b>x&lt;/b>" in body
    # Two backticks need two backticks to close, so CommonMark shows this as
    # text with an HTML tag. It must be escaped.
    assert body.endswith("``&lt;img src=x>`")
    # A fence that is not closed is code to the end of the body.
    assert review.sanitize_markdown("```\n<b>") == "```\n<b>"


def test_titles_keep_code_spans() -> None:
    assert (
        review.markdown_inline("Call `detect()` with [x](y)")
        == "Call `detect()` with \\[x\\]\\(y\\)"
    )


def test_review_is_a_comment_and_moves_unanchored_findings_to_the_body() -> None:
    result = review.parse_review_result(
        answer(
            finding(severity="suggestion"),
            finding(line=99, title="Far away"),
            summary="Ping @team <!-- ai-pr-review sha=" + "0" * 40 + " -->",
            earlier_findings=[
                {"title": "Old `x`", "severity": "blocker", "status": "fixed"}
            ],
        ),
        {"docling/a.py"},
    )
    payload = review.build_review(
        result, {"docling/a.py": review.right_side_lines(PATCH)}, HEAD, None
    )
    assert payload["event"] == "COMMENT"
    assert [(c["path"], c["line"]) for c in payload["comments"]] == [
        ("docling/a.py", 11)
    ]
    body = payload["body"]
    assert body.startswith(f"<!-- ai-pr-review sha={HEAD} -->")
    assert "`docling/a.py:99`" in body
    assert "- fixed (blocker): Old `x`" in body
    assert review.REVIEW_MARKER.findall(body) == [HEAD]
    assert "@\u200bteam" in body


def user(login: str) -> dict[str, object]:
    return {"login": login, "type": "Bot" if "[bot]" in login else "User"}


def test_discussion_groups_replies_under_their_finding() -> None:
    def comment(cid: int, login: str, body: str, **extra: object) -> dict[str, object]:
        return {
            "id": cid,
            "user": user(login),
            "author_association": extra.pop("association", "NONE"),
            "body": body,
            "path": "docling/a.py",
            "line": 11,
            **extra,
        }

    text = review.discussion_text(
        "alice",
        issue_comments=[
            comment(10, "github-actions[bot]", "<!-- ai-pr-triage --> triage"),
            comment(11, "bob", "/ai review", association="MEMBER"),
            comment(12, "bob", "Please split this PR.", association="MEMBER"),
        ],
        reviews=[],
        review_comments=[
            comment(1, "github-actions[bot]", "**blocker: Wrong value**"),
            comment(2, "carol", "Unrelated point", line=40),
            comment(3, "alice", "Fixed in abc.", in_reply_to_id=1),
        ],
    )
    thread = text.index("- github-actions[bot] (AI review): **blocker: Wrong value**")
    assert (
        text.index("- alice (PR author): Fixed in abc.") == text.index("\n", thread) + 1
    )
    assert "- bob (maintainer): Please split this PR." in text
    assert "triage" not in text
    assert "/ai review" not in text


def test_duplicates_of_an_open_pr_from_another_author_skip_the_review(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    context_path = tmp_path / "context.json"
    assert review.should_review(result, context_path)

    def write(state: str, author: str) -> None:
        context_path.write_text(
            json.dumps(
                {
                    "pr": {
                        "number": 2,
                        "title": "",
                        "body": "",
                        "author": "alice",
                        "author_association": "NONE",
                        "base_sha": HEAD,
                        "head_sha": HEAD,
                        "merge_base_sha": HEAD,
                        "changed_files": [],
                        "additions": 0,
                        "deletions": 0,
                    },
                    "issue_refs": [],
                    "candidates": [
                        {
                            "number": 1,
                            "title": "",
                            "state": state,
                            "url": "",
                            "author": author,
                        }
                    ],
                    "risk": {"forced": [], "hints": []},
                    "groundtruth_markdown": "",
                    "diff_truncated": False,
                    "topic_labels": [],
                }
            )
        )

    result.write_text(json.dumps({"duplicates": [{"pr": 1, "verdict": "duplicate"}]}))
    write("open", "bob")
    assert not review.should_review(result, context_path)
    # The author replaced an own PR, or the other PR is closed.
    write("open", "alice")
    assert review.should_review(result, context_path)
    write("closed", "bob")
    assert review.should_review(result, context_path)
    result.write_text(json.dumps({"duplicates": [{"pr": 1, "verdict": "related"}]}))
    write("open", "bob")
    assert review.should_review(result, context_path)
