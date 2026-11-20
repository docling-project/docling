# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Find out whose turn it is on a PR: the maintainers or the author.

``sync_prs.py`` calls ``fetch_turns``, prints the PRs where it is the turn of
a maintainer, and saves the turn of each PR in its ``pr.json``.

The rules:

- PRs from owners, members, collaborators, bots, and ``AUTOMATION_AUTHORS``
  are not tracked.
- Maintainer actions: reviews, review comments, and conversation comments
  from owners, members, and collaborators. Bot comments do not count.
- Author actions: commits, force pushes, comments, reviews, and review thread
  replies of the PR author. Merge commits and commits of a maintainer on the
  author's branch do not count.
- Comments from other contributors do not change whose turn it is.
- When the last maintainer action is an approval, it stays the turn of the
  maintainers: the PR waits for another approval or for the merge. The wait
  starts at the approval.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from gh_helpers import MAINTAINER_ROLES, age, graphql_prs, parse_time

# Accounts that open PRs for automation but are not GitHub bots.
AUTOMATION_AUTHORS = {"docling-ops"}
TURN_FIELDS = """
fragment TurnFields on PullRequest {
  number
  author { __typename login }
  authorAssociation
  timelineItems(last: 100, itemTypes: [ISSUE_COMMENT, PULL_REQUEST_REVIEW,
                PULL_REQUEST_COMMIT, HEAD_REF_FORCE_PUSHED_EVENT]) {
    nodes {
      __typename
      ... on IssueComment {
        createdAt authorAssociation author { __typename login }
      }
      ... on PullRequestReview {
        submittedAt state authorAssociation author { __typename login }
      }
      ... on PullRequestCommit {
        commit { committedDate parents { totalCount } author { user { login } } }
      }
      ... on HeadRefForcePushedEvent { createdAt actor { login } }
    }
  }
  reviewThreads(last: 50) {
    nodes { comments(last: 20) { nodes {
      createdAt authorAssociation author { __typename login }
    } } }
  }
}
"""
APPROVED_KIND = "review: approved"
# The timeline makes the query heavy; large batches time out (502).
BATCH_SIZE = 10


class Side(str, Enum):
    MAINTAINER = "maintainer"
    AUTHOR = "author"


class TurnState(str, Enum):
    FIRST_REVIEW = "waits for a first review"
    MAINTAINER = "maintainer's turn"
    AUTHOR = "author's turn"
    NOT_TRACKED = "not tracked"  # maintainer or bot PR


@dataclass(frozen=True, slots=True)
class Action:
    at: datetime
    side: Side
    actor: str
    kind: str  # "commit", "comment", "review: changes requested", ...


@dataclass(slots=True)
class Turn:
    state: TurnState
    last_maintainer: Action | None = None
    replies: list[Action] = field(default_factory=list)  # author, after it
    maintainers: list[str] = field(default_factory=list)  # all who acted
    # Maintainer's turn: the first author action after the last maintainer
    # action, or the approval if that action is an approval.
    waiting_since: datetime | None = None


def login(actor: dict[str, Any] | None) -> str:
    # The actor is None for deleted accounts.
    return str((actor or {}).get("login") or "ghost")


def is_bot(actor: dict[str, Any] | None) -> bool:
    return actor is not None and actor.get("__typename") == "Bot"


def classify(actor: dict[str, Any] | None, role: str, author: str) -> Side | None:
    """Return the side of a comment or review author, or None to ignore it."""
    if is_bot(actor):
        return None
    if login(actor) == author:
        return Side.AUTHOR
    if role in MAINTAINER_ROLES:
        return Side.MAINTAINER
    return None


def collect_actions(data: dict[str, Any]) -> list[Action]:
    author = login(data["author"])
    actions: list[Action] = []

    def add(at: str, side: Side | None, actor: str, kind: str) -> None:
        if side is not None:
            actions.append(Action(parse_time(at), side, actor, kind))

    for item in data["timelineItems"]["nodes"]:
        kind = item["__typename"]
        if kind == "IssueComment":
            side = classify(item["author"], item["authorAssociation"], author)
            add(item["createdAt"], side, login(item["author"]), "comment")
        elif kind == "PullRequestReview":
            # A pending review has no submittedAt; only its author sees it.
            if item["submittedAt"] is None:
                continue
            side = classify(item["author"], item["authorAssociation"], author)
            state = "review: " + item["state"].lower().replace("_", " ")
            text = state if side is Side.MAINTAINER else "review"
            add(item["submittedAt"], side, login(item["author"]), text)
        elif kind == "PullRequestCommit":
            commit = item["commit"]
            if commit["parents"]["totalCount"] > 1:
                continue  # merge commit: not a change of the author
            committer = login((commit["author"] or {}).get("user"))
            add(commit["committedDate"], Side.AUTHOR, committer, "commit")
        elif kind == "HeadRefForcePushedEvent" and login(item["actor"]) == author:
            add(item["createdAt"], Side.AUTHOR, author, "force push")
    for thread in data["reviewThreads"]["nodes"]:
        for item in thread["comments"]["nodes"]:
            side = classify(item["author"], item["authorAssociation"], author)
            text = "review comment" if side is Side.MAINTAINER else "thread reply"
            add(item["createdAt"], side, login(item["author"]), text)

    maintainers = {a.actor for a in actions if a.side is Side.MAINTAINER}
    # A maintainer who pushes to the author's branch does not hand the turn
    # back to the maintainers.
    actions = [
        a for a in actions if not (a.side is Side.AUTHOR and a.actor in maintainers)
    ]
    return sorted(actions, key=lambda a: a.at)


def compute_turn(data: dict[str, Any]) -> Turn:
    author = login(data["author"])
    if (
        is_bot(data["author"])
        or author in AUTOMATION_AUTHORS
        or data["authorAssociation"] in MAINTAINER_ROLES
    ):
        return Turn(TurnState.NOT_TRACKED)
    actions = collect_actions(data)
    by_maintainers = [a for a in actions if a.side is Side.MAINTAINER]
    if not by_maintainers:
        return Turn(TurnState.FIRST_REVIEW)
    last = by_maintainers[-1]
    replies = [a for a in actions if a.side is Side.AUTHOR and a.at > last.at]
    if replies:
        state, since = TurnState.MAINTAINER, replies[0].at
    elif last.kind == APPROVED_KIND:
        state, since = TurnState.MAINTAINER, last.at
    else:
        state, since = TurnState.AUTHOR, None
    return Turn(
        state=state,
        last_maintainer=last,
        replies=replies,
        maintainers=sorted({a.actor for a in by_maintainers}),
        waiting_since=since,
    )


def fetch_turns(repo: str, numbers: list[int]) -> dict[int, Turn]:
    """Read the timeline of the PRs and return whose turn it is on each."""
    nodes = graphql_prs(
        repo, numbers, TURN_FIELDS, "TurnFields", BATCH_SIZE, "Read timelines"
    )
    return {number: compute_turn(node) for number, node in nodes.items()}


def describe_replies(replies: list[Action]) -> str:
    counts = Counter(a.kind for a in replies)
    return ", ".join(
        f"{count} {kind}{('es' if kind.endswith('sh') else 's') if count > 1 else ''}"
        for kind, count in counts.items()
    )


def describe_turn(turn: Turn) -> str:
    """Describe whose turn it is, for a report line."""
    last = turn.last_maintainer
    if last is None:
        return turn.state.value
    if turn.waiting_since is None:
        # The author's turn: nothing happened after the maintainer action.
        return (
            f"{turn.state.value} · last maintainer action:"
            f" @{last.actor}, {last.kind}, {age(last.at)} ago"
        )
    text = (
        f"waits {age(turn.waiting_since)} · last maintainer action:"
        f" @{last.actor}, {last.kind}, {age(last.at)} ago"
    )
    if turn.replies:
        text += f" · since then: {describe_replies(turn.replies)}"
    return text
