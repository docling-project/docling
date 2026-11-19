# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Validated GitHub conversations and adapters for gh, REST, and GraphQL JSON."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, JsonValue

GitHubKind = Literal["issue", "pull_request", "discussion"]
GitHubEntryKind = Literal["comment", "review", "review_thread", "reply"]
MAX_GITHUB_JSON_BYTES = 32 * 1024 * 1024


class GitHubLocator(BaseModel):
    owner: str
    repository: str
    number: int
    kind: GitHubKind
    url: str


def parse_github_url(url: str) -> GitHubLocator:
    """Accept canonical github.com conversation URLs, with optional fragments."""
    parsed = urlsplit(url)
    match = re.fullmatch(
        r"/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/(issues|pull|discussions)/([1-9][0-9]*)/?",
        parsed.path,
    )
    if parsed.scheme != "https" or parsed.netloc != "github.com" or match is None:
        raise ValueError(
            "Expected https://github.com/OWNER/REPO/{issues,pull,discussions}/NUMBER"
        )
    owner, repository, category, number = match.groups()
    kinds: dict[str, GitHubKind] = {
        "issues": "issue",
        "pull": "pull_request",
        "discussions": "discussion",
    }
    return GitHubLocator(
        owner=owner,
        repository=repository,
        number=int(number),
        kind=kinds[category],
        url=f"https://github.com/{owner}/{repository}/{category}/{number}",
    )


def is_github_url(url: str) -> bool:
    try:
        parse_github_url(url)
    except ValueError:
        return False
    return True


class GitHubLabel(BaseModel):
    id: str | int | None = None
    name: str
    description: str | None = None
    color: str | None = None
    url: str | None = None


class GitHubEntry(BaseModel):
    id: str | int | None = None
    kind: GitHubEntryKind = "comment"
    body: str = ""
    url: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    parent_id: str | int | None = None
    state: str | None = None
    path: str | None = None
    line: int | None = None
    start_line: int | None = None
    original_line: int | None = None
    diff_hunk: str | None = None
    is_resolved: bool | None = None
    is_outdated: bool | None = None
    is_answer: bool = False
    replies: list[GitHubEntry] = Field(default_factory=list)


class GitHubConversation(BaseModel):
    source: GitHubLocator
    title: str
    body: str = ""
    id: str | int | None = None
    author: str | None = None
    state: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    repository_ref: str | None = None
    category: str | None = None
    accepted_answer_id: str | int | None = None
    labels: list[GitHubLabel] | None = None
    assignees: list[str] | None = None
    entries: list[GitHubEntry] = Field(default_factory=list)
    retrieval_complete: bool | None = None
    missing_collections: list[str] = Field(default_factory=list)


class GitHubExport(BaseModel):
    schema_name: Literal["docling.github"] = "docling.github"
    version: Literal[1] = 1
    conversation: GitHubConversation


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("Expected a GitHub JSON object")
    return value


def _author(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("login") or value.get("name")
        return name if isinstance(name, str) else None
    return None


def _nodes(value: JsonValue) -> list[JsonValue] | None:
    if value is None or isinstance(value, int):
        return None
    if isinstance(value, list):
        return value
    obj = _object(value)
    nodes = obj.get("nodes")
    if isinstance(nodes, list):
        return nodes
    edges = obj.get("edges")
    if isinstance(edges, list):
        return [_object(edge)["node"] for edge in edges]
    raise ValueError("Expected an array or a GraphQL nodes/edges connection")


def _incomplete(value: JsonValue) -> bool:
    if isinstance(value, dict):
        info = value.get("pageInfo")
        if isinstance(info, dict) and info.get("hasNextPage") is True:
            return True
        return any(_incomplete(child) for child in value.values())
    return isinstance(value, list) and any(_incomplete(child) for child in value)


def _entry(value: JsonValue, kind: GitHubEntryKind) -> GitHubEntry:
    obj = _object(value)
    reply_to = obj.get("replyTo")
    fields: dict[str, JsonValue] = {
        "id": obj.get("id"),
        "kind": kind,
        "body": obj.get("body") or "",
        "url": obj.get("html_url") or obj.get("url"),
        "author": _author(obj.get("author", obj.get("user"))),
        "created_at": obj.get(
            "createdAt",
            obj.get("created_at", obj.get("submittedAt", obj.get("submitted_at"))),
        ),
        "updated_at": obj.get("updatedAt", obj.get("updated_at")),
        "parent_id": reply_to.get("id")
        if isinstance(reply_to, dict)
        else obj.get("in_reply_to_id"),
        "state": obj.get("state"),
        "path": obj.get("path"),
        "line": obj.get("line"),
        "start_line": obj.get("startLine", obj.get("start_line")),
        "original_line": obj.get("originalLine", obj.get("original_line")),
        "diff_hunk": obj.get("diffHunk", obj.get("diff_hunk")),
        "is_resolved": obj.get("isResolved"),
        "is_outdated": obj.get("isOutdated"),
        "is_answer": obj.get("isAnswer", False),
    }
    result = GitHubEntry.model_validate(fields)
    children = obj.get("comments") if kind == "review_thread" else obj.get("replies")
    result.replies = _entries(_nodes(children) or [], "reply")
    return result


def _entries(
    values: list[JsonValue],
    kind: GitHubEntryKind,
) -> list[GitHubEntry]:
    entries: list[GitHubEntry] = []
    by_id: dict[str, GitHubEntry] = {}
    for value in values:
        entry = _entry(value, kind)
        if entry.id is not None:
            if str(entry.id) in by_id:
                continue
            by_id[str(entry.id)] = entry
        entries.append(entry)
    roots: list[GitHubEntry] = []
    for entry in entries:
        ancestors: set[str] = set()
        parent_id = entry.parent_id
        while parent_id is not None and str(parent_id) in by_id:
            if str(parent_id) in ancestors or str(parent_id) == str(entry.id):
                raise ValueError("Cyclic GitHub reply relationship")
            ancestors.add(str(parent_id))
            parent_id = by_id[str(parent_id)].parent_id
        if entry.parent_id is not None and str(entry.parent_id) in by_id:
            by_id[str(entry.parent_id)].replies.append(entry)
        else:
            roots.append(entry)
    return roots


def _unwrap(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if payload.get("errors"):
        raise ValueError(
            "GitHub GraphQL response contains errors; export is incomplete"
        )
    if "title" in payload:
        return payload
    for key in (
        "data",
        "repository",
        "node",
        "issue",
        "pullRequest",
        "discussion",
        "pull_request",
    ):
        child = payload.get(key)
        if isinstance(child, dict):
            try:
                obj = _unwrap(child)
            except ValueError:
                continue
            return {**payload, **obj}
    raise ValueError("No GitHub issue, pull request, or discussion was found")


def parse_github_json(payload: JsonValue) -> GitHubConversation:
    """Normalize native exports without requiring callers to reshape their JSON."""
    data = _object(payload)
    if data.get("schema_name") == "docling.github":
        conversation = GitHubExport.model_validate(data).conversation
        if conversation.source != parse_github_url(conversation.source.url):
            raise ValueError(
                "GitHub export source fields do not match its canonical URL"
            )
        return conversation
    obj = _unwrap(data)
    url = obj.get("html_url") or obj.get("url")
    if not isinstance(url, str):
        raise ValueError("GitHub JSON must include the object's web url or html_url")
    source = parse_github_url(url)
    if obj.get("number") != source.number:
        raise ValueError("GitHub object number does not match its URL")
    title = obj.get("title")
    if not isinstance(title, str):
        raise ValueError("GitHub JSON must include a title")
    repo = obj.get("repository")
    branch: JsonValue = None
    if isinstance(repo, dict):
        branch = repo.get("default_branch")
        ref = repo.get("defaultBranchRef")
        if isinstance(ref, dict):
            branch = ref.get("name")
    category = obj.get("category")
    result = GitHubConversation.model_validate(
        {
            "source": source.model_dump(),
            "title": title,
            "body": obj.get("body") or "",
            "id": obj.get("id"),
            "author": _author(obj.get("author", obj.get("user"))),
            "state": "MERGED" if obj.get("merged") is True else obj.get("state"),
            "created_at": obj.get("createdAt", obj.get("created_at")),
            "updated_at": obj.get("updatedAt", obj.get("updated_at")),
            "repository_ref": branch,
            "category": category.get("name")
            if isinstance(category, dict)
            else category,
        }
    )
    labels = _nodes(obj.get("labels"))
    if labels is not None:
        result.labels = []
        identities: set[str] = set()
        for label in labels:
            parsed = (
                GitHubLabel(name=label)
                if isinstance(label, str)
                else GitHubLabel.model_validate(label)
            )
            identity = (
                f"id:{parsed.id}" if parsed.id is not None else f"name:{parsed.name}"
            )
            if identity not in identities:
                result.labels.append(parsed)
                identities.add(identity)
    assignees = _nodes(obj.get("assignees"))
    if assignees is not None:
        result.assignees = [
            name for actor in assignees if (name := _author(actor)) is not None
        ]
    collections: list[tuple[str, GitHubEntryKind]] = [("comments", "comment")]
    if source.kind == "pull_request":
        collections += [("reviews", "review"), ("reviewThreads", "review_thread")]
    for name, kind in collections:
        values = _nodes(obj.get(name))
        if values is None:
            result.missing_collections.append(name)
        else:
            result.entries.extend(_entries(values, kind))
    answer = obj.get("answer")
    if isinstance(answer, dict):
        answer_id = answer.get("id")
        if isinstance(answer_id, str | int) and not isinstance(answer_id, bool):
            result.accepted_answer_id = answer_id
            pending = list(result.entries)
            while pending:
                entry = pending.pop()
                if entry.id == answer_id:
                    entry.is_answer = True
                pending.extend(entry.replies)
    if _incomplete(data):
        result.retrieval_complete = False
    return result
