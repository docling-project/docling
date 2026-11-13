# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read complete GitHub conversations through the user's authenticated gh CLI."""

import subprocess
import time
from io import BytesIO

from pydantic import JsonValue, TypeAdapter

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.github import GitHubExport, parse_github_json, parse_github_url

_COMMENT_FIELDS = "id url body author { login } createdAt updatedAt"
_PAGE_INFO = "pageInfo { hasNextPage endCursor }"


class _GhReader:
    def __init__(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("GitHub retrieval timeout must be positive")
        self._deadline = time.monotonic() + timeout

    def query(self, query: str, **variables: str) -> dict[str, JsonValue]:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "GitHub retrieval timed out before the conversation was complete"
            )
        command = ["gh", "api", "graphql", "-f", f"query={query}"]
        for name, value in variables.items():
            command.extend(["-f", f"{name}={value}"])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=remaining,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "GitHub retrieval requires gh. Install it and run `gh auth login`."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "GitHub retrieval timed out; no complete export was produced"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"gh api failed; check authentication, permissions, and rate limits: {exc.stderr.strip()}"
            ) from exc
        response = TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout)
        if response.get("errors"):
            raise RuntimeError(f"GitHub returned GraphQL errors: {response['errors']}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(
                "GitHub response has no data; no complete export was produced"
            )
        return data

    def collection(
        self, node_id: str, node_type: str, field: str, selection: str
    ) -> list[JsonValue]:
        query = (
            "query($id: ID!, $cursor: String) { node(id: $id) { ... on "
            + node_type
            + " { "
            + field
            + "(first: 100, after: $cursor) { nodes { "
            + selection
            + " } "
            + _PAGE_INFO
            + " } } } }"
        )
        nodes: list[JsonValue] = []
        variables = {"id": node_id}
        seen_cursors: set[str] = set()
        while True:
            data = self.query(query, **variables)
            node = data.get("node")
            connection = node.get(field) if isinstance(node, dict) else None
            if not isinstance(connection, dict):
                raise RuntimeError(
                    f"GitHub omitted {node_type}.{field}; export is incomplete"
                )
            page = connection.get("nodes")
            info = connection.get("pageInfo")
            if not isinstance(page, list) or not isinstance(info, dict):
                raise RuntimeError(f"Invalid GitHub connection for {field}")
            nodes.extend(page)
            if info.get("hasNextPage") is False:
                return nodes
            cursor = info.get("endCursor")
            if (
                info.get("hasNextPage") is not True
                or not isinstance(cursor, str)
                or cursor in seen_cursors
            ):
                raise RuntimeError(
                    f"GitHub pagination did not advance for {field}; export is incomplete"
                )
            seen_cursors.add(cursor)
            variables["cursor"] = cursor


def fetch_github_conversation(url: str, *, timeout: float = 60) -> GitHubExport:
    """Retrieve one conversation, following every top-level and nested cursor.

    Authentication stays in gh. An error raises instead of returning a partial
    conversation as complete. The timeout covers the entire retrieval.
    """
    source = parse_github_url(url)
    reader = _GhReader(timeout)
    common = "id title number url body author { login } createdAt updatedAt repository { defaultBranchRef { name } }"
    object_fields = {
        "issue": "issue",
        "pull_request": "pullRequest",
        "discussion": "discussion",
    }
    extra = (
        "category { name } answer { id }" if source.kind == "discussion" else "state"
    )
    query = (
        "query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { resource: "
        + object_fields[source.kind]
        + f"(number: {source.number}) {{ "
        + common
        + " "
        + extra
        + " } } }"
    )
    repository = reader.query(query, owner=source.owner, name=source.repository).get(
        "repository"
    )
    resource = repository.get("resource") if isinstance(repository, dict) else None
    if not isinstance(resource, dict) or not isinstance(resource.get("id"), str):
        raise RuntimeError("GitHub conversation was not found or is not accessible")
    node_id = resource["id"]
    node_types = {
        "issue": "Issue",
        "pull_request": "PullRequest",
        "discussion": "Discussion",
    }
    node_type = node_types[source.kind]
    comment_fields = _COMMENT_FIELDS
    if source.kind == "discussion":
        comment_fields += " isAnswer replyTo { id }"
    comments = reader.collection(node_id, node_type, "comments", comment_fields)
    if source.kind == "discussion":
        for comment in comments:
            if not isinstance(comment, dict) or not isinstance(comment.get("id"), str):
                raise RuntimeError("Invalid discussion comment in GitHub response")
            comment["replies"] = reader.collection(
                comment["id"], "DiscussionComment", "replies", comment_fields
            )
    resource["comments"] = comments
    if source.kind != "discussion":
        resource["labels"] = reader.collection(
            node_id, node_type, "labels", "id name description color url"
        )
        resource["assignees"] = reader.collection(
            node_id, node_type, "assignees", "login"
        )
    if source.kind == "pull_request":
        resource["reviews"] = reader.collection(
            node_id,
            node_type,
            "reviews",
            "id url body state author { login } createdAt: submittedAt",
        )
        threads = reader.collection(
            node_id,
            node_type,
            "reviewThreads",
            "id path line startLine originalLine isResolved isOutdated",
        )
        for thread in threads:
            if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
                raise RuntimeError("Invalid review thread in GitHub response")
            thread["comments"] = reader.collection(
                thread["id"],
                "PullRequestReviewThread",
                "comments",
                _COMMENT_FIELDS + " diffHunk replyTo { id }",
            )
        resource["reviewThreads"] = threads
    conversation = parse_github_json(resource)
    conversation.retrieval_complete = True
    return GitHubExport(conversation=conversation)


def github_document_stream(export: GitHubExport) -> DocumentStream:
    """Pass a fetched export through the same backend as an on-disk JSON file."""
    source = export.conversation.source
    return DocumentStream(
        name=f"{source.owner}-{source.repository}-{source.kind}-{source.number}.github.json",
        stream=BytesIO(export.model_dump_json().encode("utf-8")),
    )
