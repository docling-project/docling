# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import base64
import json
import re
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import CodeItem, DoclingDocument, TextItem
from PIL import Image
from pydantic import JsonValue
from typer.testing import CliRunner

from docling.cli.main import app
from docling.datamodel.backend_options import GitHubJSONBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.datamodel.github import GitHubExport, parse_github_json, parse_github_url
from docling.document_converter import DocumentConverter, GitHubJSONFormatOption
from docling.utils.github import fetch_github_conversation, github_document_stream

pytestmark = pytest.mark.cross_platform


@pytest.fixture
def pull_request() -> dict[str, JsonValue]:
    return {
        "id": "pr1",
        "url": "https://github.com/owner/repo/pull/1",
        "number": 1,
        "title": "Keep code and links",
        "state": "OPEN",
        "author": {"login": "maintainer"},
        "repository": {"defaultBranchRef": {"name": "main"}},
        "body": "```python\n\n    x = '<a>&amp;</a>'  \n\n```\n\n"
        "[docs](docs/guide.md?q=1#part) and `literal <a>`\n\n"
        "<details><summary>Logs</summary><pre><code>\n  y &lt; z  \n\n</code></pre>"
        '<p><a href="https://example.com/?a=1&amp;b=2#part">HTML link</a></p></details>\n\n'
        "| API | Example |\n| --- | --- |\n| [docs](https://example.com/api) | `f(x)` |\n",
        "labels": [
            {
                "id": "l1",
                "name": "bug",
                "description": "Unexpected behavior",
                "color": "ff0000",
                "url": "https://github.com/owner/repo/labels/bug",
            }
        ],
        "comments": [
            {
                "id": "c1",
                "body": "Comment body",
                "author": None,
                "url": "https://github.com/owner/repo/pull/1#issuecomment-1",
                "createdAt": "2026-01-01T12:00:00Z",
            }
        ],
        "reviews": [
            {
                "id": "r1",
                "body": "Review body",
                "author": {"login": "reviewer"},
                "state": "CHANGES_REQUESTED",
            }
        ],
        "reviewThreads": [
            {
                "id": "t1",
                "path": "app.py",
                "line": 5,
                "isResolved": False,
                "isOutdated": True,
                "comments": [
                    {
                        "id": "tc1",
                        "body": "Thread root",
                        "diffHunk": "@@ -1 +1 @@\n-old\n+new\n",
                    },
                    {"id": "tc2", "body": "Thread reply", "replyTo": {"id": "tc1"}},
                ],
            }
        ],
    }


def _convert(payload: dict[str, JsonValue], **options: object) -> DoclingDocument:
    converter = DocumentConverter(
        allowed_formats=[InputFormat.JSON_GITHUB],
        format_options={
            InputFormat.JSON_GITHUB: GitHubJSONFormatOption(
                backend_options=GitHubJSONBackendOptions.model_validate(options),
            )
        },
    )
    return converter.convert_string(
        json.dumps(payload), format=InputFormat.JSON_GITHUB
    ).document


def _links(doc: DoclingDocument) -> list[tuple[str, str]]:
    return [
        (item.text, str(item.hyperlink))
        for item in doc.texts
        if item.hyperlink is not None
    ]


def test_native_pull_request_preserves_content_and_ownership(
    pull_request: dict[str, JsonValue], tmp_path: Path
) -> None:
    doc = _convert(pull_request)
    archive = tmp_path / "conversation.dclx"
    doc.save_as_doclang_archive(archive, add_named_groups=True)
    loaded = DoclingDocument.load_from_doclang_archive(archive)
    for candidate in (doc, loaded):
        code = [item.text for item in candidate.texts if isinstance(item, CodeItem)]
        assert "\n    x = '<a>&amp;</a>'  \n\n" in code
        assert "\n  y < z  \n\n" in code
        assert "@@ -1 +1 @@\n-old\n+new\n" in code
        assert (
            "docs",
            "https://github.com/owner/repo/blob/main/docs/guide.md?q=1#part",
        ) in _links(candidate)
        assert ("HTML link", "https://example.com/?a=1&b=2#part") in _links(candidate)
        assert ("bug", "https://github.com/owner/repo/labels/bug") in _links(candidate)
        assert candidate.tables[0].data.grid[1][1].text == "Unexpected behavior"
        assert candidate.tables[1].data.grid[1][0].text == "docs"
        assert ("docs", "https://example.com/api") in _links(candidate)
        groups = {group.name: group for group in candidate.groups}
        assert groups["github:reply:tc2"].parent == groups["github:reply:tc1"].get_ref()
        headings = [item.text for item in candidate.texts]
        assert (
            headings.index("Source information")
            < headings.index("Labels")
            < headings.index("Body")
        )
    source_meta = doc.texts[0].meta
    assert source_meta is not None
    labels = source_meta.get_custom_part()["github__source"]["labels"]
    assert labels[0]["color"] == "ff0000"
    assert labels[0]["id"] == "l1"


@pytest.mark.parametrize("shape", ["gh", "rest", "graphql", "envelope"])
def test_native_shapes_route_to_github(shape: str, tmp_path: Path) -> None:
    raw: dict[str, JsonValue] = {
        "title": "Issue",
        "number": 2,
        "body": "Issue body",
        "url": "https://github.com/o/r/issues/2",
    }
    if shape == "rest":
        raw["html_url"] = raw["url"]
        raw["url"] = "https://api.github.com/repos/o/r/issues/2"
        raw["user"] = {"login": "author"}
        raw["comments"] = 3
    elif shape == "graphql":
        raw = {"data": {"repository": {"issue": raw}}}
    elif shape == "envelope":
        raw = json.loads(
            GitHubExport(conversation=parse_github_json(raw)).model_dump_json()
        )
    path = tmp_path / "issue.json"
    path.write_text("\ufeff" + json.dumps(raw), encoding="utf-8")
    converter = DocumentConverter(allowed_formats=[InputFormat.JSON_GITHUB])
    from_path = converter.convert(path)
    from_stream = converter.convert(
        DocumentStream(name="issue.json", stream=BytesIO(path.read_bytes()))
    )
    assert from_path.input.format == InputFormat.JSON_GITHUB
    assert (
        from_path.document.export_to_markdown()
        == from_stream.document.export_to_markdown()
    )
    assert "Issue body" in from_path.document.export_to_text()


def test_detection_reads_past_large_body_without_stealing_docling_json(
    tmp_path: Path,
) -> None:
    raw = {
        "body": "x" * 100_000,
        "title": "Large body",
        "url": "https://github.com/o/r/issues/2",
        "number": 2,
    }
    guesser = _DocumentConversionInput(path_or_stream_iterator=iter([]))
    assert (
        guesser._guess_format(
            DocumentStream(name="large.json", stream=BytesIO(json.dumps(raw).encode()))
        )
        == InputFormat.JSON_GITHUB
    )
    original = DoclingDocument(name="Existing saved JSON")
    original.add_text(label="text", text="Existing content")
    path = tmp_path / "saved.json"
    original.save_as_json(path)
    result = DocumentConverter(allowed_formats=[InputFormat.JSON_DOCLING]).convert(path)
    assert result.input.format == InputFormat.JSON_DOCLING
    assert result.document.texts[0].text == "Existing content"


def test_discussion_reply_and_accepted_answer() -> None:
    raw: dict[str, JsonValue] = {
        "url": "https://github.com/o/r/discussions/3",
        "number": 3,
        "title": "Question",
        "body": "Question body",
        "category": {"name": "Q&A"},
        "answer": {"id": "c"},
        "comments": {
            "nodes": [
                {
                    "id": "c",
                    "body": "Answer",
                    "replies": {"edges": [{"node": {"id": "r", "body": "Thanks"}}]},
                }
            ],
            "pageInfo": {"hasNextPage": False},
        },
    }
    doc = _convert(raw)
    assert "Accepted answer" in doc.export_to_text()
    assert "Category: Q&A" in doc.export_to_text()
    groups = {group.name: group for group in doc.groups}
    assert groups["github:reply:r"].parent == groups["github:comment:c"].get_ref()


@pytest.mark.parametrize(
    ("labels", "expected"), [(None, "Not supplied"), ([], "None"), (["bug"], "bug")]
)
def test_label_presence(labels: JsonValue, expected: str) -> None:
    raw: dict[str, JsonValue] = {
        "title": "Issue",
        "number": 1,
        "url": "https://github.com/o/r/issues/1",
    }
    if labels is not None:
        raw["labels"] = labels
    doc = _convert(raw)
    labels_group = next(group for group in doc.groups if group.name == "github:labels")
    texts = [
        item.text
        for item, _ in doc.iterate_items(root=labels_group)
        if isinstance(item, TextItem)
    ]
    assert expected in texts


def test_incomplete_and_malformed_inputs() -> None:
    raw: dict[str, JsonValue] = {
        "title": "Issue",
        "number": 1,
        "url": "https://github.com/o/r/issues/1",
        "comments": {"nodes": [], "pageInfo": {"hasNextPage": True}},
    }
    assert parse_github_json(raw).retrieval_complete is False
    for invalid in (
        {"title": "Not GitHub"},
        {**raw, "number": 9},
        {"errors": [{"message": "Denied"}], "data": {"issue": raw}},
    ):
        with pytest.raises(ValueError):
            parse_github_json(invalid)
    with pytest.raises(ValueError, match="Cyclic"):
        parse_github_json(
            {
                **raw,
                "comments": [
                    {"id": "a", "in_reply_to_id": "b"},
                    {"id": "b", "in_reply_to_id": "a"},
                ],
            }
        )


def test_local_image_assets_and_strict_missing_policy(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), "red").save(tmp_path / "pixel.png")
    raw = {
        "title": "Images",
        "url": "https://github.com/o/r/issues/1",
        "number": 1,
        "body": "![pixel](pixel.png)",
    }
    path = tmp_path / "issue.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    converter = DocumentConverter(
        allowed_formats=[InputFormat.JSON_GITHUB],
        format_options={
            InputFormat.JSON_GITHUB: GitHubJSONFormatOption(
                backend_options=GitHubJSONBackendOptions(
                    enable_local_fetch=True, strict_assets=True
                )
            )
        },
    )
    doc = converter.convert(path).document
    archive = tmp_path / "images.dclx"
    doc.save_as_doclang_archive(archive, add_named_groups=True)
    loaded = DoclingDocument.load_from_doclang_archive(archive)
    assert loaded.pictures[0].get_image(loaded).getpixel((0, 0)) == (255, 0, 0)
    with zipfile.ZipFile(archive) as zipped:
        assert any(name.startswith("assets/") for name in zipped.namelist())
    with pytest.raises(Exception, match="Pipeline SimplePipeline failed") as error:
        _convert(raw, strict_assets=True)
    assert error.value.__cause__ is not None
    assert "picture asset" in str(error.value.__cause__)


def test_cli_converts_url_and_saved_json(
    pull_request: dict[str, JsonValue], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = GitHubExport(conversation=parse_github_json(pull_request))
    monkeypatch.setattr(
        "docling.cli.main.fetch_github_conversation", lambda url, timeout: export
    )
    path = tmp_path / "saved.json"
    path.write_text(export.model_dump_json(), encoding="utf-8")
    for source, directory in (
        (str(path), tmp_path / "local"),
        (export.conversation.source.url, tmp_path / "remote"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "convert",
                source,
                "--from",
                "json_github",
                "--to",
                "dclx",
                "--output",
                str(directory),
            ],
        )
        assert result.exit_code == 0, result.output
        archive = next(directory.glob("*.dclx"))
        loaded = DoclingDocument.load_from_doclang_archive(archive)
        assert ("bug", "https://github.com/owner/repo/labels/bug") in _links(loaded)
        assert any(group.name == "github:reply:tc2" for group in loaded.groups)


def test_gh_paginates_nested_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        variables = dict(
            command[index + 1].split("=", 1)
            for index, value in enumerate(command)
            if value == "-f"
        )
        query = variables["query"]
        assert kwargs["check"] is True
        if "resource:" in query:
            data: JsonValue = {
                "repository": {
                    "resource": {
                        "id": "p",
                        "url": "https://github.com/o/r/pull/1",
                        "number": 1,
                        "title": "PR",
                    }
                }
            }
        else:
            field = re.search(r"\{ (\w+)\(first:", query).group(1)
            cursor = variables.get("cursor")
            calls.append((field, cursor))
            if field == "reviewThreads":
                nodes: list[JsonValue] = [{"id": "t", "path": "a.py", "line": 1}]
            elif field == "comments" and variables["id"] == "t":
                nodes = [
                    {
                        "id": "reply" if cursor else "root",
                        "body": "Second" if cursor else "First",
                    }
                ]
            else:
                nodes = []
            more = field == "comments" and variables["id"] == "t" and cursor is None
            data = {
                "node": {
                    field: {
                        "nodes": nodes,
                        "pageInfo": {
                            "hasNextPage": more,
                            "endCursor": "next" if more else None,
                        },
                    }
                }
            }
        return subprocess.CompletedProcess(command, 0, json.dumps({"data": data}), "")

    monkeypatch.setattr("docling.utils.github.subprocess.run", run)
    export = fetch_github_conversation("https://github.com/o/r/pull/1")
    assert export.conversation.retrieval_complete is True
    thread = next(
        entry for entry in export.conversation.entries if entry.kind == "review_thread"
    )
    assert [entry.body for entry in thread.replies] == ["First", "Second"]
    assert ("comments", "next") in calls
    result = DocumentConverter(allowed_formats=[InputFormat.JSON_GITHUB]).convert(
        github_document_stream(export)
    )
    assert "Second" in result.document.export_to_text()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired("gh", 1),
        subprocess.CalledProcessError(1, "gh", stderr="HTTP 403: rate limited"),
    ],
)
def test_gh_errors_do_not_return_complete_export(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("docling.utils.github.subprocess.run", run)
    with pytest.raises(RuntimeError):
        fetch_github_conversation("https://github.com/o/r/issues/1")


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/o/r/issues/1",
        "https://github.com.evil/o/r/issues/1",
        "https://github.com/o/r/blob/main/README.md",
    ],
)
def test_only_conversation_urls_are_fetched(url: str) -> None:
    with pytest.raises(ValueError):
        parse_github_url(url)


def test_fenced_html_is_literal_and_relative_links_need_context() -> None:
    snippet = '<details><a href="docs/page.md">&amp;</a></details>\n'
    doc = _convert(
        {
            "url": "https://github.com/o/r/issues/1",
            "number": 1,
            "title": "HTML example",
            "body": f"```html\n{snippet}```\n\n[relative](docs/page.md) and [anchor](#part)\n",
        }
    )
    assert any(
        isinstance(item, CodeItem) and item.text == snippet and item.hyperlink is None
        for item in doc.texts
    )
    assert ("relative", "docs/page.md") in _links(doc)
    assert ("anchor", "https://github.com/o/r/issues/1#part") in _links(doc)


def test_gh_discussion_paginates_comments_and_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        variables = dict(
            command[index + 1].split("=", 1)
            for index, value in enumerate(command)
            if value == "-f"
        )
        if "resource:" in variables["query"]:
            data: JsonValue = {
                "repository": {
                    "resource": {
                        "id": "d",
                        "number": 3,
                        "title": "Discussion",
                        "url": "https://github.com/o/r/discussions/3",
                    }
                }
            }
        else:
            field = re.search(r"\{ (\w+)\(first:", variables["query"]).group(1)
            node_id = variables["id"]
            cursor = variables.get("cursor")
            calls.append((node_id, field, cursor))
            if field == "comments":
                nodes: list[JsonValue] = [
                    {
                        "id": "c2" if cursor else "c1",
                        "body": "second" if cursor else "first",
                        "isAnswer": cursor is None,
                    }
                ]
                more = cursor is None
            elif node_id == "c1":
                nodes = [
                    {
                        "id": "r2" if cursor else "r1",
                        "body": "reply 2" if cursor else "reply 1",
                    }
                ]
                more = cursor is None
            else:
                nodes = []
                more = False
            data = {
                "node": {
                    field: {
                        "nodes": nodes,
                        "pageInfo": {
                            "hasNextPage": more,
                            "endCursor": "next" if more else None,
                        },
                    }
                }
            }
        return subprocess.CompletedProcess(command, 0, json.dumps({"data": data}), "")

    monkeypatch.setattr("docling.utils.github.subprocess.run", run)
    export = fetch_github_conversation("https://github.com/o/r/discussions/3")
    assert export.conversation.retrieval_complete is True
    assert len(export.conversation.entries) == 2
    assert export.conversation.entries[0].is_answer
    assert [reply.body for reply in export.conversation.entries[0].replies] == [
        "reply 1",
        "reply 2",
    ]
    assert ("d", "comments", "next") in calls
    assert ("c1", "replies", "next") in calls


@pytest.mark.parametrize(
    "failure", ["graphql", "repeated_cursor", "missing_connection"]
)
def test_gh_rejects_partial_exports(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        variables = dict(
            command[index + 1].split("=", 1)
            for index, value in enumerate(command)
            if value == "-f"
        )
        if failure == "graphql":
            payload: JsonValue = {"data": {}, "errors": [{"message": "rate limit"}]}
        elif "resource:" in variables["query"]:
            payload = {
                "data": {
                    "repository": {
                        "resource": {
                            "id": "i",
                            "number": 1,
                            "title": "Issue",
                            "url": "https://github.com/o/r/issues/1",
                        }
                    }
                }
            }
        elif failure == "missing_connection":
            payload = {"data": {"node": {}}}
        else:
            payload = {
                "data": {
                    "node": {
                        "comments": {
                            "nodes": [],
                            "pageInfo": {"hasNextPage": True, "endCursor": "same"},
                        }
                    }
                }
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr("docling.utils.github.subprocess.run", run)
    with pytest.raises(RuntimeError):
        fetch_github_conversation("https://github.com/o/r/issues/1")


def test_embedded_image_is_packaged_without_remote_access(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "blue").save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    doc = _convert(
        {
            "title": "Embedded image",
            "number": 1,
            "url": "https://github.com/o/r/issues/1",
            "body": f"![blue]({uri})",
        }
    )
    assert doc.pictures[0].caption_text(doc) == "blue"
    path = tmp_path / "embedded.dclx"
    doc.save_as_doclang_archive(path, add_named_groups=True)
    loaded = DoclingDocument.load_from_doclang_archive(path)
    assert loaded.pictures[0].caption_text(loaded) == "blue"
    assert loaded.pictures[0].get_image(loaded).getpixel((0, 0)) == (0, 0, 255)
