# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pr_topic_labels",
    Path(__file__).resolve().parents[1] / ".github/scripts/pr_topic_labels.py",
)
assert SPEC is not None and SPEC.loader is not None
topics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = topics
SPEC.loader.exec_module(topics)


def test_title_type_scope_and_paths_give_labels() -> None:
    labels = topics.deterministic_topics(
        "fix(md, tables): keep pipe rows",
        [
            "docling/backend/md_backend.py",
            "docling/backend/xml/jats_backend.py",
            "tests/test_backend_markdown.py",
        ],
    )
    assert labels == ["bug", "markdown", "table structure", "xml"]


def test_documentation_label_only_when_all_source_changes_are_docs() -> None:
    assert topics.deterministic_topics(
        "chore: update guide", ["docs/usage.md", "README.md"]
    ) == ["documentation"]
    assert "documentation" not in topics.deterministic_topics(
        "chore: update", ["docs/usage.md", "docling/cli/main.py"]
    )


def test_unknown_title_types_and_scopes_add_nothing() -> None:
    assert topics.deterministic_topics("chore(release): 2.0", []) == []
    assert topics.deterministic_topics("Not a conventional title", []) == []


def test_model_topics_are_validated_and_limited() -> None:
    with pytest.raises(ValueError, match="unknown topic"):
        topics.parse_model_topics(["wontfix"])
    assert topics.parse_model_topics(None) == []
    assert topics.parse_model_topics(["ocr", "ocr", "layout", "pdf", "csv"]) == [
        "ocr",
        "layout",
        "pdf",
    ]


def test_only_existing_repository_labels_are_selected() -> None:
    selected = topics.select_topics(["bug", "docx"], ["docx", "ocr"], {"bug", "ocr"})
    assert selected == ["bug", "ocr"]


def test_every_title_and_path_label_is_in_the_allowlist() -> None:
    mapped = {
        *topics.TYPE_LABELS.values(),
        *topics.SCOPE_LABELS.values(),
        *(label for _, label in topics.PATH_LABELS),
    }
    assert mapped <= set(topics.TOPIC_LABELS)
