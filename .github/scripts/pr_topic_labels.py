# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Select topic labels for a pull request from the repository label set.

Deterministic rules map the conventional-commit title and the changed source
paths to labels. The AI triage can add a few more labels, but only names from
``TOPIC_LABELS``. The workflow only adds topic labels and never removes them,
because maintainers own these labels.
"""

from __future__ import annotations

import fnmatch
import re

# Existing repository labels that describe what a PR changes. Process labels
# (priority, triage, duplicate, wontfix, ...) are intentionally not here.
TOPIC_LABELS: tuple[str, ...] = (
    "bug",
    "enhancement",
    "documentation",
    "performance",
    "tests",
    "dependency mgmt",
    "error-handling",
    "asciidoc",
    "csv",
    "docx",
    "html",
    "iwork",
    "markdown",
    "odf",
    "pdf",
    "pdf parsing",
    "pptx",
    "vtt",
    "xlsx",
    "xml",
    "asr",
    "ocr",
    "vlm-pipeline",
    "layout",
    "table structure",
    "reading_order",
    "chunker",
    "CLI",
    "accelerators",
    "docling-document",
    "language support",
    "rtl-language",
    "mimetype",
)
MAX_MODEL_TOPICS = 3
MAX_TOPICS = 6

TITLE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?!?:")
TYPE_LABELS = {
    "fix": "bug",
    "feat": "enhancement",
    "docs": "documentation",
    "perf": "performance",
}
SCOPE_LABELS = {
    "asciidoc": "asciidoc",
    "csv": "csv",
    "docx": "docx",
    "html": "html",
    "iwork": "iwork",
    "md": "markdown",
    "markdown": "markdown",
    "odf": "odf",
    "opendocument": "odf",
    "pdf": "pdf",
    "pptx": "pptx",
    "vtt": "vtt",
    "webvtt": "vtt",
    "xlsx": "xlsx",
    "xml": "xml",
    "jats": "xml",
    "uspto": "xml",
    "xbrl": "xml",
    "asr": "asr",
    "ocr": "ocr",
    "tesseract": "ocr",
    "vlm": "vlm-pipeline",
    "layout": "layout",
    "table": "table structure",
    "tables": "table structure",
    "reading-order": "reading_order",
    "reading_order": "reading_order",
    "chunking": "chunker",
    "chunker": "chunker",
    "cli": "CLI",
    "deps": "dependency mgmt",
}
PATH_LABELS: tuple[tuple[str, str], ...] = (
    ("docling/backend/msword_backend.py", "docx"),
    ("docling/backend/docx/*", "docx"),
    ("docling/backend/html_backend.py", "html"),
    ("docling/backend/md_backend.py", "markdown"),
    ("docling/backend/csv_backend.py", "csv"),
    ("docling/backend/msexcel_backend.py", "xlsx"),
    ("docling/backend/mspowerpoint_backend.py", "pptx"),
    ("docling/backend/asciidoc_backend.py", "asciidoc"),
    ("docling/backend/opendocument_backend.py", "odf"),
    ("docling/backend/iwork_backend.py", "iwork"),
    ("docling/backend/iwork/*", "iwork"),
    ("docling/backend/xml/*", "xml"),
    ("docling/backend/webvtt_backend.py", "vtt"),
    ("docling/backend/docling_parse*", "pdf parsing"),
    ("docling/backend/pypdfium2_backend.py", "pdf"),
    ("docling/backend/pdf_backend.py", "pdf"),
    ("docling/models/base_ocr_model.py", "ocr"),
    ("docling/models/factories/ocr_factory.py", "ocr"),
    ("docling/models/stages/ocr/*", "ocr"),
    ("docling/pipeline/asr_*", "asr"),
    ("docling/pipeline/vlm_pipeline.py", "vlm-pipeline"),
    ("docling/models/vlm_pipeline_models/*", "vlm-pipeline"),
    ("docling/models/stages/layout/*", "layout"),
    ("docling/models/stages/table_structure/*", "table structure"),
    ("docling/models/stages/reading_order/*", "reading_order"),
    ("docling/models/postprocessing/reading_order_rb.py", "reading_order"),
    ("docling/chunking/*", "chunker"),
    ("docling/cli/*", "CLI"),
)
DOC_PATHS = ("docs/*", "*.md")


def deterministic_topics(title: str, paths: list[str]) -> list[str]:
    """Return topic labels from the PR title and the changed paths."""
    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    match = TITLE.match(title.strip())
    if match:
        if match.group("type") in TYPE_LABELS:
            add(TYPE_LABELS[match.group("type")])
        for scope in re.split(r"[,/ ]+", match.group("scope") or ""):
            if scope.lower() in SCOPE_LABELS:
                add(SCOPE_LABELS[scope.lower()])
    for path in paths:
        for pattern, label in PATH_LABELS:
            if fnmatch.fnmatchcase(path, pattern):
                add(label)
    source = [path for path in paths if not path.startswith("tests/data/")]
    if source and all(
        any(fnmatch.fnmatchcase(path, pattern) for pattern in DOC_PATHS)
        for path in source
    ):
        add("documentation")
    return labels


def parse_model_topics(value: object) -> list[str]:
    """Validate the topics of the model answer against the allowlist."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("topics must be a list")
    topics: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in TOPIC_LABELS:
            raise ValueError(f"unknown topic label: {item!r}")
        if item not in topics:
            topics.append(item)
    return topics[:MAX_MODEL_TOPICS]


def select_topics(
    deterministic: list[str], model: list[str], existing: set[str]
) -> list[str]:
    """Merge the topics. Only labels that exist in the repository are kept."""
    merged: list[str] = []
    for label in [*deterministic, *model]:
        if label in existing and label not in merged:
            merged.append(label)
    return merged[:MAX_TOPICS]
