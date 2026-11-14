# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from triage_classification import (  # noqa: E402
    Classification,
    classification_labels,
    parse_classification,
)


@pytest.mark.parametrize(
    "change",
    [
        {"severity": "urgent"},
        {"severity": ["critical"]},
        {"application": "fix"},
        {"application": {"name": "bug"}},
        {"formats": ["pdf", "tests:full"]},
        {"formats": "pdf"},
        {"formats": ["pdf", ["docx"]]},
    ],
)
def test_untrusted_classification_cannot_select_arbitrary_labels(
    change: dict[str, object],
) -> None:
    answer = {"severity": "high", "application": "bug", "formats": ["pdf"]}
    answer.update(change)
    with pytest.raises(ValueError, match="invalid classification"):
        parse_classification(answer)


def test_current_groups_override_title_and_model_but_allow_format_labels() -> None:
    assert classification_labels(
        Classification("critical", "bug", ["pdf", "docx"]),
        ["bug", "documentation", "pdf"],
        {"priority:low", "enhancement"},
        {"priority:critical", "bug", "documentation", "pdf", "docx"},
    ) == ["documentation", "pdf", "docx"]


def test_title_application_overrides_model_and_missing_labels_are_skipped() -> None:
    assert classification_labels(
        Classification("medium", "enhancement", ["pdf", "docx"]),
        ["bug", "pdf"],
        set(),
        {"bug", "enhancement", "priority:medium", "pdf"},
    ) == ["bug", "pdf", "priority:medium"]


def test_insufficient_evidence_does_not_assign_severity_or_application() -> None:
    result = parse_classification(
        {"severity": None, "application": None, "formats": []}
    )
    assert classification_labels(result, [], set(), {"bug", "priority:low"}) == []
