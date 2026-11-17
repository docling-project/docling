# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Shared, allowlisted classification for issues and pull requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pr_topic_labels import FORMAT_LABELS

SEVERITIES = ("critical", "high", "medium", "low")
APPLICATION_LABELS = frozenset({"bug", "enhancement", "chore"})
PRIORITY_LABELS = frozenset(f"priority:{severity}" for severity in SEVERITIES)


@dataclass(slots=True)
class Classification:
    severity: str | None
    application: str | None
    formats: list[str]


def parse_classification(value: Any) -> Classification:
    """Reject unknown values before the model output reaches a write token."""
    if (
        not isinstance(value, dict)
        or not {"severity", "application", "formats"} <= value.keys()
    ):
        raise ValueError(
            "classification must contain severity, application, and formats"
        )
    severity = value.get("severity")
    application = value.get("application")
    formats = value.get("formats")
    if severity is not None and (
        not isinstance(severity, str) or severity not in SEVERITIES
    ):
        raise ValueError("invalid classification severity")
    if application is not None and (
        not isinstance(application, str) or application not in APPLICATION_LABELS
    ):
        raise ValueError("invalid classification application")
    if not isinstance(formats, list):
        raise ValueError("invalid classification formats")
    validated_formats: list[str] = []
    for item in formats:
        if not isinstance(item, str) or item not in FORMAT_LABELS:
            raise ValueError("invalid classification formats")
        if item not in validated_formats:
            validated_formats.append(item)
    return Classification(severity, application, validated_formats)


def classification_labels(
    classification: Classification | None,
    topics: list[str],
    current: set[str],
    available: set[str],
) -> list[str]:
    """Add labels once. Existing application and priority labels take precedence."""
    candidates = list(topics)
    if classification is not None:
        if classification.application is not None:
            candidates.append(classification.application)
        if classification.severity is not None:
            candidates.append(f"priority:{classification.severity}")
        candidates.extend(classification.formats)
    selected: list[str] = []
    occupied = set(current)
    for label in candidates:
        if label not in available or label in occupied:
            continue
        if label in APPLICATION_LABELS and occupied & APPLICATION_LABELS:
            continue
        if label in PRIORITY_LABELS and occupied & PRIORITY_LABELS:
            continue
        selected.append(label)
        occupied.add(label)
    return selected
