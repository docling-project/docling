# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Summarize changes to regression reference data without running PR code.

The summary classifies each changed groundtruth file so that reviewers can see
at a glance which changes are formatting noise and which change the document
content, structure, or tables.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

GROUNDTRUTH_SEGMENT = "groundtruth"
MAX_LISTED_FILES = 60
# Keys that only carry layout coordinates or page geometry.
GEOMETRY_KEYS = frozenset({"prov", "bbox", "size", "pages", "charspan"})


class ChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    FORMATTING = "formatting only"
    COORDINATES = "coordinates only"
    TEXT = "text"
    STRUCTURE = "structure"
    TABLES = "tables"
    UNPARSABLE = "unparsable"


# Order used in the rendered summary: the most review-relevant kinds first.
KIND_ORDER = (
    ChangeKind.STRUCTURE,
    ChangeKind.TABLES,
    ChangeKind.TEXT,
    ChangeKind.ADDED,
    ChangeKind.REMOVED,
    ChangeKind.UNPARSABLE,
    ChangeKind.COORDINATES,
    ChangeKind.FORMATTING,
)


@dataclass(slots=True)
class FileChange:
    path: str
    kind: ChangeKind
    details: list[str] = field(default_factory=list)


def is_groundtruth_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) > 2
        and parts[:2] == ("tests", "data")
        and (GROUNDTRUTH_SEGMENT in parts)
    )


def summarize_file(
    path: str, old: bytes | None, new: bytes | None, *, status: str
) -> FileChange:
    """Classify one changed reference file.

    `status` is the file status that GitHub reports. A missing blob is not
    enough to call a file added or removed: a failed read also gives None,
    and a modified file must not look like a new one.
    """
    if status == "added":
        if new is None:
            return FileChange(path, ChangeKind.UNPARSABLE, ["cannot read the file"])
        return FileChange(path, ChangeKind.ADDED)
    if status == "removed":
        if old is None:
            return FileChange(path, ChangeKind.UNPARSABLE, ["cannot read the file"])
        return FileChange(path, ChangeKind.REMOVED)
    if old is None or new is None:
        missing = [
            name for name, blob in (("base", old), ("head", new)) if blob is None
        ]
        return FileChange(
            path,
            ChangeKind.UNPARSABLE,
            [f"cannot read the {' and '.join(missing)} version"],
        )
    if path.endswith(".json"):
        return _summarize_json(path, old, new)
    return _summarize_text(path, old, new)


def _summarize_text(path: str, old: bytes, new: bytes) -> FileChange:
    old_text = old.decode("utf-8", errors="replace")
    new_text = new.decode("utf-8", errors="replace")
    if _normalized_lines(old_text) == _normalized_lines(new_text):
        return FileChange(path, ChangeKind.FORMATTING, ["whitespace only"])
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    if old_text.split() == new_text.split():
        # A joined or split paragraph keeps the words but changes the content.
        return FileChange(path, ChangeKind.TEXT, ["line or paragraph breaks changed"])
    removed = added = 0
    for line in difflib.unified_diff(old_lines, new_lines, n=0, lineterm=""):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            removed += 1
        elif line.startswith("+"):
            added += 1
    details = [f"+{added} / -{removed} lines"]
    kind = ChangeKind.TEXT
    if path.endswith(".md") and _table_lines(old_lines) != _table_lines(new_lines):
        kind = ChangeKind.TABLES
        details.append("Markdown table rows changed")
    return FileChange(path, kind, details)


def _normalized_lines(text: str) -> list[str]:
    """Drop trailing spaces and collapse runs of blank lines."""
    lines: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if line or (lines and lines[-1]):
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _table_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.lstrip().startswith("|")]


def _summarize_json(path: str, old: bytes, new: bytes) -> FileChange:
    try:
        old_doc = json.loads(old)
        new_doc = json.loads(new)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return FileChange(path, ChangeKind.UNPARSABLE, [f"invalid JSON: {exc}"])
    if old_doc == new_doc:
        return FileChange(path, ChangeKind.FORMATTING, ["same JSON value"])
    if _strip_geometry(old_doc) == _strip_geometry(new_doc):
        return FileChange(path, ChangeKind.COORDINATES)
    if not (_is_docling_document(old_doc) and _is_docling_document(new_doc)):
        return FileChange(path, ChangeKind.TEXT, ["JSON values differ"])
    return _summarize_docling_document(path, old_doc, new_doc)


def _strip_geometry(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_geometry(item)
            for key, item in value.items()
            if key not in GEOMETRY_KEYS
        }
    if isinstance(value, list):
        return [_strip_geometry(item) for item in value]
    return value


def _is_docling_document(value: Any) -> bool:
    return isinstance(value, dict) and "body" in value and "texts" in value


def _summarize_docling_document(
    path: str, old_doc: dict[str, Any], new_doc: dict[str, Any]
) -> FileChange:
    details: list[str] = []
    old_counts = _label_counts(old_doc)
    new_counts = _label_counts(new_doc)
    for label in sorted(old_counts.keys() | new_counts.keys()):
        before, after = old_counts[label], new_counts[label]
        if before != after:
            details.append(f"{label}: {before} → {after}")

    old_order = _reading_order(old_doc)
    new_order = _reading_order(new_doc)
    old_labels = [label for label, _ in old_order]
    new_labels = [label for label, _ in new_order]
    structure_changed = old_labels != new_labels
    if structure_changed and old_counts == new_counts:
        details.append("reading order changed")

    table_changes = _table_changes(old_doc, new_doc)
    details.extend(table_changes)

    changed_texts = 0
    if not structure_changed:
        changed_texts = sum(
            1 for (_, a), (_, b) in zip(old_order, new_order, strict=True) if a != b
        )
        if changed_texts:
            details.append(f"{changed_texts} text item(s) changed")

    if structure_changed:
        kind = ChangeKind.STRUCTURE
    elif table_changes:
        kind = ChangeKind.TABLES
    else:
        kind = ChangeKind.TEXT
        if not changed_texts:
            details.append("item attributes changed")
    return FileChange(path, kind, details)


def _label_counts(doc: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for collection in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        for item in doc.get(collection) or []:
            if isinstance(item, dict):
                counts[str(item.get("label", collection))] += 1
    return counts


def _reading_order(doc: dict[str, Any]) -> list[tuple[str, str]]:
    """Walk the body tree and return (label, normalized text) per item."""
    order: list[tuple[str, str]] = []
    seen: set[str] = set()
    stack = list(reversed(_children(doc.get("body"))))
    while stack:
        ref = stack.pop()
        if ref in seen:
            continue
        seen.add(ref)
        item = _resolve(doc, ref)
        if item is None:
            continue
        label = str(item.get("label", ""))
        text = " ".join(str(item.get("text", "")).split())
        order.append((label, text))
        stack.extend(reversed(_children(item)))
    return order


def _children(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return []
    refs: list[str] = []
    for child in item.get("children") or []:
        if isinstance(child, dict) and isinstance(child.get("$ref"), str):
            refs.append(child["$ref"])
    return refs


def _resolve(doc: dict[str, Any], ref: str) -> dict[str, Any] | None:
    parts = ref.lstrip("#/").split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    collection = doc.get(parts[0])
    index = int(parts[1])
    if not isinstance(collection, list) or index >= len(collection):
        return None
    item = collection[index]
    return item if isinstance(item, dict) else None


def _table_changes(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> list[str]:
    old_tables = [_table_signature(t) for t in old_doc.get("tables") or []]
    new_tables = [_table_signature(t) for t in new_doc.get("tables") or []]
    changes: list[str] = []
    for index, (old, new) in enumerate(zip(old_tables, new_tables, strict=False)):
        if old[0] != new[0]:
            changes.append(
                f"table {index}: shape {old[0][0]}x{old[0][1]} → {new[0][0]}x{new[0][1]}"
            )
        elif old[1] != new[1]:
            changed = sum(1 for a, b in zip(old[1], new[1], strict=True) if a != b)
            changes.append(f"table {index}: {changed} cell(s) changed")
    return changes


def _table_signature(table: Any) -> tuple[tuple[int, int], list[str]]:
    data = table.get("data") if isinstance(table, dict) else None
    if not isinstance(data, dict):
        return (0, 0), []
    shape = (int(data.get("num_rows") or 0), int(data.get("num_cols") or 0))
    cells = [
        " ".join(str(cell.get("text", "")).split())
        for cell in data.get("table_cells") or []
        if isinstance(cell, dict)
    ]
    return shape, cells


def render_markdown(changes: list[FileChange]) -> str:
    if not changes:
        return ""
    by_kind: dict[ChangeKind, list[FileChange]] = {}
    for change in changes:
        by_kind.setdefault(change.kind, []).append(change)
    lines = ["| Change | Files |", "|---|---|"]
    lines.extend(
        f"| {kind.value} | {len(by_kind[kind])} |"
        for kind in KIND_ORDER
        if kind in by_kind
    )
    lines.append("")
    listed = 0
    for kind in KIND_ORDER:
        for change in by_kind.get(kind, []):
            if listed >= MAX_LISTED_FILES:
                break
            detail = f": {'; '.join(change.details)}" if change.details else ""
            lines.append(f"- **{kind.value}** `{change.path}`{detail}")
            listed += 1
    if len(changes) > listed:
        lines.append(f"- … {len(changes) - listed} more file(s) not listed")
    return "\n".join(lines)
