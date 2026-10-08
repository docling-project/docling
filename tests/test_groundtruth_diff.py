# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "groundtruth_diff",
    Path(__file__).resolve().parents[1] / ".github/scripts/groundtruth_diff.py",
)
assert SPEC is not None and SPEC.loader is not None
gt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gt
SPEC.loader.exec_module(gt)

DOC_PATH = "tests/data/groundtruth/docling_v2/sample.json"


def make_doc() -> dict[str, Any]:
    prov = [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]
    return {
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/tables/0"}]},
        "texts": [
            {"label": "section_header", "text": "Intro", "prov": prov, "children": []},
            {"label": "text", "text": "Hello world", "prov": prov, "children": []},
        ],
        "tables": [
            {
                "label": "table",
                "prov": prov,
                "children": [{"$ref": "#/texts/1"}],
                "data": {
                    "num_rows": 1,
                    "num_cols": 2,
                    "table_cells": [{"text": "a"}, {"text": "b"}],
                },
            }
        ],
        "pages": {"1": {"size": {"width": 10, "height": 10}}},
    }


def summarize(old: dict[str, Any], new: dict[str, Any]) -> Any:
    return gt.summarize_file(
        DOC_PATH, json.dumps(old).encode(), json.dumps(new, indent=2).encode()
    )


def test_only_files_below_test_data_groundtruth_count() -> None:
    assert gt.is_groundtruth_path("tests/data/docx/groundtruth/a.md")
    assert gt.is_groundtruth_path("tests/data/groundtruth/docling_v2/a.json")
    assert not gt.is_groundtruth_path("docs/groundtruth/a.md")
    assert not gt.is_groundtruth_path("tests/data/docx/a.docx")


def test_reformatted_json_and_moved_boxes_are_not_content_changes() -> None:
    old = make_doc()
    assert summarize(old, copy.deepcopy(old)).kind == gt.ChangeKind.FORMATTING

    moved = copy.deepcopy(old)
    moved["texts"][0]["prov"][0]["bbox"]["l"] = 5
    moved["pages"]["1"]["size"]["width"] = 11
    assert summarize(old, moved).kind == gt.ChangeKind.COORDINATES


def test_text_table_and_reading_order_changes_are_classified() -> None:
    old = make_doc()

    text = copy.deepcopy(old)
    text["texts"][1]["text"] = "Hello there"
    change = summarize(old, text)
    assert change.kind == gt.ChangeKind.TEXT
    assert change.details == ["1 text item(s) changed"]

    cells = copy.deepcopy(old)
    cells["tables"][0]["data"]["table_cells"][1]["text"] = "c"
    assert summarize(old, cells).details == ["table 0: 1 cell(s) changed"]

    order = copy.deepcopy(old)
    order["body"]["children"].reverse()
    change = summarize(old, order)
    assert change.kind == gt.ChangeKind.STRUCTURE
    assert "reading order changed" in change.details

    relabel = copy.deepcopy(old)
    relabel["texts"][0]["label"] = "title"
    change = summarize(old, relabel)
    assert change.kind == gt.ChangeKind.STRUCTURE
    assert "section_header: 1 → 0" in change.details


def test_markdown_whitespace_and_table_rows() -> None:
    path = "tests/data/md/groundtruth/a.md"
    old = b"# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert gt.summarize_file(path, old, old + b"\n\n").kind == gt.ChangeKind.FORMATTING
    joined = gt.summarize_file(path, b"one\n\ntwo\n", b"one two\n")
    assert joined.kind == gt.ChangeKind.TEXT
    assert joined.details == ["line or paragraph breaks changed"]
    change = gt.summarize_file(path, old, old.replace(b"| 1 |", b"| 9 |"))
    assert change.kind == gt.ChangeKind.TABLES
    assert change.details == ["+1 / -1 lines", "Markdown table rows changed"]


def test_render_lists_relevant_kinds_first() -> None:
    changes = [
        gt.FileChange("f1.json", gt.ChangeKind.FORMATTING),
        gt.FileChange("f2.json", gt.ChangeKind.STRUCTURE, ["reading order changed"]),
    ]
    lines = gt.render_markdown(changes).splitlines()
    assert lines[2] == "| structure | 1 |"
    assert lines[5] == "- **structure** `f2.json`: reading order changed"
