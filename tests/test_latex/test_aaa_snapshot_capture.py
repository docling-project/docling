# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Temporary CI fixture capture; remove after updating groundtruth snapshots."""

import json
import subprocess
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from ..verify_utils import verify_document, verify_export

LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


def test_capture_updated_latex_groundtruth(latex_paths: list[Path]) -> None:
    converter = DocumentConverter(allowed_formats=[InputFormat.LATEX])
    for latex_path in latex_paths:
        if latex_path.parent.resolve() == LATEX_DATA_DIR.resolve():
            gt_name = latex_path.name
        else:
            gt_name = f"{latex_path.parent.name}_{latex_path.name}"
        gt_path = LATEX_DATA_DIR.parent / "groundtruth" / gt_name
        doc = converter.convert(latex_path).document
        verify_export(
            doc.export_to_markdown(compact_tables=True),
            str(gt_path) + ".md",
            generate=True,
        )
        verify_export(
            doc._export_to_indented_text(
                max_text_len=70, explicit_tables=False
            ),
            str(gt_path) + ".itxt",
            generate=True,
        )
        verify_document(doc, str(gt_path) + ".json", generate=True)

    patch = subprocess.run(
        ["git", "diff", "--unified=0", "--", "tests/data/latex/groundtruth"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    print("DOCFIX_BEGIN", flush=True)
    for line in patch.splitlines():
        print("DOCFIX_LINE:" + json.dumps(line, ensure_ascii=True), flush=True)
    print("DOCFIX_END", flush=True)
    raise AssertionError("Temporary fixture capture (remove after applying snapshot updates)")
