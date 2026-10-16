# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Opt-in validation of LaTeX source files in secemp9/arxiv-complete.

Run with ``uv run pytest tests/test_latex/test_large_scale.py --large-scale``.
Install ``datasets`` and ``doclang`` first. Use ``--max-items N`` for a short run.
Pass ``--write-success`` to save valid output, original source bytes, and
LaTeX and JSON serialized from the resulting DoclingDocument.
The full stream contains millions of source files and can take a long time.
"""

import hashlib
import json
import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from docling_core.transforms.serializer.doclang import DocLangParams
from docling_core.transforms.serializer.latex import LaTeXParams
from docling_core.types.doc import ContentLayer, DocItemLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .latex_serializer import (
    SourceDocLangSerializer,
    SourceLaTeXDocSerializer,
    SourceLaTeXInlineSerializer,
)

_DATASET = "secemp9/arxiv-complete"
_SOURCE_EXTENSIONS = {"tex", "ltx"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _latex_id(row: dict) -> str:
    paper_id = row["paper_id"].replace("/", "_")
    path_digest = hashlib.sha256(row["path"].encode()).hexdigest()
    return f"{paper_id}-{path_digest}"


def _has_inline_formula_content(root: ET.Element) -> bool:
    text_elements = {"text", "heading", "caption", "footnote"}
    parents = {child: parent for parent in root.iter() for child in parent}
    for formula in root.iter():
        if _local_name(formula.tag) != "formula":
            continue
        if not any(_local_name(child.tag) == "content" for child in formula.iter()):
            continue
        ancestor = parents.get(formula)
        while ancestor is not None:
            if _local_name(ancestor.tag) in text_elements:
                return True
            ancestor = parents.get(ancestor)
    return False


def test_arxiv_latex_corpus(
    pytestconfig: pytest.Config, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Report each parse, XSD, and consistency failure in the streamed corpus."""
    if not pytestconfig.getoption("--large-scale"):
        pytest.skip("Use --large-scale to run the arXiv LaTeX corpus test")

    max_items: int | None = pytestconfig.getoption("--max-items")
    if max_items is not None and max_items <= 0:
        pytest.fail("--max-items must be a positive integer")
    write_success: bool = pytestconfig.getoption("--write-success")

    from datasets import load_dataset
    from doclang import ValidationError, validate

    repo_root = Path(__file__).resolve().parents[2]
    report_path = repo_root / "latex-failures.jsonl"
    report_path.write_text("", encoding="utf-8")
    dclg_path = tmp_path / "current.dclg"
    artifacts_dir = repo_root / "latex-failures"
    source_dir = artifacts_dir / "sources"
    invalid_dir = artifacts_dir / "invalid-dclg"
    success_dir = repo_root / "latex-success"
    if write_success:
        success_dir.mkdir(exist_ok=True)
    failures: Counter[str] = Counter()
    checked = 0

    def report(row: dict, stage: str, error: str, dclg: str | None = None) -> None:
        entry = {
            "paper_id": row["paper_id"],
            "path": row["path"],
            "stage": stage,
            "error": error,
        }
        digest = hashlib.sha256(f"{row['paper_id']}/{row['path']}".encode()).hexdigest()
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"{digest}.{row['ext']}"
        source_path.write_bytes(row["content"])
        entry["source"] = str(source_path)
        if dclg is not None:
            invalid_dir.mkdir(parents=True, exist_ok=True)
            artifact = invalid_dir / f"{digest}.dclg"
            artifact.write_text(dclg, encoding="utf-8")
            entry["dclg"] = str(artifact)
        with report_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        failures[stage] += 1

    corpus = load_dataset(_DATASET, "latex", split="train", streaming=True)
    with caplog.at_level(logging.WARNING, logger="docling.backend.latex"):
        for row in corpus:
            if max_items is not None and checked >= max_items:
                break
            if row["ext"] not in _SOURCE_EXTENSIONS:
                continue
            checked += 1
            caplog.clear()

            source = BytesIO(row["content"])
            try:
                input_doc = InputDocument(
                    path_or_stream=source,
                    format=InputFormat.LATEX,
                    backend=LatexDocumentBackend,
                    filename=Path(row["path"]).name,
                )
                backend = LatexDocumentBackend(in_doc=input_doc, path_or_stream=source)
                if not backend.is_valid():
                    raise ValueError("The LaTeX source is empty")
                document = backend.convert()
            except Exception as exc:
                report(row, "parse", repr(exc))
                continue

            backend_errors = [
                record.getMessage()
                for record in caplog.records
                if record.name.startswith("docling.backend.latex")
                and record.levelno >= logging.WARNING
            ]
            if backend_errors:
                report(row, "parse", "; ".join(backend_errors))
            row_failed = bool(backend_errors)

            for item in document.texts:
                if item.content_layer == ContentLayer.INVISIBLE or item.label in {
                    DocItemLabel.FORMULA,
                    DocItemLabel.CODE,
                }:
                    continue
                if any(char in item.text or char in item.orig for char in "\r\n\t"):
                    report(
                        row,
                        "consistency",
                        f"Source line break or tab in text item {item.self_ref}",
                    )
                    row_failed = True
                    break
            if any(
                char in cell.text
                for table in document.tables
                for cell in table.data.table_cells
                for char in "\r\n\t"
            ):
                report(row, "consistency", "Source line break or tab in table cell")
                row_failed = True

            try:
                # The current DocLang XSD does not define the invisible layer.
                # Keep source commands in the document and JSON, but omit them
                # from the XSD-validated corpus export.
                dclg = (
                    SourceDocLangSerializer(
                        doc=document,
                        params=DocLangParams(
                            layers={
                                ContentLayer.BODY,
                                ContentLayer.FURNITURE,
                                ContentLayer.BACKGROUND,
                            }
                        ),
                    )
                    .serialize()
                    .text
                )
                dclg_path.write_text(dclg, encoding="utf-8")
            except Exception as exc:
                report(row, "serialize", repr(exc))
                continue

            try:
                root = ET.fromstring(dclg)
            except ET.ParseError as exc:
                report(row, "xsd", str(exc), dclg)
                continue

            if _has_inline_formula_content(root):
                report(row, "consistency", "Inline <formula> contains <content>", dclg)
                row_failed = True

            try:
                validate(dclg_path, allow_empty_namespace=True, xsd_only=True)
            except ValidationError as exc:
                report(row, "xsd", str(exc), dclg)
                row_failed = True

            if write_success and not row_failed:
                try:
                    params = LaTeXParams(
                        layers={ContentLayer.BODY, ContentLayer.INVISIBLE},
                        include_annotations=False,
                        enable_chart_tables=False,
                        document_class="",
                        packages=[],
                    )
                    serializer = SourceLaTeXDocSerializer(
                        doc=document,
                        params=params,
                        inline_serializer=SourceLaTeXInlineSerializer(),
                    )
                    parsed_latex = serializer.serialize().text
                except Exception as exc:
                    report(row, "serialize_latex", repr(exc))
                    continue
                latex_id = _latex_id(row)
                try:
                    document.save_as_json(success_dir / f"{latex_id}.json")
                except Exception as exc:
                    report(row, "serialize_json", repr(exc))
                    continue
                (success_dir / f"{latex_id}.tex").write_bytes(row["content"])
                (success_dir / f"{latex_id}.dclg").write_text(dclg, encoding="utf-8")
                (success_dir / f"{latex_id}.parsed.tex").write_text(
                    parsed_latex, encoding="utf-8"
                )

    summary = f"Checked {checked} source files; report: {report_path}"
    terminal_reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is None:
        print(summary)
    else:
        terminal_reporter.write_line(summary)
    assert checked > 0, "The LaTeX stream contained no .tex or .ltx files"
    assert not failures, f"Corpus failures: {dict(failures)}; report: {report_path}"
