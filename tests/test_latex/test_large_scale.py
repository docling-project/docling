# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Opt-in validation of LaTeX source files in secemp9/arxiv-complete.

Run with ``uv run pytest tests/test_latex/test_large_scale.py --large-scale``.
Install ``datasets`` and ``doclang`` first. Use ``--max-items N`` for a short run.
Pass ``--write-success`` to save valid output, original source bytes, and
LaTeX and JSON serialized from the resulting DoclingDocument.
The full stream contains millions of source files and can take a long time.
Completed files are appended to ``latex-manifest.jsonl``. Run the same command
again after an interruption to continue with the next unprocessed source file.
Add ``--recheck-failures`` to convert the files that failed earlier again, for
example after a fix. The latest result of each file replaces the earlier one.
``latex-failures.jsonl`` keeps every failure found, also failures that a later
recheck cleared.
The stream starts with shard 0, which holds the arXiv papers of 2007 and 2008.
Use ``--corpus-shard N`` (0-79, repeatable) to read other shards, and
``--corpus-dir DIR`` to write the output files outside the repository, for
example one folder for each of several parallel runs.
"""

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from docling_core.transforms.serializer.latex import LaTeXParams
from docling_core.types.doc import ContentLayer, DocItemLabel
from pydantic import BaseModel, ValidationError as PydanticValidationError

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .latex_serializer import (
    SourceDocLangSerializer,
    SourceLaTeXDocSerializer,
    SourceLaTeXInlineSerializer,
)

_DATASET = "secemp9/arxiv-complete"
_SHARD_COUNT = 80
_SOURCE_EXTENSIONS = {"tex", "ltx"}


def _shard_files(shards: list[int]) -> list[str]:
    return [
        f"latex/train-{shard:05d}-of-{_SHARD_COUNT:05d}.parquet" for shard in shards
    ]


class _ManifestEntry(BaseModel):
    paper_id: str
    path: str
    stages: list[str]


def _read_manifest(path: Path) -> dict[tuple[str, str], _ManifestEntry]:
    """Load completed files and discard a line interrupted during a write."""
    if not path.exists():
        return {}

    entries: dict[tuple[str, str], _ManifestEntry] = {}
    with path.open("r+b") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_number += 1
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            try:
                entry = _ManifestEntry.model_validate_json(line)
            except PydanticValidationError as exc:
                raise ValueError(
                    f"Invalid manifest line {line_number}: {path}"
                ) from exc
            entries[(entry.paper_id, entry.path)] = entry
    return entries


def _needs_check(entry: _ManifestEntry | None, recheck_failures: bool) -> bool:
    return entry is None or (recheck_failures and bool(entry.stages))


def _failure_counts(entries: Iterable[_ManifestEntry]) -> Counter[str]:
    return Counter(stage for entry in entries for stage in entry.stages)


def test_manifest_recovers_interrupted_write(tmp_path: Path) -> None:
    manifest_path = tmp_path / "latex-manifest.jsonl"
    entry = _ManifestEntry(paper_id="1234.5678", path="main.tex", stages=[])
    complete_line = (entry.model_dump_json() + "\n").encode()
    manifest_path.write_bytes(complete_line + b'{"paper_id":"unfinished"')

    assert _read_manifest(manifest_path) == {("1234.5678", "main.tex"): entry}
    assert manifest_path.read_bytes() == complete_line


def test_recheck_replaces_earlier_failure(tmp_path: Path) -> None:
    """A file that passes on a recheck no longer counts as a corpus failure."""
    failed = _ManifestEntry(paper_id="1234.5678", path="main.tex", stages=["parse"])
    passed = _ManifestEntry(paper_id="1234.5678", path="main.tex", stages=[])
    other = _ManifestEntry(paper_id="1234.5679", path="main.tex", stages=[])

    assert _needs_check(None, recheck_failures=False)
    assert not _needs_check(failed, recheck_failures=False)
    assert _needs_check(failed, recheck_failures=True)
    assert not _needs_check(other, recheck_failures=True)

    manifest_path = tmp_path / "latex-manifest.jsonl"
    manifest_path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in (failed, other, passed)),
        encoding="utf-8",
    )
    completed = _read_manifest(manifest_path)
    assert completed[("1234.5678", "main.tex")] == passed
    assert not _failure_counts(completed.values())


def test_shard_files_name_the_latex_parquet_files() -> None:
    assert _shard_files([0, 79]) == [
        "latex/train-00000-of-00080.parquet",
        "latex/train-00079-of-00080.parquet",
    ]


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
    recheck_failures: bool = pytestconfig.getoption("--recheck-failures")
    shards: list[int] = pytestconfig.getoption("--corpus-shard") or []
    if any(not 0 <= shard < _SHARD_COUNT for shard in shards):
        pytest.fail(f"--corpus-shard must be between 0 and {_SHARD_COUNT - 1}")
    corpus_dir: Path | None = pytestconfig.getoption("--corpus-dir")

    from datasets import load_dataset
    from doclang import ValidationError, validate

    output_dir = corpus_dir or Path(__file__).resolve().parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "latex-manifest.jsonl"
    completed = _read_manifest(manifest_path)
    report_path = output_dir / "latex-failures.jsonl"
    if completed:
        report_path.touch(exist_ok=True)
    else:
        report_path.write_text("", encoding="utf-8")
    dclg_path = tmp_path / "current.dclg"
    artifacts_dir = output_dir / "latex-failures"
    source_dir = artifacts_dir / "sources"
    invalid_dir = artifacts_dir / "invalid-dclg"
    success_dir = output_dir / "latex-success"
    if write_success:
        success_dir.mkdir(exist_ok=True)
    checked = 0
    row_stages: list[str] = []

    def finish_row(row: dict) -> None:
        entry = _ManifestEntry(
            paper_id=row["paper_id"], path=row["path"], stages=row_stages.copy()
        )
        with manifest_path.open("a", encoding="utf-8") as output:
            output.write(entry.model_dump_json() + "\n")
        completed[(entry.paper_id, entry.path)] = entry

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
        row_stages.append(stage)

    if shards:
        corpus = load_dataset(
            _DATASET, data_files=_shard_files(shards), split="train", streaming=True
        )
    else:
        corpus = load_dataset(_DATASET, "latex", split="train", streaming=True)
    with caplog.at_level(logging.WARNING, logger="docling.backend.latex"):
        for row in corpus:
            if max_items is not None and checked >= max_items:
                break
            if row["ext"] not in _SOURCE_EXTENSIONS:
                continue
            if not _needs_check(
                completed.get((row["paper_id"], row["path"])), recheck_failures
            ):
                continue
            checked += 1
            row_stages = []
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
                finish_row(row)
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
                if item.content_layer == ContentLayer.FURNITURE or item.label in {
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
                dclg = SourceDocLangSerializer(doc=document).serialize().text
                dclg_path.write_text(dclg, encoding="utf-8")
            except Exception as exc:
                report(row, "serialize", repr(exc))
                finish_row(row)
                continue

            try:
                root = ET.fromstring(dclg)
            except ET.ParseError as exc:
                report(row, "xsd", str(exc), dclg)
                finish_row(row)
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
                        layers={ContentLayer.BODY, ContentLayer.FURNITURE},
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
                    finish_row(row)
                    continue
                latex_id = _latex_id(row)
                try:
                    document.save_as_json(success_dir / f"{latex_id}.json")
                except Exception as exc:
                    report(row, "serialize_json", repr(exc))
                    finish_row(row)
                    continue
                (success_dir / f"{latex_id}.tex").write_bytes(row["content"])
                (success_dir / f"{latex_id}.dclg").write_text(dclg, encoding="utf-8")
                (success_dir / f"{latex_id}.parsed.tex").write_text(
                    parsed_latex, encoding="utf-8"
                )
            finish_row(row)

    summary = (
        f"Checked {checked} source files ({len(completed)} in the manifest); "
        f"manifest: {manifest_path}; report: {report_path}"
    )
    terminal_reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is None:
        print(summary)
    else:
        terminal_reporter.write_line(summary)
    assert completed, "The LaTeX stream contained no .tex or .ltx files"
    failures = _failure_counts(completed.values())
    assert not failures, f"Corpus failures: {dict(failures)}; report: {report_path}"
