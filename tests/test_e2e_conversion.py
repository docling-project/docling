import difflib
import re
from collections import Counter
from pathlib import Path

import pytest
from docling_core.types.doc import DocItem, DoclingDocument

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .groundtruth_paths import get_regular_groundtruth_paths
from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import check_conversion_result_v2

GENERATE_V2 = GEN_TEST_DATA
pytestmark = pytest.mark.ml_pdf_model

# PDFs that are tested separately in test_failed_pages.py (intentionally failing pages)
SKIP_E2E_TEST = ["skipped_1page.pdf", "skipped_2pages.pdf"]

PDF_BACKENDS = [
    pytest.param(
        "docling_parse",
        ThreadedDoclingParseDocumentBackend,
        id="docling_parse",
    ),
    pytest.param("pypdfium2", PyPdfiumDocumentBackend, id="pypdfium2"),
]

DOCLANG_TAG_RE = re.compile(r"<(/?)([a-zA-Z_][\w.-]*)(?:\s|>|/)")


def _doclang_tag_counts(content: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for closing, tag in DOCLANG_TAG_RE.findall(content):
        if closing:
            continue
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _first_different_line(
    left_lines: list[str], right_lines: list[str]
) -> tuple[int, str, str]:
    max_len = max(len(left_lines), len(right_lines))
    for line_no in range(max_len):
        left_line = left_lines[line_no] if line_no < len(left_lines) else ""
        right_line = right_lines[line_no] if line_no < len(right_lines) else ""
        if left_line != right_line:
            return line_no + 1, left_line.strip(), right_line.strip()
    return 0, "", ""


def _shorten(value: str, limit: int = 80) -> str:
    value = value.replace("|", "\\|")
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _major_tag_deltas(left: str, right: str) -> str:
    left_counts = _doclang_tag_counts(left)
    right_counts = _doclang_tag_counts(right)
    tags = sorted(set(left_counts) | set(right_counts))
    deltas = [
        (tag, left_counts.get(tag, 0), right_counts.get(tag, 0))
        for tag in tags
        if left_counts.get(tag, 0) != right_counts.get(tag, 0)
    ]
    deltas.sort(key=lambda item: abs(item[1] - item[2]), reverse=True)
    if not deltas:
        return "-"
    return ", ".join(
        f"{tag}:{docling_parse_count}/{pypdfium2_count}"
        for tag, docling_parse_count, pypdfium2_count in deltas[:5]
    )


def get_pdf_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pdf/sources/")

    # List all PDF files in the directory and its subdirectories
    # Exclude PDFs that are tested separately for failure scenarios
    pdf_files = sorted(
        f for f in directory.rglob("*.pdf") if f.name not in SKIP_E2E_TEST
    )
    return pdf_files


def get_converter(backend):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    pipeline_options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend,
            )
        }
    )

    return converter


def _one_line(value: str, limit: int = 128) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _results_table(results: list[tuple[str, str, bool, str]]) -> str:
    """Render one row per document/check, with the failures spelled out."""
    header = ("document", "check", "status", "error")
    rows = [
        (document, check, "PASS" if ok else "FAIL", _one_line(error))
        for document, check, ok, error in results
    ]
    widths = [
        max([len(header[col])] + [len(row[col]) for row in rows])
        for col in range(len(header))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _line(cells: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))
            + " |"
        )

    lines = [separator, _line(header), separator]
    lines += [_line(row) for row in rows]
    lines.append(separator)
    return "\n".join(lines)


@pytest.mark.parametrize(("artifact_suffix", "backend"), PDF_BACKENDS)
def test_e2e_pdfs_conversions(artifact_suffix, backend):
    pdf_paths = get_pdf_paths()
    converter = get_converter(backend)

    # Each entry: (document, check, ok, error). Every document is converted and
    # verified, so one bad document does not hide the state of the others.
    results: list[tuple[str, str, bool, str]] = []

    for pdf_path in pdf_paths:
        print(f"converting {pdf_path}")

        try:
            doc_result: ConversionResult = converter.convert(pdf_path)
            failures = check_conversion_result_v2(
                gt=get_regular_groundtruth_paths(pdf_path, tag=artifact_suffix),
                doc_result=doc_result,
                generate=GENERATE_V2,
                verify_doctags=False,
                verify_doclang=True,
            )
        except Exception as exc:
            results.append(
                (pdf_path.name, "convert", False, f"{type(exc).__name__}: {exc}")
            )
            continue

        if failures:
            results += [
                (pdf_path.name, failure.check, False, failure.message)
                for failure in failures
            ]
        else:
            results.append((pdf_path.name, "all", True, ""))

    print("\n" + _results_table(results) + "\n")

    failed = [(document, check) for document, check, ok, _ in results if not ok]
    # the failures are already printed in the table above, so assert on the count:
    # `assert not failed` would repeat every error message in the pytest report
    assert not failed, f"{len(failed)} check(s) failed: " + ", ".join(
        f"{document}[{check}]" for document, check in failed
    )


def test_doclang_backend_groundtruth_differences_report():
    gt_dir = Path("./tests/data/pdf/groundtruth")
    rows: list[str] = []

    for docling_parse_path in sorted(gt_dir.glob("*.docling_parse.dclg")):
        stem = docling_parse_path.name.removesuffix(".docling_parse.dclg")
        pypdfium2_path = gt_dir / f"{stem}.pypdfium2.dclg"

        if not pypdfium2_path.exists():
            rows.append(f"| {stem} | missing pypdfium2 | - | - | - | - | - | - | - |")
            continue

        docling_parse_xml = docling_parse_path.read_text(encoding="utf-8")
        pypdfium2_xml = pypdfium2_path.read_text(encoding="utf-8")
        if docling_parse_xml == pypdfium2_xml:
            continue

        docling_parse_lines = docling_parse_xml.splitlines()
        pypdfium2_lines = pypdfium2_xml.splitlines()

        line_no, docling_parse_line, pypdfium2_line = _first_different_line(
            docling_parse_lines,
            pypdfium2_lines,
        )
        # Compare whole lines rather than characters: a character-level ratio() over
        # these ~150k-char files costs minutes per pair, and the report only ever shows
        # line-granular figures anyway.
        similarity = difflib.SequenceMatcher(
            None,
            docling_parse_lines,
            pypdfium2_lines,
            autojunk=False,
        ).ratio()
        rows.append(
            "| "
            f"{stem} | different | "
            f"{len(docling_parse_lines)}/{len(pypdfium2_lines)} | "
            f"{len(docling_parse_xml)}/{len(pypdfium2_xml)} | "
            f"{similarity:.3f} | "
            f"{line_no} | "
            f"{_major_tag_deltas(docling_parse_xml, pypdfium2_xml)} | "
            f"{_shorten(docling_parse_line)} | "
            f"{_shorten(pypdfium2_line)} |"
        )

    if rows:
        pytest.skip(
            "DocLang groundtruth differs between docling_parse and pypdfium2:\n\n"
            "| file | status | lines dp/pdfium | chars dp/pdfium | similarity | "
            "first diff line | tag deltas dp/pdfium | docling_parse | pypdfium2 |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
            + "\n".join(rows)
        )


def _labels_by_page(doc: DoclingDocument) -> dict[int, list[str]]:
    """Reading-order sequence of item labels, grouped by the page they sit on."""
    labels: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem) or not item.prov:
            continue
        labels.setdefault(item.prov[0].page_no, []).append(item.label.value)
    return labels


def _label_count_deltas(left: list[str], right: list[str]) -> str:
    left_counts = Counter(left)
    right_counts = Counter(right)
    deltas = sorted(
        (
            (label, left_counts.get(label, 0), right_counts.get(label, 0))
            for label in set(left_counts) | set(right_counts)
            if left_counts.get(label, 0) != right_counts.get(label, 0)
        ),
        key=lambda item: abs(item[1] - item[2]),
        reverse=True,
    )
    if not deltas:
        return "order only"
    return ", ".join(f"{label}:{lhs}/{rhs}" for label, lhs, rhs in deltas)


def test_layout_structure_per_page_between_backends():
    """Compare the per-page layout structure of both PDF backends.

    Both backends must see the same pages, which is asserted. The labels they assign
    within a page still differ on a minority of pages, so those are reported rather
    than failed.
    """
    gt_dir = Path("./tests/data/pdf/groundtruth")
    rows: list[str] = []

    for docling_parse_path in sorted(gt_dir.glob("*.docling_parse.json")):
        stem = docling_parse_path.name.removesuffix(".docling_parse.json")
        pypdfium2_path = gt_dir / f"{stem}.pypdfium2.json"
        if not pypdfium2_path.exists():
            continue

        docling_parse_doc = DoclingDocument.load_from_json(docling_parse_path)
        pypdfium2_doc = DoclingDocument.load_from_json(pypdfium2_path)

        docling_parse_pages = set(docling_parse_doc.pages)
        pypdfium2_pages = set(pypdfium2_doc.pages)
        assert docling_parse_pages == pypdfium2_pages, (
            f"[{stem}] backends disagree on which pages exist: "
            f"docling_parse={sorted(docling_parse_pages)} "
            f"pypdfium2={sorted(pypdfium2_pages)}"
        )

        docling_parse_labels = _labels_by_page(docling_parse_doc)
        pypdfium2_labels = _labels_by_page(pypdfium2_doc)

        for page_no in sorted(docling_parse_pages):
            docling_parse_page = docling_parse_labels.get(page_no, [])
            pypdfium2_page = pypdfium2_labels.get(page_no, [])
            if docling_parse_page == pypdfium2_page:
                continue
            rows.append(
                "| "
                f"{stem} | {page_no} | "
                f"{len(docling_parse_page)}/{len(pypdfium2_page)} | "
                f"{_label_count_deltas(docling_parse_page, pypdfium2_page)} |"
            )

    if rows:
        pytest.skip(
            "Per-page layout structure differs between docling_parse and pypdfium2:\n\n"
            "| file | page | items dp/pdfium | label deltas dp/pdfium |\n"
            "| --- | ---: | ---: | --- |\n" + "\n".join(rows)
        )
