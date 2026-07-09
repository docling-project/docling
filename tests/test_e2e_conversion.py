import difflib
import re
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .groundtruth_paths import get_regular_groundtruth_paths
from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

GENERATE_V2 = GEN_TEST_DATA
pytestmark = pytest.mark.ml_pdf_model

SKIP_DOCTAGS_COMPARISON = ["2203.01017v2.pdf"]

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


def _first_different_line(left: str, right: str) -> tuple[int, str, str]:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
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


@pytest.mark.parametrize(("artifact_suffix", "backend"), PDF_BACKENDS)
def test_e2e_pdfs_conversions(artifact_suffix, backend):
    pdf_paths = get_pdf_paths()
    converter = get_converter(backend)

    for pdf_path in pdf_paths:
        print(f"converting {pdf_path}")

        doc_result: ConversionResult = converter.convert(pdf_path)

        # Decide if to skip doctags comparison
        verify_doctags = pdf_path.name not in SKIP_DOCTAGS_COMPARISON

        verify_conversion_result_v2(
            gt=get_regular_groundtruth_paths(pdf_path, tag=artifact_suffix),
            doc_result=doc_result,
            generate=GENERATE_V2,
            verify_doctags=verify_doctags,
            verify_doclang=True,
        )


def test_doclang_backend_groundtruth_differences_report():
    gt_dir = Path("./tests/data/pdf/groundtruth")
    rows: list[str] = []

    for docling_parse_path in sorted(gt_dir.glob("*.docling_parse.doclang.xml")):
        stem = docling_parse_path.name.removesuffix(".docling_parse.doclang.xml")
        pypdfium2_path = gt_dir / f"{stem}.pypdfium2.doclang.xml"

        if not pypdfium2_path.exists():
            rows.append(
                f"| {stem} | missing pypdfium2 | - | - | - | - | - | - | - |"
            )
            continue

        docling_parse_xml = docling_parse_path.read_text(encoding="utf-8")
        pypdfium2_xml = pypdfium2_path.read_text(encoding="utf-8")
        if docling_parse_xml == pypdfium2_xml:
            continue

        line_no, docling_parse_line, pypdfium2_line = _first_different_line(
            docling_parse_xml,
            pypdfium2_xml,
        )
        similarity = difflib.SequenceMatcher(
            None,
            docling_parse_xml,
            pypdfium2_xml,
            autojunk=False,
        ).ratio()
        rows.append(
            "| "
            f"{stem} | different | "
            f"{len(docling_parse_xml.splitlines())}/{len(pypdfium2_xml.splitlines())} | "
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
