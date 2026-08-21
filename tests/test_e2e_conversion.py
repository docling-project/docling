from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE, PageRange
from docling.document_converter import DocumentConverter, PdfFormatOption

from .groundtruth_paths import get_regular_groundtruth_paths
from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import check_conversion_result_v2

GENERATE_V2 = GEN_TEST_DATA
pytestmark = pytest.mark.ml_pdf_model

# PDFs that are tested separately in test_failed_pages.py (intentionally failing pages)
SKIP_E2E_TEST = ["skipped_1page.pdf", "skipped_2pages.pdf"]

# Page selection per source PDF, keyed by file name. Each value is an inclusive,
# 1-based `(first_page, last_page)` range; documents without an entry are converted
# in full. Every source longer than 5 pages gets an entry, so the ground truth stays
# small and the suite exercises page-offset handling instead of only full documents.
#
# The ground truth is tied to this selection: after changing an entry, regenerate the
# affected files with
# `DOCLING_GEN_TEST_DATA=1 uv run pytest tests/test_e2e_conversion.py` and review the
# result. Note that `2305.03393v1-pg9.pdf` shares its ground truth with
# tests/test_interfaces.py, which always converts that document in full.
PAGE_SELECTION: dict[str, PageRange] = {
    "2203.01017v2.pdf": (2, 5),
    "2206.01062.pdf": (2, 5),
    "2305.03393v1.pdf": (2, 5),
    "redp5110_sampled.pdf": (2, 5),
}


def get_page_range(pdf_path: Path) -> PageRange:
    """Pages to convert for `pdf_path`, defaulting to the whole document."""
    return PAGE_SELECTION.get(pdf_path.name, DEFAULT_PAGE_RANGE)


def get_pdf_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pdf/sources/")

    # List all PDF files in the directory and its subdirectories
    # Exclude PDFs that are tested separately for failure scenarios
    pdf_files = sorted(
        f for f in directory.rglob("*.pdf") if f.name not in SKIP_E2E_TEST
    )
    return pdf_files


def get_converter():
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
                backend=PdfFormatOption().backend,
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


def test_e2e_pdfs_conversions():
    pdf_paths = get_pdf_paths()
    converter = get_converter()

    # Each entry: (document, check, ok, error). Every document is converted and
    # verified, so one bad document does not hide the state of the others.
    results: list[tuple[str, str, bool, str]] = []

    for pdf_path in pdf_paths:
        page_range = get_page_range(pdf_path)
        selection = (
            "all pages"
            if page_range == DEFAULT_PAGE_RANGE
            else f"pages {page_range[0]}-{page_range[1]}"
        )
        print(f"converting {pdf_path} ({selection})")

        try:
            doc_result: ConversionResult = converter.convert(
                pdf_path, page_range=page_range
            )
            failures = check_conversion_result_v2(
                gt=get_regular_groundtruth_paths(pdf_path),
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
