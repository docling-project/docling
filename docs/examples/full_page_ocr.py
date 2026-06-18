# %% [markdown]
# Force full-page OCR on a PDF using different OCR backends.
#
# What this example does
# - Enables full-page OCR and table structure extraction for a sample PDF.
# - Demonstrates how to switch between OCR backends via `ocr_options`.
#
# Prerequisites
# - Install Docling and the desired OCR backend's dependencies (Tesseract, EasyOCR,
#   RapidOCR, or macOS OCR).
#
# How to run
# - From the repo root: `python docs/examples/full_page_ocr.py`.
# - The script prints Markdown text to stdout.
#
# Choosing an OCR backend
# - Uncomment one `ocr_options = ...` line below. Exactly one should be active.
# - `force_full_page_ocr=True` processes each page purely via OCR (often slower
#   than hybrid detection). Use when layout extraction is unreliable or the PDF
#   contains scanned pages.
# - If you switch OCR backends, ensure the corresponding option class is imported,
#   e.g., `EasyOcrOptions`, `TesseractOcrOptions`, `OcrMacOptions`, `RapidOcrOptions`.
#
# Input document
# - Defaults to `tests/data/pdf/2206.01062.pdf`. Change `input_doc_path` as needed.

# %%

import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import DocumentConverter, PdfFormatOption

# Under CI we limit the conversion to a representative page range to keep the
# example fast; locally the full document is processed.
IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")
CI_PAGE_RANGE = (3, 4)


def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )

    # Any of the OCR options can be used: EasyOcrOptions, TesseractOcrOptions,
    # TesseractCliOcrOptions, OcrMacOptions (macOS only), RapidOcrOptions
    # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
    # ocr_options = OcrMacOptions(force_full_page_ocr=True)
    # ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    page_range = CI_PAGE_RANGE if IS_CI else DEFAULT_PAGE_RANGE
    doc = converter.convert(input_doc_path, page_range=page_range).document
    md = doc.export_to_markdown()
    print(md)


if __name__ == "__main__":
    main()
