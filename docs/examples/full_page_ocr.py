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

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

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

    doc = converter.convert(input_doc_path).document
    md = doc.export_to_markdown()
    print(md)


if __name__ == "__main__":
    main()
