# %% [markdown]
# Detect language automatically with Tesseract OCR and force full-page OCR.
#
# What this example does
# - Configures Tesseract (CLI in this snippet) with an empty `lang` list.
# - Forces full-page OCR and prints the recognized text as Markdown.
#
# How to run
# - From the repo root: `python docs/examples/tesseract_lang_detection.py`.
# - Ensure Tesseract CLI (or library) is installed and on PATH.
#
# Notes
# - You can switch to `TesseractOcrOptions` instead of `TesseractCliOcrOptions`.
# - Language packs must be installed; set `TESSDATA_PREFIX` if Tesseract
#   cannot find language data. An empty `lang` list means "let the engine
#   decide": for Tesseract that is per-page orientation and script detection,
#   which requires the `osd` traineddata plus the per-script files it may
#   detect.

# %%

import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
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
    input_doc_path = data_folder / "pdf/sources/2206.01062.pdf"

    # Set lang=[] with a tesseract OCR engine: TesseractOcrOptions, TesseractCliOcrOptions
    # ocr_options = TesseractOcrOptions(lang=[], mode=OcrMode.FULL_PAGE)
    ocr_options = TesseractCliOcrOptions(lang=[], mode=OcrMode.FULL_PAGE)

    pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)

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
