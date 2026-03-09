# %% [markdown]
# Detect language automatically with Tesseract OCR and force full-page OCR.
#
# What this example does
# - Configures Tesseract (CLI in this snippet) with `lang=["auto"]`.
# - Forces full-page OCR and prints the recognized text as Markdown.
#
# How to run
# - From the repo root: `python docs/examples/tesseract_lang_detection.py`.
# - Ensure Tesseract CLI (or library) is installed and on PATH.
#
# Notes
# - You can switch to `TesseractOcrOptions` instead of `TesseractCliOcrOptions`.
# - Language packs must be installed; set `TESSDATA_PREFIX` if Tesseract
#   cannot find language data. Using `lang=["auto"]` requires traineddata
#   that supports script/language detection on your system.

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

    # Set lang=["auto"] with a tesseract OCR engine: TesseractOcrOptions, TesseractCliOcrOptions
    # ocr_options = TesseractOcrOptions(lang=["auto"])
    ocr_options = TesseractCliOcrOptions(lang=["auto"])

    pipeline_options = PdfPipelineOptions(
        do_ocr=True, force_full_page_ocr=True, ocr_options=ocr_options
    )

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
