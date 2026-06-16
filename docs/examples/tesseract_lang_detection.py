# %% [markdown]
# Detect language automatically with Tesseract OCR and force full-page OCR.
#
# What this example does
# - Configures Tesseract (CLI in this snippet) with `lang=["auto"]`.
# - Forces full-page OCR and prints the recognized text as Markdown.
#
# Prerequisites  
# - Install Docling or Docling Serve.
# - If working with Docling Serve, set API_KEY and SERVICE_URL for your docling-serve instance.
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
# - If working with docling-serve, the API handles Tesseract OCR configuration server-side.

# %% [markdown]
# Using Docling:

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


# %% [markdown]
# Using Docling Serve:

# %%

from pathlib import Path
import os
from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

# Configure your SERVICE_URL, and your API_KEY should your service need authentication.
SERVICE_URL = os.environ["DOCLING_SERVICE_URL"]
API_KEY = os.environ.get("DOCLING_SERVICE_API_KEY")

def main():
    data_folder = Path(__file__).parent / "../.../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    with DoclingServiceClient(url=SERVICE_URL, api_key=API_KEY) as client:
        result = client.convert(
            source=input_doc_path,
            options=ConvertDocumentsOptions(
                do_ocr=True,              # Enable OCR
                force_ocr=True,           # Force full-page OCR (equivalent to force_full_page_ocr)
                ocr_preset="tesseract",   # Specify Tesseract as the OCR engine
                ocr_lang=["auto"],        # Auto language detection (equivalent to lang=["auto"])
                to_formats=[OutputFormat.MARKDOWN],  # Request markdown output
            ),
        )

        # Export and print the markdown
        md = result.document.export_to_markdown()
        print(md)


if __name__ == "__main__":
    main()
