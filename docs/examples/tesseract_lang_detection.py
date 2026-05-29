# %% [markdown]
# Detect language automatically with Tesseract OCR using docling-serve and force full-age OCR.
#
# What this example does
# - Uses docling-serve API to perform OCR with Tesseract and automatic language detection.
# - Forces full-page OCR and prints the recognized text as Markdown.
#
# Prerequisites
# - Install Docling service client.
# - Set API_KEY and SERVICE_URL for your docling-serve instance.
#
# How to run
# - From the repo root: `python docs/examples/tesseract_lang_detection.py`.
# - Ensure the docling-serve backend is configured with Tesseract OCR.
# 
# Notes
# - The docling-serve API handles Tesseract OCR configuration server-side.
# - `ocr_preset="tesseract"` specifies Tesseract as the OCR engine.
# - `force_ocr=True` forces full-page OCR.
# - `ocr_lang=["auto"]` enables automatic language detection (server must support this).

# %%

from pathlib import Path
import os

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

from DoclingTests.servetest import SERVICE_URL

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
