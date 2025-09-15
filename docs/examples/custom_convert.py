# %% [markdown]
# Customize PDF conversion by toggling OCR/backends and pipeline options.
#
# What this example does
# - Shows several alternative configurations for the Docling PDF pipeline.
# - Lets you try OCR engines (EasyOCR, Tesseract, system OCR) or no OCR.
# - Converts a single sample PDF and exports results to `scratch/`.
#
# Prerequisites
# - Install Docling and its optional OCR backends per the docs.
# - Ensure you can import `docling` from your Python environment.
#
# How to run
# - From the repository root, run: `python docs/examples/custom_convert.py`.
# - Outputs are written under `scratch/` next to where you run the script.
#
# Choosing a configuration
# - Only one configuration block should be active at a time.
# - Uncomment exactly one of the sections below to experiment.
# - The file ships with "Docling Parse with EasyOCR" enabled as a sensible default.
# - If you uncomment a backend or OCR option that is not imported above, also
#   import its class, e.g.:
#   - `from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend`
#   - `from docling.datamodel.pipeline_options import TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions`
#
# Input document
# - Defaults to a single PDF from `tests/data/pdf/` in the repo.
# - If you don’t have the test data, update `input_doc_path` to a local PDF.
#
# Notes
# - EasyOCR language: adjust `pipeline_options.ocr_options.lang` (e.g., ["en"], ["es"], ["en", "de"]).
# - Accelerators: tune `AcceleratorOptions` to select CPU/GPU or threads.
# - Exports: JSON, plain text, Markdown, and doctags are saved in `scratch/`.

# %%

import json
import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    ###########################################################################

    # The sections below demo combinations of PdfPipelineOptions and backends.
    # Tip: Uncomment exactly one section at a time to compare outputs.

    # PyPdfium without EasyOCR
    # --------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = False

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # PyPdfium with EasyOCR
    # -----------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # Docling Parse without EasyOCR
    # -------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with EasyOCR (default)
    # -------------------------------
    # Enables OCR and table structure with EasyOCR, using automatic device
    # selection via AcceleratorOptions. Adjust languages as needed.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Docling Parse with EasyOCR (CPU only)
    # -------------------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.ocr_options.use_gpu = False  # <-- set this.
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract
    # ----------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract CLI
    # --------------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractCliOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with ocrmac (macOS only)
    # --------------------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = OcrMacOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    ###########################################################################

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export Docling document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Text format (plain text via Markdown export):
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown(strict_text=True))

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())


if __name__ == "__main__":
    main()
