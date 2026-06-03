# %% [markdown]
# Customize PDF conversion by toggling OCR/backends and pipeline options.
#
# What this example does
# - Shows several alternative configurations for the Docling PDF pipeline.
# - Lets you try OCR engines (EasyOCR, Tesseract, system OCR) or no OCR.
# - Converts a single sample PDF and exports results to `scratch/`.
# - Lets you try either the Docling or Docling Serve implementations
#
# Prerequisites
# - Install Docling or Docling Serve and the docling optional OCR backends per the docs.
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
#
# Input document
# - Defaults to a sample PDF
# - Update `input_doc_path` to a desired file path
#
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

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMacOptions,
    PdfBackend,
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.service_client import DoclingServiceClient

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    def create_pipeline_options(
        do_ocr: bool, do_table_structure: bool, do_cell_matching: bool
    ) -> PdfPipelineOptions:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=do_cell_matching
        )
        return pipeline_options

    ###########################################################################

    # The sections below demo combinations of PdfPipelineOptions and backends.
    # Tip: Uncomment exactly one section at a time to compare outputs.

    # PyPdfium without EasyOCR
    # --------------------
    # pipeline_options = create_pipeline_options(do_ocr=False, do_table_structure=True, do_cell_matching=False)
    # backend = PyPdfiumDocumentBackend

    # PyPdfium with EasyOCR
    # -----------------
    # pipeline_options = create_pipeline_options(do_ocr=True, do_table_structure=True, do_cell_matching=True)
    # backend = PyPdfiumDocumentBackend

    # Docling Parse without EasyOCR
    # -------------------------
    # pipeline_options = create_pipeline_options(do_ocr=False, do_table_structure=True, do_cell_matching=True)
    # backend = None

    # Docling Parse with EasyOCR (default)
    # -------------------------------
    # Enables OCR and table structure with EasyOCR, using automatic device
    # selection via AcceleratorOptions. Adjust languages as needed.
    pipeline_options = create_pipeline_options(
        do_ocr=True, do_table_structure=True, do_cell_matching=True
    )
    pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )
    backend = None

    # Docling Parse with EasyOCR (CPU only)
    # -------------------------------------
    # pipeline_options = create_pipeline_options(do_ocr=True, do_table_structure=True, do_cell_matching=True)
    # pipeline_options.ocr_options.use_gpu = False
    # backend = None

    # Docling Parse with Tesseract
    # ----------------------------
    # pipeline_options = create_pipeline_options(do_ocr=True, do_table_structure=True, do_cell_matching=True)
    # pipeline_options.ocr_options = TesseractOcrOptions()
    # backend = None

    # Docling Parse with Tesseract CLI
    # --------------------------------
    # pipeline_options = create_pipeline_options(do_ocr=True, do_table_structure=True, do_cell_matching=True)
    # pipeline_options.ocr_options = TesseractCliOcrOptions()
    # backend = None

    # Docling Parse with ocrmac (macOS only)
    # --------------------------------------
    # pipeline_options = create_pipeline_options(do_ocr=True, do_table_structure=True, do_cell_matching=True)
    # pipeline_options.ocr_options = OcrMacOptions()
    # backend = None

    ###########################################################################
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    start_time = time.time()

    ###########################################################################

    # The two sections below are for either direct Docling usage or Docling Serve
    # Uncomment exactly one section at a time

    # Docling (local conversion)
    # --------------------------------------
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=backend
            )
            if backend
            else PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    conv_result = doc_converter.convert(input_doc_path)

    # Docling Serve (remote conversion)
    # --------------------------------------
    # SERVE_URL = "http://localhost:5001"
    # table_cell_matching = getattr(pipeline_options.table_structure_options, 'do_cell_matching', True) if pipeline_options.table_structure_options else True
    # pdf_backend = PdfBackend.PYPDFIUM2 if backend == PyPdfiumDocumentBackend else PdfBackend.DOCLING_PARSE
    # options = ConvertDocumentsOptions(
    #         do_ocr=pipeline_options.do_ocr,
    #         do_table_structure=pipeline_options.do_table_structure,
    #         table_cell_matching=table_cell_matching,
    #         ocr_lang=pipeline_options.ocr_options.lang if pipeline_options.do_ocr else None,
    #         pdf_backend=pdf_backend,
    #     )
    # with DoclingServiceClient(url=SERVE_URL) as client:
    #     conv_result = client.convert(input_doc_path, options=options)

    ###########################################################################
    _log.info(f"Document converted in {time.time() - start_time:.2f} seconds.")

    # Export results
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = Path(input_doc_path).stem

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
        fp.write(conv_result.document.export_to_doctags())


if __name__ == "__main__":
    main()
