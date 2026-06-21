import logging
from pathlib import Path
from typing import List, Tuple

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.ocr.nemotron_ocr_model import NemotronOcrModel

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

_log = logging.getLogger(__name__)

GENERATE_V2 = GEN_TEST_DATA


def _nemotron_available() -> bool:
    """Reuse the model's own runtime gate; only the package import is extra.

    `NemotronOcrModel.validate_runtime` covers the OS/arch/Python/CUDA checks
    but does not verify that the optional `nemotron_ocr` package is installed,
    so that import is probed separately here.
    """
    try:
        import nemotron_ocr.inference.pipeline_v2
    except ImportError as exc:
        _log.warning("Nemotron OCR package is not importable: %s", exc)
        return False
    try:
        NemotronOcrModel.validate_runtime(
            AcceleratorOptions(device=AcceleratorDevice.AUTO)
        )
    except RuntimeError as exc:
        _log.warning("Nemotron OCR runtime validation failed: %s", exc)
        return False
    return True


pytestmark = [
    pytest.mark.ml_ocr,
    pytest.mark.skipif(
        not _nemotron_available(),
        reason="Nemotron OCR requires Linux x86_64, Python 3.12 and CUDA 13.x.",
    ),
]


def get_pdf_paths():
    # Define the directory you want to search
    directory = Path("./tests/data_scanned")

    # List all PDF files in the directory and its subdirectories
    pdf_files = sorted(directory.rglob("ocr_test*.pdf"))

    return pdf_files


def get_converter(ocr_options: OcrOptions):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = ocr_options
    # Nemotron OCR requires a CUDA accelerator.
    pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }
    )

    return converter


def test_e2e_nemotron_ocr_conversions():
    pdf_paths = get_pdf_paths()

    # Each engine config is verified against its own (namespaced) groundtruth so it
    # does not clash with the shared `test_e2e_ocr_conversion` groundtruth.
    engines: List[Tuple[OcrOptions, str, bool]] = [
        (NemotronOcrOptions(), "nemotron-ocr", True),
        (
            NemotronOcrOptions(force_full_page_ocr=True),
            "nemotron-ocr.full-page",
            True,
        ),
    ]

    for ocr_options, engine_suffix, supports_rotation in engines:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, "
            f"merge_level: {ocr_options.merge_level}, "
            f"force_full_page_ocr: {ocr_options.force_full_page_ocr}"
        )
        converter = get_converter(ocr_options=ocr_options)
        for pdf_path in pdf_paths:
            if not supports_rotation and "rotated" in pdf_path.name:
                continue
            print(f"converting {pdf_path}")

            doc_result: ConversionResult = converter.convert(pdf_path)

            verify_conversion_result_v2(
                input_path=pdf_path,
                doc_result=doc_result,
                generate=GENERATE_V2,
                ocr_engine=engine_suffix,
                fuzzy=True,
            )
