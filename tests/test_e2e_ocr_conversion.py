import sys
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    OcrMode,
    OcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .groundtruth_paths import get_ocr_groundtruth_paths
from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

GENERATE_V2 = GEN_TEST_DATA
pytestmark = pytest.mark.ml_ocr


def get_pdf_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/ocr/sources")

    # List all PDF files in the directory and its subdirectories
    pdf_files = sorted(directory.rglob("ocr_test*.pdf"))

    return pdf_files


def get_converter(ocr_options: OcrOptions):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,  # PdfFormatOption().backend,
            )
        }
    )

    return converter


def test_e2e_conversions():
    pdf_paths = get_pdf_paths()

    configs: list[OcrOptions] = [
        # Default OCR mode
        TesseractOcrOptions(),
        TesseractCliOcrOptions(),
        EasyOcrOptions(),
        TesseractOcrOptions(psm=3),
        # Full page OCR
        TesseractOcrOptions(mode=OcrMode.FULL_PAGE_OCR),
        TesseractOcrOptions(mode=OcrMode.FULL_PAGE_OCR, lang=["auto"]),
        TesseractCliOcrOptions(mode=OcrMode.FULL_PAGE_OCR),
        TesseractCliOcrOptions(mode=OcrMode.FULL_PAGE_OCR, lang=["auto"]),
        EasyOcrOptions(mode=OcrMode.FULL_PAGE_OCR),
    ]

    for rapidocr_backend in ["onnxruntime", "torch"]:
        if sys.version_info >= (3, 14) and rapidocr_backend == "onnxruntime":
            # skip onnxruntime backend on Python 3.14
            continue

        configs.append(RapidOcrOptions(backend=rapidocr_backend))
        configs.append(
            RapidOcrOptions(backend=rapidocr_backend, mode=OcrMode.FULL_PAGE_OCR)
        )
        configs.append(
            RapidOcrOptions(
                backend=rapidocr_backend,
                mode=OcrMode.FULL_PAGE_OCR,
                rec_font_path="test",
                rapidocr_params={"Rec.font_path": None},  # overwrites rec_font_path
            )
        )

    # only works on mac
    if "darwin" == sys.platform:
        configs.append(OcrMacOptions())
        configs.append(OcrMacOptions(mode=OcrMode.FULL_PAGE_OCR))

    for ocr_options in configs:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, language: {ocr_options.lang}"
        )
        converter = get_converter(ocr_options=ocr_options)
        for pdf_path in pdf_paths:
            print(f"converting {pdf_path}")

            doc_result: ConversionResult = converter.convert(pdf_path)

            verify_conversion_result_v2(
                gt=get_ocr_groundtruth_paths(pdf_path, mode=ocr_options.mode),
                doc_result=doc_result,
                generate=GENERATE_V2,
                fuzzy=True,
            )
