import sys
from pathlib import Path
from typing import List

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    EasyOcrOptions,
    OcrMacOptions,
    OcrOptions,
    OnnxtrOcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v1, verify_conversion_result_v2

GENERATE_V1 = GEN_TEST_DATA
GENERATE_V2 = GEN_TEST_DATA


def get_pdf_paths():
    # Define the directory you want to search
    directory = Path("./tests/data_scanned")

    # List all PDF files in the directory and its subdirectories
    pdf_files = sorted(directory.rglob("*.pdf"))
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
                backend=DoclingParseDocumentBackend,
            )
        }
    )

    return converter


def test_e2e_conversions():
    pdf_paths = get_pdf_paths()

    engines: List[OcrOptions] = [
        EasyOcrOptions(),
        TesseractOcrOptions(),
        TesseractCliOcrOptions(),
        EasyOcrOptions(force_full_page_ocr=True),
        TesseractOcrOptions(force_full_page_ocr=True),
        TesseractOcrOptions(force_full_page_ocr=True, lang=["auto"]),
        TesseractCliOcrOptions(force_full_page_ocr=True),
        TesseractCliOcrOptions(force_full_page_ocr=True, lang=["auto"]),
    ]

    # rapidocr is only available for Python >=3.6,<3.13
    if sys.version_info < (3, 13):
        engines.append(RapidOcrOptions())
        engines.append(RapidOcrOptions(force_full_page_ocr=True))

    # onnxtr is only available for Python >=3.10 and <3.13
    if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
        engines.append(OnnxtrOcrOptions())
        engines.append(OnnxtrOcrOptions(force_full_page_ocr=True))

    # only works on mac
    if "darwin" == sys.platform:
        engines.append(OcrMacOptions())
        engines.append(OcrMacOptions(force_full_page_ocr=True))

    for ocr_options in engines:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, language: {ocr_options.lang}"
        )
        converter = get_converter(ocr_options=ocr_options)
        for pdf_path in pdf_paths:
            print(f"converting {pdf_path}")

            doc_result: ConversionResult = converter.convert(pdf_path)

            verify_conversion_result_v1(
                input_path=pdf_path,
                doc_result=doc_result,
                generate=GENERATE_V1,
                fuzzy=True,
            )

            verify_conversion_result_v2(
                input_path=pdf_path,
                doc_result=doc_result,
                generate=GENERATE_V2,
                fuzzy=True,
            )
