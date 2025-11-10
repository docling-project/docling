import sys
from pathlib import Path
from typing import List, Tuple

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    OcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

GENERATE_V2 = GEN_TEST_DATA


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

    engines: List[Tuple[OcrOptions, bool]] = [
        (TesseractOcrOptions(), True),
        (TesseractCliOcrOptions(), True),
        (EasyOcrOptions(), False),
        (TesseractOcrOptions(psm=3), True),
        (TesseractOcrOptions(force_full_page_ocr=True), True),
        (TesseractOcrOptions(force_full_page_ocr=True, lang=["auto"]), True),
        (TesseractCliOcrOptions(force_full_page_ocr=True), True),
        (TesseractCliOcrOptions(force_full_page_ocr=True, lang=["auto"]), True),
        (EasyOcrOptions(force_full_page_ocr=True), False),
    ]

    for rapidocr_backend in ["onnxruntime", "torch"]:
        if sys.version_info >= (3, 14) and rapidocr_backend == "onnxruntime":
            # skip onnxruntime backend on Python 3.14
            continue

        engines.append((RapidOcrOptions(backend=rapidocr_backend), False))
        engines.append(
            (RapidOcrOptions(backend=rapidocr_backend, force_full_page_ocr=True), False)
        )
        engines.append(
            (
                RapidOcrOptions(
                    backend=rapidocr_backend,
                    force_full_page_ocr=True,
                    rec_font_path="test",
                    rapidocr_params={"Rec.font_path": None},  # overwrites rec_font_path
                ),
                False,
            )
        )

    # only works on mac
    if "darwin" == sys.platform:
        engines.append((OcrMacOptions(), True))
        engines.append((OcrMacOptions(force_full_page_ocr=True), True))

    for ocr_options, supports_rotation in engines:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, language: {ocr_options.lang}"
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
                fuzzy=True,
            )
