import sys
from pathlib import Path
from typing import List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    OcrOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption
from tests.verify_utils import verify_conversion_result_v2

from .test_data_gen_flag import GEN_TEST_DATA

GENERATE = GEN_TEST_DATA


def get_webp_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/webp/")

    # List all WEBP files in the directory and its subdirectories
    webp_files = sorted(directory.rglob("*.webp"))
    return webp_files


def get_converter(ocr_options: OcrOptions):
    image_format_option = ImageFormatOption()
    image_format_option.pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: image_format_option},
        allowed_formats=[InputFormat.IMAGE],
    )

    return converter


def test_e2e_webp_conversions():
    webp_paths = get_webp_paths()

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

    # only works on mac
    if "darwin" == sys.platform:
        engines.append(OcrMacOptions())
        engines.append(OcrMacOptions(force_full_page_ocr=True))
    for ocr_options in engines:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, language: {ocr_options.lang}"
        )
        converter = get_converter(ocr_options=ocr_options)
        for webp_path in webp_paths:
            print(f"converting {webp_path}")

            doc_result: ConversionResult = converter.convert(webp_path)

            verify_conversion_result_v2(
                input_path=webp_path,
                doc_result=doc_result,
                generate=GENERATE,
                fuzzy=True,
            )
