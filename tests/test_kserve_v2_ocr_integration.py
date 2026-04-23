"""Opt-in integration test for KServe v2 OCR."""

import os
import socket
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import KserveV2OcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

KSERVE_OCR_HTTP_URL_ENV = "DOCLING_KSERVE_OCR_HTTP_URL"
KSERVE_OCR_GRPC_URL_ENV = "DOCLING_KSERVE_OCR_GRPC_URL"
KSERVE_OCR_URL_ENVS = {
    "http": KSERVE_OCR_HTTP_URL_ENV,
    "grpc": KSERVE_OCR_GRPC_URL_ENV,
}

# TODO: Add support for binary mode for the http transfer
# KSERVE_OCR_TRANSPORTS = ["http", "grpc"]
KSERVE_OCR_TRANSPORTS = ["grpc"]
KSERVE_OCR_LANGUAGES = [
    "en",
    # "ch",
    # "arabic",
    # "korean",
    # "latin",
]


@pytest.mark.skipif(
    not all(os.getenv(env_name) for env_name in KSERVE_OCR_URL_ENVS.values()),
    reason=(
        f"Set {KSERVE_OCR_HTTP_URL_ENV} and {KSERVE_OCR_GRPC_URL_ENV} to run "
        "the KServe v2 OCR integration test."
    ),
)
def test_kserve_v2_ocr_conversion() -> None:
    input_path = Path("tests/data_scanned/ocr_test.pdf")

    for transport in KSERVE_OCR_TRANSPORTS:
        url = os.environ[KSERVE_OCR_URL_ENVS[transport]]
        ###################################################
        # TODO: Clarify what's wrong with the DNS resolution of doclingforce2.zurich.ibm.com
        if transport == "grpc":
            host, _, port = url.rpartition(":")
            resolved_host = socket.getaddrinfo(
                host, int(port), type=socket.SOCK_STREAM
            )[0][4][0]
            url = f"{resolved_host}:{port}"
        ###################################################
        for lang in KSERVE_OCR_LANGUAGES:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
            pipeline_options.do_table_structure = False
            pipeline_options.ocr_options = KserveV2OcrOptions(
                url=url,
                transport=transport,
                model_name="rapidocr",
                model_version="1",
                lang=[lang],
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=DoclingParseDocumentBackend,
                    )
                }
            )

            doc_result: ConversionResult = converter.convert(input_path)

            verify_conversion_result_v2(
                input_path=input_path,
                doc_result=doc_result,
                generate=GEN_TEST_DATA,
                ocr_engine="kserve_v2_ocr",
                fuzzy=True,
            )


r""" Run against a local endpoint with:
DOCLING_KSERVE_OCR_HTTP_URL=localhost:8000 \
DOCLING_KSERVE_OCR_GRPC_URL=localhost:8001 \
uv run pytest tests/test_kserve_v2_ocr_integration.py
"""
if __name__ == "__main__":
    test_kserve_v2_ocr_conversion()
