import time
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_TEST_FILES = [
    "tests/data/pdf/regression/2203.01017v2.pdf",
    "tests/data/pdf/regression/2206.01062.pdf",
    "tests/data/pdf/regression/2305.03393v1.pdf",
]
_SINGLE_FILE = "tests/data/pdf/regression/2206.01062.pdf"

pytestmark = pytest.mark.ml_pdf_model


def _make_threaded_converter(**kwargs) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=ThreadedDoclingParseDocumentBackend,
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_table_structure=False,
                    do_ocr=False,
                    **kwargs,
                ),
            )
        }
    )


def _make_standard_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=DoclingParseDocumentBackend,
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_table_structure=False,
                    do_ocr=False,
                ),
            )
        }
    )


def test_threaded_pipeline_multiple_documents():
    converter = _make_threaded_converter()
    converter.initialize_pipeline(InputFormat.PDF)

    results = list(converter.convert_all(_TEST_FILES, raises_on_error=True))

    assert len(results) == len(_TEST_FILES)
    assert all(r.status == ConversionStatus.SUCCESS for r in results)


def test_threaded_and_standard_backends_convert_with_standard_pipeline():
    threaded_converter = _make_threaded_converter()
    standard_converter = _make_standard_converter()

    threaded_result = threaded_converter.convert(_SINGLE_FILE)
    standard_result = standard_converter.convert(_SINGLE_FILE)

    assert threaded_result.status == ConversionStatus.SUCCESS
    assert standard_result.status == ConversionStatus.SUCCESS


def test_threaded_pipeline_with_pypdfium_backend():
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_table_structure=False,
                    do_ocr=False,
                ),
            )
        }
    )
    converter.initialize_pipeline(InputFormat.PDF)

    for i in range(3):
        result = converter.convert(_SINGLE_FILE)
        assert result.status == ConversionStatus.SUCCESS, f"iteration {i} failed"


def test_threaded_pipeline_page_range():
    converter = _make_threaded_converter()

    result = converter.convert(
        _SINGLE_FILE,
        raises_on_error=True,
        page_range=(2, 4),
    )

    assert result.status == ConversionStatus.SUCCESS
    assert [p.page_no for p in result.pages] == [2, 3, 4]
