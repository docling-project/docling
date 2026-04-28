import time
from pathlib import Path

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_TEST_FILES = [
    "tests/data/pdf/2203.01017v2.pdf",
    "tests/data/pdf/2206.01062.pdf",
    "tests/data/pdf/2305.03393v1.pdf",
]
_SINGLE_FILE = "tests/data/pdf/2206.01062.pdf"


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


def _make_legacy_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=LegacyStandardPdfPipeline,
                backend=DoclingParseDocumentBackend,
                pipeline_options=PdfPipelineOptions(
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


def test_threaded_pipeline_matches_legacy():
    threaded_converter = _make_threaded_converter()
    legacy_converter = _make_legacy_converter()

    threaded_results = {
        Path(r.input.file).name: r
        for r in threaded_converter.convert_all([_SINGLE_FILE])
    }
    legacy_results = {
        Path(r.input.file).name: r for r in legacy_converter.convert_all([_SINGLE_FILE])
    }

    assert set(threaded_results) == set(legacy_results)
    for name in threaded_results:
        threaded_result = threaded_results[name]
        legacy_result = legacy_results[name]
        assert (
            threaded_result.status == legacy_result.status == ConversionStatus.SUCCESS
        )
        assert len(threaded_result.document.pages) == len(legacy_result.document.pages)
        # Text item counts differ slightly between the threaded and sequential
        # backends due to different internal segmentation paths; verify order
        # of magnitude agreement only.
        assert abs(
            len(threaded_result.document.texts) - len(legacy_result.document.texts)
        ) < 0.1 * len(legacy_result.document.texts)


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
