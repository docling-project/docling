"""Tests for failed page handling in StandardPdfPipeline.

These tests verify that when some PDF pages fail to parse, they are still
added to DoclingDocument.pages to maintain correct page numbering and
ensure page break markers are generated correctly during export.

Related: https://github.com/docling-project/docling-core/pull/466
"""

from pathlib import Path

import pytest

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


@pytest.fixture
def skipped_1page_path():
    return Path("./tests/data/pdf/skipped_1page.pdf")


@pytest.fixture
def skipped_2pages_path():
    return Path("./tests/data/pdf/skipped_2pages.pdf")


@pytest.fixture
def normal_4pages_path():
    return Path("./tests/data/pdf/normal_4pages.pdf")


def test_normal_pages_all_present(normal_4pages_path):
    """Test that all pages are present in DoclingDocument.pages for a normal PDF."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        }
    )

    result = converter.convert(normal_4pages_path, raises_on_error=False)

    # Document should succeed completely
    assert result.status == ConversionStatus.SUCCESS, (
        f"Expected SUCCESS status, got: {result.status}"
    )

    # Get expected page count from input
    expected_page_count = result.input.page_count

    # DoclingDocument.pages should contain all pages
    assert result.document is not None, "Document should not be None"
    actual_page_count = len(result.document.pages)

    assert actual_page_count == expected_page_count, (
        f"DoclingDocument.pages should contain all {expected_page_count} pages, "
        f"but got {actual_page_count}"
    )

    # Verify all page numbers are present
    expected_page_nos = set(range(1, expected_page_count + 1))
    actual_page_nos = set(result.document.pages.keys())

    assert actual_page_nos == expected_page_nos, (
        f"Missing page numbers in DoclingDocument.pages. "
        f"Expected: {expected_page_nos}, Got: {actual_page_nos}"
    )

    # No errors should be recorded
    assert len(result.errors) == 0, (
        f"No errors should be recorded for normal PDF, but got: {result.errors}"
    )


def test_failed_pages_added_to_document_1page(skipped_1page_path):
    """Test that a single failed page is added to DoclingDocument.pages."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        }
    )

    result = converter.convert(skipped_1page_path, raises_on_error=False)

    # Document should have partial success due to failed page(s)
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, (
        f"Unexpected status: {result.status}"
    )

    # Get expected page count from input
    expected_page_count = result.input.page_count

    # DoclingDocument.pages should contain all pages (including failed ones)
    assert result.document is not None, "Document should not be None"
    actual_page_count = len(result.document.pages)

    assert actual_page_count == expected_page_count, (
        f"DoclingDocument.pages should contain all {expected_page_count} pages "
        f"(including failed ones), but got {actual_page_count}"
    )

    # Verify all page numbers are present
    expected_page_nos = set(range(1, expected_page_count + 1))
    actual_page_nos = set(result.document.pages.keys())

    assert actual_page_nos == expected_page_nos, (
        f"Missing page numbers in DoclingDocument.pages. "
        f"Expected: {expected_page_nos}, Got: {actual_page_nos}"
    )


def test_failed_pages_added_to_document_2pages(skipped_2pages_path):
    """Test that multiple failed pages are added to DoclingDocument.pages."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        }
    )

    result = converter.convert(skipped_2pages_path, raises_on_error=False)

    # Document should have partial success due to failed page(s)
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, (
        f"Unexpected status: {result.status}"
    )

    # Get expected page count from input
    expected_page_count = result.input.page_count

    # DoclingDocument.pages should contain all pages (including failed ones)
    assert result.document is not None, "Document should not be None"
    actual_page_count = len(result.document.pages)

    assert actual_page_count == expected_page_count, (
        f"DoclingDocument.pages should contain all {expected_page_count} pages "
        f"(including failed ones), but got {actual_page_count}"
    )

    # Verify all page numbers are present
    expected_page_nos = set(range(1, expected_page_count + 1))
    actual_page_nos = set(result.document.pages.keys())

    assert actual_page_nos == expected_page_nos, (
        f"Missing page numbers in DoclingDocument.pages. "
        f"Expected: {expected_page_nos}, Got: {actual_page_nos}"
    )


def test_failed_pages_have_size_info(skipped_1page_path):
    """Test that failed pages have size information when available."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        }
    )

    result = converter.convert(skipped_1page_path, raises_on_error=False)

    assert result.document is not None, "Document should not be None"

    # All pages should have size information
    for page_no, page_item in result.document.pages.items():
        assert page_item.size is not None, (
            f"Page {page_no} should have size information"
        )
        # Size should be valid (either from backend or default 0.0)
        assert page_item.size.width >= 0, f"Page {page_no} width should be >= 0"
        assert page_item.size.height >= 0, f"Page {page_no} height should be >= 0"


def test_errors_recorded_for_failed_pages(skipped_1page_path):
    """Test that errors are recorded in conv_res.errors for failed pages."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        }
    )

    result = converter.convert(skipped_1page_path, raises_on_error=False)

    # If status is PARTIAL_SUCCESS, there should be errors recorded
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        assert len(result.errors) > 0, (
            "PARTIAL_SUCCESS status should have errors recorded"
        )
