"""Test for page_range bug where conversion stops prematurely at page 32."""

from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def test_page_range_beyond_32():
    """
    Test that page_range works correctly when requesting pages beyond index 32.
    
    This test verifies the fix for the bug where page_range would stop at page 32
    when the range started from page 30 or higher.
    
    Bug scenario:
    - page_range=(30, 35) would only extract pages 30-32 instead of 30-35
    - Pages with page_no >= 32 (0-indexed) were not being processed
    
    Root cause was a hardcoded batch_size=32 in the drain loop.
    """
    # Use a multi-page PDF for testing
    # Note: 2206.01062.pdf is a research paper that should have enough pages
    test_pdf = Path("tests/data/pdf/2206.01062.pdf")
    
    # Skip test if PDF doesn't exist
    if not test_pdf.exists():
        pytest.skip(f"Test PDF not found: {test_pdf}")
    
    # Create converter with StandardPdfPipeline
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline),
        },
    )
    
    # First, convert without page_range to get total page count
    result_full = converter.convert(test_pdf)
    total_pages = len(result_full.pages)
    
    # Skip if PDF doesn't have enough pages for this test
    if total_pages < 40:
        pytest.skip(f"PDF has only {total_pages} pages, need at least 40 for this test")
    
    # Test case 1: page_range=(1, 45) should work
    result1 = converter.convert(test_pdf, page_range=(1, min(45, total_pages)))
    expected_pages1 = min(45, total_pages)
    assert len(result1.pages) == expected_pages1, (
        f"Expected {expected_pages1} pages for range (1, {min(45, total_pages)}), "
        f"got {len(result1.pages)}"
    )
    
    # Test case 2: page_range=(30, 35) should extract pages 30-35 (not just 30-32)
    # This was the failing case in the bug report
    result2 = converter.convert(test_pdf, page_range=(30, min(35, total_pages)))
    expected_pages2 = min(6, total_pages - 29)  # pages 30-35 is 6 pages
    assert len(result2.pages) == expected_pages2, (
        f"Expected {expected_pages2} pages for range (30, {min(35, total_pages)}), "
        f"got {len(result2.pages)}. This is the bug: conversion stopped prematurely!"
    )
    
    # Verify that the page numbers are correct
    page_numbers2 = [p.page_no for p in result2.pages]
    expected_page_nos = list(range(29, 29 + expected_pages2))  # 0-indexed: 29, 30, 31, 32, 33, 34
    assert page_numbers2 == expected_page_nos, (
        f"Expected page numbers {expected_page_nos}, got {page_numbers2}"
    )
    
    # Test case 3: page_range=(30, 45) should extract pages 30-45 (not just 30-32)
    result3 = converter.convert(test_pdf, page_range=(30, min(45, total_pages)))
    expected_pages3 = min(16, total_pages - 29)  # pages 30-45 is 16 pages
    assert len(result3.pages) == expected_pages3, (
        f"Expected {expected_pages3} pages for range (30, {min(45, total_pages)}), "
        f"got {len(result3.pages)}. This is the bug: conversion stopped prematurely!"
    )
    
    # Test case 4: page_range with pages entirely beyond 32
    if total_pages >= 40:
        result4 = converter.convert(test_pdf, page_range=(35, min(40, total_pages)))
        expected_pages4 = min(6, total_pages - 34)  # pages 35-40 is 6 pages
        assert len(result4.pages) == expected_pages4, (
            f"Expected {expected_pages4} pages for range (35, {min(40, total_pages)}), "
            f"got {len(result4.pages)}"
        )
        
        # Verify that pages with page_no >= 32 are actually processed
        page_numbers4 = [p.page_no for p in result4.pages]
        assert all(page_no >= 34 for page_no in page_numbers4), (
            f"Expected all page_no >= 34, got {page_numbers4}"
        )


def test_page_range_edge_cases():
    """Test edge cases for page_range parameter."""
    test_pdf = Path("tests/data/pdf/2206.01062.pdf")
    
    if not test_pdf.exists():
        pytest.skip(f"Test PDF not found: {test_pdf}")
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline),
        },
    )
    
    # Get total page count
    result_full = converter.convert(test_pdf)
    total_pages = len(result_full.pages)
    
    if total_pages < 5:
        pytest.skip(f"PDF has only {total_pages} pages, need at least 5 for this test")
    
    # Edge case 1: Single page at boundary (page 32)
    if total_pages >= 32:
        result = converter.convert(test_pdf, page_range=(32, 32))
        assert len(result.pages) == 1, f"Expected 1 page, got {len(result.pages)}"
        assert result.pages[0].page_no == 31, f"Expected page_no=31, got {result.pages[0].page_no}"
    
    # Edge case 2: Single page after boundary (page 33)
    if total_pages >= 33:
        result = converter.convert(test_pdf, page_range=(33, 33))
        assert len(result.pages) == 1, f"Expected 1 page, got {len(result.pages)}"
        assert result.pages[0].page_no == 32, f"Expected page_no=32, got {result.pages[0].page_no}"
    
    # Edge case 3: Range crossing the boundary (31-34)
    if total_pages >= 34:
        result = converter.convert(test_pdf, page_range=(31, 34))
        assert len(result.pages) == 4, f"Expected 4 pages, got {len(result.pages)}"
        expected_nos = [30, 31, 32, 33]
        actual_nos = [p.page_no for p in result.pages]
        assert actual_nos == expected_nos, f"Expected {expected_nos}, got {actual_nos}"
