#!/usr/bin/env python3
"""Test to reproduce and validate the fix for document_timeout AssertionError issue."""

import tempfile
from pathlib import Path
import pytest

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def test_document_timeout_no_assertion_error():
    """
    Test that setting document_timeout doesn't cause an AssertionError in ReadingOrderModel.
    
    This test validates the fix for the issue where setting pipeline_options.document_timeout
    would lead to an AssertionError in ReadingOrderModel._readingorder_elements_to_docling_doc
    when page.size is None for uninitialized pages after timeout.
    """
    # Test PDF path - using an existing test file
    test_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    
    if not test_doc_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Configure pipeline with a very short timeout to trigger the timeout condition
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 0.001  # Very short timeout to trigger timeout
    pipeline_options.do_ocr = False  # Disable OCR to make processing faster but still trigger timeout
    pipeline_options.do_table_structure = False  # Disable table structure for faster processing
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # This should not raise an AssertionError even with timeout
    # Before the fix, this would fail with: AssertionError at line 140 in readingorder_model.py
    try:
        doc_result = converter.convert(test_doc_path, raises_on_error=False)
        
        # The conversion should complete without throwing an AssertionError
        # It may result in PARTIAL_SUCCESS due to timeout, but should not crash
        assert doc_result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS], \
            f"Expected SUCCESS or PARTIAL_SUCCESS, got {doc_result.status}"
        
        # Verify that we have a document with pages
        assert doc_result.document is not None, "Document should not be None"
        
        print(f"Test passed: Conversion completed with status {doc_result.status}")
        print(f"Document has {doc_result.document.num_pages()} pages")
        
    except AssertionError as e:
        if "size is not None" in str(e):
            pytest.fail(f"The original AssertionError still occurs: {e}")
        else:
            # Re-raise other assertion errors
            raise


def test_document_timeout_with_longer_timeout():
    """
    Test that document_timeout works correctly with a reasonable timeout value.
    """
    test_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    
    if not test_doc_path.exists():
        pytest.skip("Test PDF file not found")
    
    # Configure pipeline with a reasonable timeout
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 10.0  # 10 seconds should be enough for a small document
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # This should complete successfully
    doc_result = converter.convert(test_doc_path)
    
    assert doc_result.status == ConversionStatus.SUCCESS, \
        f"Expected SUCCESS, got {doc_result.status}"
    assert doc_result.document is not None, "Document should not be None"
    assert doc_result.document.num_pages() > 0, "Document should have pages"
    
    print(f"Test passed: Conversion completed successfully with {doc_result.document.num_pages()} pages")


if __name__ == "__main__":
    test_document_timeout_no_assertion_error()
    test_document_timeout_with_longer_timeout()
    print("All tests passed!")