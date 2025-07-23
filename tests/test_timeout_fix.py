#!/usr/bin/env python3
"""Test to validate timeout handling and ReadingOrderModel compatibility."""

import time
from unittest.mock import Mock

from docling.datamodel.base_models import Page, ConversionStatus
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.backend.pdf_backend import PdfDocumentBackend


class MockPdfBackend(PdfDocumentBackend):
    """Mock PDF backend for testing timeout scenarios."""
    
    def __init__(self, in_doc, path_or_stream):
        # Skip parent __init__ since we're mocking
        self.input_format = getattr(in_doc, 'format', 'PDF')
        self.path_or_stream = path_or_stream
        self._page_count = 10
    
    def load_page(self, page_no):
        mock_page = Mock()
        mock_page.is_valid.return_value = True
        return mock_page
    
    def page_count(self):
        return self._page_count
    
    def is_valid(self):
        return True
    
    def unload(self):
        pass


class TimeoutTestPipeline(PaginatedPipeline):
    """Test implementation of PaginatedPipeline for timeout testing."""
    
    def __init__(self, pipeline_options: PdfPipelineOptions):
        super().__init__(pipeline_options)
        self.build_pipe = [self.slow_processing_model]
    
    def slow_processing_model(self, conv_res, page_batch):
        """Mock model that takes time to process to trigger timeout."""
        for page in page_batch:
            # Simulate slow processing
            time.sleep(0.1)
            # Only initialize size for first few pages before timeout hits
            if page.page_no < 2:  # Only first 2 pages get initialized
                page.size = Mock()
                page.size.height = 800
                page.size.width = 600
            # Other pages remain with size=None to simulate timeout scenario
            yield page
    
    def initialize_page(self, conv_res, page):
        """Initialize page with mock backend."""
        page._backend = Mock()
        page._backend.is_valid.return_value = True
        return page
    
    def _determine_status(self, conv_res):
        """Determine conversion status - inherit from parent to test the actual fix."""
        return super()._determine_status(conv_res)
    
    @classmethod
    def get_default_options(cls):
        return PdfPipelineOptions()
    
    @classmethod
    def is_backend_supported(cls, backend):
        return True


def test_document_timeout_filters_uninitialized_pages():
    """
    Test that setting document_timeout doesn't cause an AssertionError in ReadingOrderModel.
    
    This test validates the fix for the issue where setting pipeline_options.document_timeout
    would lead to an AssertionError in ReadingOrderModel._readingorder_elements_to_docling_doc
    when page.size is None for uninitialized pages after timeout.
    
    The fix filters out pages with size=None after timeout, preventing the assertion error.
    """
    
    # Create a test document with multiple pages
    input_doc = Mock(spec=InputDocument)
    input_doc.page_count = 10
    input_doc.limits = Mock()
    input_doc.limits.page_range = (1, 10)
    input_doc.file = Mock()
    input_doc.file.name = "test.pdf"
    input_doc._backend = MockPdfBackend(input_doc, "test.pdf")
    
    # Configure pipeline with very short timeout to trigger timeout condition
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 0.05  # 50ms timeout - intentionally short
    
    pipeline = TimeoutTestPipeline(pipeline_options)
    
    # Process document using full pipeline execution - this should trigger timeout but not cause AssertionError
    result = pipeline.execute(input_doc, raises_on_error=False)
    
    # Verify that uninitialized pages (with size=None) were filtered out
    # This is the key fix - pages with size=None are removed before ReadingOrderModel processes them
    assert len(result.pages) < input_doc.page_count, \
        "Should have fewer pages after timeout filtering"
    
    # All remaining pages should have size initialized
    # This ensures ReadingOrderModel won't encounter pages with size=None
    for page in result.pages:
        assert page.size is not None, \
            f"Page {page.page_no} should have size initialized, got None"
    
    # Status should indicate partial success due to timeout
    # This validates the fix for status preservation
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, \
        f"Expected PARTIAL_SUCCESS, got {result.status}"


def test_readingorder_model_compatibility():
    """
    Test that the filtered pages are compatible with ReadingOrderModel expectations.
    
    This test ensures that after the timeout filtering fix, all remaining pages
    have the required 'size' attribute that ReadingOrderModel expects.
    """
    
    # Create a test document
    input_doc = Mock(spec=InputDocument)
    input_doc.page_count = 5
    input_doc.limits = Mock()
    input_doc.limits.page_range = (1, 5)
    input_doc.file = Mock()
    input_doc.file.name = "test.pdf"
    input_doc._backend = MockPdfBackend(input_doc, "test.pdf")
    
    # Configure pipeline with timeout that allows some pages to be processed
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 0.15  # 150ms timeout
    
    pipeline = TimeoutTestPipeline(pipeline_options)
    
    # Process document using full pipeline execution
    result = pipeline.execute(input_doc, raises_on_error=False)
    
    # Simulate what ReadingOrderModel expects - all pages should have size
    for page in result.pages:
        # This would be the assertion that failed before the fix:
        # assert size is not None (line 132 in readingorder_model.py)
        assert page.size is not None, \
            "ReadingOrderModel requires all pages to have size != None"
        assert hasattr(page.size, 'height'), \
            "Size should have height attribute"
        assert hasattr(page.size, 'width'), \
            "Size should have width attribute"


def test_timeout_status_preservation():
    """
    Test that ConversionStatus.PARTIAL_SUCCESS set during timeout is preserved 
    by _determine_status method.
    
    This validates the fix for the issue where _determine_status would overwrite
    the PARTIAL_SUCCESS status that was correctly set due to document_timeout.
    """
    
    # Create a test document
    input_doc = Mock(spec=InputDocument)
    input_doc.page_count = 10
    input_doc.limits = Mock()
    input_doc.limits.page_range = (1, 10)
    input_doc.file = Mock()
    input_doc.file.name = "test.pdf"
    input_doc._backend = MockPdfBackend(input_doc, "test.pdf")
    
    conv_res = ConversionResult(input=input_doc)
    
    # Configure pipeline with very short timeout
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 0.05  # 50ms timeout
    
    pipeline = TimeoutTestPipeline(pipeline_options)
    
    # Process document - this should trigger timeout and set PARTIAL_SUCCESS
    result = pipeline._build_document(conv_res)
    
    # Verify that timeout was triggered and status was set to PARTIAL_SUCCESS
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, \
        f"Expected PARTIAL_SUCCESS after timeout, got {result.status}"
    
    # Now test that _determine_status preserves the PARTIAL_SUCCESS status
    final_status = pipeline._determine_status(result)
    assert final_status == ConversionStatus.PARTIAL_SUCCESS, \
        f"_determine_status should preserve PARTIAL_SUCCESS from timeout, got {final_status}"
    
    # Also test the full pipeline execution to ensure end-to-end behavior
    final_result = pipeline.execute(input_doc, raises_on_error=False)
    assert final_result.status == ConversionStatus.PARTIAL_SUCCESS, \
        f"Final result should maintain PARTIAL_SUCCESS status, got {final_result.status}"


def test_no_timeout_scenario():
    """Test that normal processing without timeout works correctly."""
    
    # Create a test document
    input_doc = Mock(spec=InputDocument)
    input_doc.page_count = 3  # Small number to avoid timeout
    input_doc.limits = Mock()
    input_doc.limits.page_range = (1, 3)
    input_doc.file = Mock()
    input_doc.file.name = "test.pdf"
    input_doc._backend = MockPdfBackend(input_doc, "test.pdf")
    
    # Configure pipeline with sufficient timeout
    pipeline_options = PdfPipelineOptions()
    pipeline_options.document_timeout = 2.0  # 2 seconds - should be enough
    
    pipeline = TimeoutTestPipeline(pipeline_options)
    
    # Process document using full pipeline execution
    result = pipeline.execute(input_doc, raises_on_error=False)
    
    # All pages should be processed successfully without timeout
    assert len(result.pages) >= 2, \
        "Should process at least 2 pages without timeout"
    
    # All pages should have size initialized
    for page in result.pages:
        assert page.size is not None, \
            f"Page {page.page_no} should have size initialized"
    
    # Status should be SUCCESS since no timeout occurred
    assert result.status == ConversionStatus.SUCCESS, \
        f"Expected SUCCESS without timeout, got {result.status}"