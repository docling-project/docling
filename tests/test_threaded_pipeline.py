import time
from pathlib import Path
from typing import List

import pytest

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ThreadedPdfPipelineOptions
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

def test_threaded_pipeline_multiple_documents():
    """Test threaded pipeline with multiple documents"""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=ThreadedPdfPipelineOptions(
                    layout_batch_size=48,
                    ocr_batch_size=24,
                    batch_timeout_seconds=1.0,
                )
            )
        }
    )
    
    # Test threaded pipeline with multiple documents
    results = []
    start_time = time.perf_counter()
    for result in converter.convert_all([
        "tests/data/pdf/2206.01062.pdf", 
        "tests/data/pdf/2305.03393v1.pdf"
    ]):
        results.append(result)
    end_time = time.perf_counter()
    
    conversion_duration = end_time - start_time
    print(f"Threaded multi-doc conversion took {conversion_duration:.2f} seconds")
    
    assert len(results) == 2
    for result in results:
        assert result.status == ConversionStatus.SUCCESS


def test_pipeline_comparison():
    """Compare all three pipeline implementations"""
    test_file = "tests/data/pdf/2206.01062.pdf"
    
    # Sync pipeline
    sync_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
            )
        }
    )
    
    start_time = time.perf_counter()
    sync_results = list(sync_converter.convert_all([test_file]))
    sync_time = time.perf_counter() - start_time
    
    # Threaded pipeline
    threaded_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=ThreadedPdfPipelineOptions(
                    layout_batch_size=1,
                    ocr_batch_size=1,
                    table_batch_size=1,
                )
            )
        }
    )
    
    start_time = time.perf_counter()
    threaded_results = list(threaded_converter.convert_all([test_file]))
    threaded_time = time.perf_counter() - start_time
    
    print(f"\nPipeline Comparison:")
    print(f"Sync pipeline:     {sync_time:.2f} seconds")
    print(f"Threaded pipeline: {threaded_time:.2f} seconds")
    print(f"Speedup:           {sync_time/threaded_time:.2f}x")
    
    # Verify results are equivalent
    assert len(sync_results) == len(threaded_results) == 1
    assert sync_results[0].status == threaded_results[0].status == ConversionStatus.SUCCESS
    
    # Basic content comparison
    sync_doc = sync_results[0].document
    threaded_doc = threaded_results[0].document
    
    assert len(sync_doc.pages) == len(threaded_doc.pages)
    assert len(sync_doc.texts) == len(threaded_doc.texts)





if __name__ == "__main__":
    # Run basic performance test
    test_pipeline_comparison() 