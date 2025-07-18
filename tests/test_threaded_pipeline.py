import logging
import time
from pathlib import Path
from typing import List

import pytest

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline


def test_threaded_pipeline_multiple_documents():
    """Test threaded pipeline with multiple documents and compare with standard pipeline"""
    test_files = [str(f) for f in Path("tests/data/pdf").rglob("*.pdf")] or [
        "tests/data/pdf/2203.01017v2.pdf",
        "tests/data/pdf/2206.01062.pdf",
        "tests/data/pdf/2305.03393v1.pdf",
    ]

    # Threaded pipeline
    threaded_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=ThreadedPdfPipelineOptions(
                    layout_batch_size=1,
                    table_batch_size=1,
                    ocr_batch_size=1,
                    batch_timeout_seconds=1.0,
                    do_table_structure=True,
                    do_ocr=True,
                ),
            )
        }
    )

    threaded_converter.initialize_pipeline(InputFormat.PDF)

    # Test threaded pipeline
    threaded_results = []
    start_time = time.perf_counter()
    for result in threaded_converter.convert_all(test_files, raises_on_error=True):
        print(
            "Finished converting document with threaded pipeline:",
            result.input.file.name,
        )
        threaded_results.append(result)
    threaded_time = time.perf_counter() - start_time

    del threaded_converter

    print("\nMulti-document Pipeline Comparison:")
    print(f"Threaded pipeline:  {threaded_time:.2f} seconds")

    # Standard pipeline
    standard_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=PdfPipelineOptions(
                    do_table_structure=True,
                    do_ocr=True,
                ),
            )
        }
    )

    standard_converter.initialize_pipeline(InputFormat.PDF)

    # Test standard pipeline
    standard_results = []
    start_time = time.perf_counter()
    for result in standard_converter.convert_all(test_files, raises_on_error=True):
        print(
            "Finished converting document with standard pipeline:",
            result.input.file.name,
        )
        standard_results.append(result)
    standard_time = time.perf_counter() - start_time

    del standard_converter

    print(f"Standard pipeline:  {standard_time:.2f} seconds")
    print(f"Speedup:            {standard_time / threaded_time:.2f}x")

    # Verify results
    assert len(standard_results) == len(threaded_results)
    for result in standard_results:
        assert result.status == ConversionStatus.SUCCESS
    for result in threaded_results:
        assert result.status == ConversionStatus.SUCCESS

    # Basic content comparison
    for i, (standard_result, threaded_result) in enumerate(
        zip(standard_results, threaded_results)
    ):
        standard_doc = standard_result.document
        threaded_doc = threaded_result.document

        assert len(standard_doc.pages) == len(threaded_doc.pages), (
            f"Document {i} page count mismatch"
        )
        assert len(standard_doc.texts) == len(threaded_doc.texts), (
            f"Document {i} text count mismatch"
        )


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
                ),
            )
        }
    )

    start_time = time.perf_counter()
    threaded_results = list(threaded_converter.convert_all([test_file]))
    threaded_time = time.perf_counter() - start_time

    print("\nPipeline Comparison:")
    print(f"Sync pipeline:     {sync_time:.2f} seconds")
    print(f"Threaded pipeline: {threaded_time:.2f} seconds")
    print(f"Speedup:           {sync_time / threaded_time:.2f}x")

    # Verify results are equivalent
    assert len(sync_results) == len(threaded_results) == 1
    assert (
        sync_results[0].status == threaded_results[0].status == ConversionStatus.SUCCESS
    )

    # Basic content comparison
    sync_doc = sync_results[0].document
    threaded_doc = threaded_results[0].document

    assert len(sync_doc.pages) == len(threaded_doc.pages)
    assert len(sync_doc.texts) == len(threaded_doc.texts)


if __name__ == "__main__":
    # Run basic performance test
    test_pipeline_comparison()
