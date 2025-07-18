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
    test_files = [
        "tests/data/pdf/2203.01017v2.pdf",
        "tests/data/pdf/2206.01062.pdf",
        "tests/data/pdf/2305.03393v1.pdf",
    ]
    # test_files = [str(f) for f in Path("test/data/pdf").rglob("*.pdf")]

    do_ts = False
    do_ocr = False

    run_threaded = True
    run_serial = True

    if run_threaded:
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
                        do_table_structure=do_ts,
                        do_ocr=do_ocr,
                    ),
                )
            }
        )

        threaded_converter.initialize_pipeline(InputFormat.PDF)

        # Test threaded pipeline
        threaded_success_count = 0
        threaded_failure_count = 0
        start_time = time.perf_counter()
        for result in threaded_converter.convert_all(test_files, raises_on_error=True):
            print(
                "Finished converting document with threaded pipeline:",
                result.input.file.name,
            )
            if result.status == ConversionStatus.SUCCESS:
                threaded_success_count += 1
            else:
                threaded_failure_count += 1
        threaded_time = time.perf_counter() - start_time

        del threaded_converter

        print(f"Threaded pipeline:  {threaded_time:.2f} seconds")

    if run_serial:
        # Standard pipeline
        standard_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=PdfPipelineOptions(
                        do_table_structure=do_ts,
                        do_ocr=do_ocr,
                    ),
                )
            }
        )

        standard_converter.initialize_pipeline(InputFormat.PDF)

        # Test standard pipeline
        standard_success_count = 0
        standard_failure_count = 0
        start_time = time.perf_counter()
        for result in standard_converter.convert_all(test_files, raises_on_error=True):
            print(
                "Finished converting document with standard pipeline:",
                result.input.file.name,
            )
            if result.status == ConversionStatus.SUCCESS:
                standard_success_count += 1
            else:
                standard_failure_count += 1
        standard_time = time.perf_counter() - start_time

        del standard_converter

        print(f"Standard pipeline:  {standard_time:.2f} seconds")

    # Verify results
    if run_threaded and run_serial:
        assert standard_success_count == threaded_success_count
        assert standard_failure_count == threaded_failure_count
    if run_serial:
        assert standard_success_count == len(test_files)
        assert standard_failure_count == 0
    if run_threaded:
        assert threaded_success_count == len(test_files)
        assert threaded_failure_count == 0


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
