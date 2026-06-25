"""Tests for the page-level progress callback on DocumentConverter.

The converter accepts an optional ``progress_callback`` that receives
``PageStartedProgress`` / ``PageCompletedProgress`` events for paginated
pipelines and a single ``DocumentCompletedProgress`` event for every
conversion (paginated or not).

Related issue: https://github.com/docling-project/docling/issues/3493
"""

import threading
from pathlib import Path

import pytest

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.progress import (
    ConversionProgressKind,
    DocumentCompletedProgress,
    PageCompletedProgress,
    PageStartedProgress,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


@pytest.fixture
def normal_4pages_path():
    return Path("./tests/data/pdf/sources/normal_4pages.pdf")


@pytest.fixture
def markdown_path():
    return Path("./tests/data/md/sources/duck.md")


def _pdf_converter(callback, pipeline_cls=StandardPdfPipeline):
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_cls,
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=False,
                ),
            )
        },
        progress_callback=callback,
    )


@pytest.mark.parametrize(
    "pipeline_cls", [StandardPdfPipeline, LegacyStandardPdfPipeline]
)
def test_paginated_pdf_emits_page_and_document_events(normal_4pages_path, pipeline_cls):
    """A 4-page PDF emits one started + one completed event per page and a
    single document_completed event at the end, for both the threaded and the
    legacy paginated pipelines."""
    events = []
    lock = threading.Lock()

    def callback(event):
        # Page events may arrive from the producer/drain threads.
        with lock:
            events.append(event)

    converter = _pdf_converter(callback, pipeline_cls=pipeline_cls)
    result = converter.convert(normal_4pages_path, raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS

    expected_pages = result.input.page_count
    assert expected_pages == 4

    started = [e for e in events if isinstance(e, PageStartedProgress)]
    completed = [e for e in events if isinstance(e, PageCompletedProgress)]
    document = [e for e in events if isinstance(e, DocumentCompletedProgress)]

    assert {e.page_no for e in started} == set(range(1, expected_pages + 1))
    assert {e.page_no for e in completed} == set(range(1, expected_pages + 1))
    assert all(e.total_pages == expected_pages for e in started + completed)
    assert all(e.success for e in completed)

    assert len(document) == 1
    assert document[0].num_pages == expected_pages
    assert document[0].status == ConversionStatus.SUCCESS
    # The document_completed event is always the final one emitted.
    assert isinstance(events[-1], DocumentCompletedProgress)


def test_non_paginated_emits_only_document_completed(markdown_path):
    """A declarative (non-paginated) input emits no page events, only a single
    document_completed event."""
    events = []

    converter = DocumentConverter(progress_callback=events.append)
    result = converter.convert(markdown_path, raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS

    assert not [e for e in events if isinstance(e, PageStartedProgress)]
    assert not [e for e in events if isinstance(e, PageCompletedProgress)]

    document = [e for e in events if isinstance(e, DocumentCompletedProgress)]
    assert len(document) == 1
    assert document[0].kind == ConversionProgressKind.DOCUMENT_COMPLETED
    assert document[0].status == ConversionStatus.SUCCESS


def test_no_callback_is_a_noop(normal_4pages_path):
    """Conversion works unchanged when no callback is provided."""
    converter = _pdf_converter(None)
    result = converter.convert(normal_4pages_path, raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS


def test_callback_exception_does_not_break_conversion(markdown_path):
    """A callback that raises must not interrupt the conversion."""

    def boom(event):
        raise RuntimeError("callback failure")

    converter = DocumentConverter(progress_callback=boom)
    result = converter.convert(markdown_path, raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS


def test_callback_exception_on_page_events_does_not_break_pdf(normal_4pages_path):
    """A callback raising on page events (from the threaded PDF pipeline) must
    not interrupt the conversion."""

    def boom(event):
        raise RuntimeError("callback failure")

    converter = _pdf_converter(boom)
    result = converter.convert(normal_4pages_path, raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS
    assert len(result.document.pages) == 4


def test_page_range_total_pages_reflects_requested_subset(normal_4pages_path):
    """``total_pages`` and emitted page numbers follow the requested page_range,
    not the physical page count."""
    events = []
    lock = threading.Lock()

    def callback(event):
        with lock:
            events.append(event)

    converter = _pdf_converter(callback)
    result = converter.convert(
        normal_4pages_path, raises_on_error=False, page_range=(2, 3)
    )
    assert result.status == ConversionStatus.SUCCESS

    started = [e for e in events if isinstance(e, PageStartedProgress)]
    completed = [e for e in events if isinstance(e, PageCompletedProgress)]
    document = [e for e in events if isinstance(e, DocumentCompletedProgress)]

    assert {e.page_no for e in started} == {2, 3}
    assert {e.page_no for e in completed} == {2, 3}
    assert all(e.total_pages == 2 for e in started + completed)
    assert document[0].num_pages == 2
