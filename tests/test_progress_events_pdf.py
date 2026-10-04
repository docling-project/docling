# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.progress import (
    ConversionProgressEvent,
    DocumentCompletedProgress,
    DocumentStartedProgress,
    PageCompletedProgress,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model

PDF_9_PAGES = Path("tests/data/pdf/sources/2206.01062.pdf")


class _Recorder:
    def __init__(self) -> None:
        self.events: list[ConversionProgressEvent] = []
        self.threads: set[str] = set()

    def __call__(self, event: ConversionProgressEvent) -> None:
        self.events.append(event)
        self.threads.add(threading.current_thread().name)

    def pages(self) -> list[PageCompletedProgress]:
        return [ev for ev in self.events if isinstance(ev, PageCompletedProgress)]


def test_pdf_pages_are_reported_once_on_the_calling_thread():
    recorder = _Recorder()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF], progress_callback=recorder
    )

    converter.convert(PDF_9_PAGES, page_range=(3, 6))

    pages = recorder.pages()
    assert sorted(ev.page_no for ev in pages) == [3, 4, 5, 6]
    assert [ev.completed_pages for ev in pages] == [1, 2, 3, 4]
    assert {ev.total_pages for ev in pages} == {4}
    assert all(ev.success for ev in pages)
    # Pages are drained on the thread that called convert(), so a plain
    # callback needs no locking for a single document.
    assert recorder.threads == {threading.current_thread().name}
    assert isinstance(recorder.events[0], DocumentStartedProgress)
    assert isinstance(recorder.events[-1], DocumentCompletedProgress)


def test_pdf_pages_cut_by_the_timeout_are_reported_as_failed():
    recorder = _Recorder()
    options = PdfPipelineOptions(document_timeout=1e-6, do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
        progress_callback=recorder,
    )

    converter.convert(PDF_9_PAGES, page_range=(1, 3), raises_on_error=False)

    pages = recorder.pages()
    assert sorted(ev.page_no for ev in pages) == [1, 2, 3]
    assert pages[-1].completed_pages == pages[-1].total_pages == 3
    assert not all(ev.success for ev in pages)
