"""Tests for the progress event callback system."""

from pathlib import Path

import pytest

from docling.datamodel.progress_event import (
    ConversionPhase,
    DocumentProgressEvent,
    PageProgressEvent,
    PhaseProgressEvent,
    ProgressEvent,
    ProgressEventType,
)

# ---------------------------------------------------------------------------
# Unit tests for event model definitions
# ---------------------------------------------------------------------------


def test_progress_event_is_frozen():
    event = ProgressEvent(
        event_type=ProgressEventType.DOCUMENT_START,
        document_name="test.pdf",
    )
    with pytest.raises(Exception):
        event.document_name = "other.pdf"  # type: ignore[misc]


def test_document_progress_event():
    event = DocumentProgressEvent(
        event_type=ProgressEventType.DOCUMENT_START,
        document_name="test.pdf",
        page_count=10,
    )
    assert event.event_type == ProgressEventType.DOCUMENT_START
    assert event.document_name == "test.pdf"
    assert event.page_count == 10


def test_document_progress_event_page_count_defaults_to_none():
    event = DocumentProgressEvent(
        event_type=ProgressEventType.DOCUMENT_START,
        document_name="test.html",
    )
    assert event.page_count is None


def test_phase_progress_event():
    event = PhaseProgressEvent(
        event_type=ProgressEventType.PHASE_START,
        document_name="test.pdf",
        phase=ConversionPhase.BUILD,
    )
    assert event.phase == ConversionPhase.BUILD


def test_page_progress_event():
    event = PageProgressEvent(
        event_type=ProgressEventType.PAGE_COMPLETE,
        document_name="test.pdf",
        page_no=3,
        total_pages=10,
    )
    assert event.page_no == 3
    assert event.total_pages == 10


# ---------------------------------------------------------------------------
# Integration tests with actual document conversion
# ---------------------------------------------------------------------------

_PDF_PATH = Path(__file__).parent / "data" / "pdf" / "2305.03393v1-pg9.pdf"
_MULTI_PAGE_PDF = Path(__file__).parent / "data" / "pdf" / "normal_4pages.pdf"


@pytest.mark.skipif(not _PDF_PATH.exists(), reason="Test PDF not available")
def test_event_sequence_for_pdf():
    """Convert a single-page PDF and verify the full event sequence."""
    from docling.document_converter import DocumentConverter

    events: list[ProgressEvent] = []

    def _collect(event: ProgressEvent) -> None:
        events.append(event)

    converter = DocumentConverter(progress_callback=_collect)
    converter.convert(source=_PDF_PATH)

    assert len(events) > 0, "No progress events were emitted"

    assert events[0].event_type == ProgressEventType.DOCUMENT_START
    assert isinstance(events[0], DocumentProgressEvent)

    assert events[-1].event_type == ProgressEventType.DOCUMENT_COMPLETE
    assert isinstance(events[-1], DocumentProgressEvent)

    phase_events = [e for e in events if isinstance(e, PhaseProgressEvent)]
    phase_starts = [
        e.phase for e in phase_events if e.event_type == ProgressEventType.PHASE_START
    ]
    phase_completes = [
        e.phase
        for e in phase_events
        if e.event_type == ProgressEventType.PHASE_COMPLETE
    ]
    assert phase_starts == [
        ConversionPhase.BUILD,
        ConversionPhase.ASSEMBLE,
        ConversionPhase.ENRICH,
    ]
    assert phase_completes == [
        ConversionPhase.BUILD,
        ConversionPhase.ASSEMBLE,
        ConversionPhase.ENRICH,
    ]

    page_events = [e for e in events if isinstance(e, PageProgressEvent)]
    assert len(page_events) >= 1, "Expected at least one PAGE_COMPLETE event"
    for pe in page_events:
        assert pe.event_type == ProgressEventType.PAGE_COMPLETE
        assert pe.page_no >= 1
        assert pe.total_pages >= 1


@pytest.mark.skipif(not _PDF_PATH.exists(), reason="Test PDF not available")
def test_no_callback_no_overhead():
    """Conversion works normally when no callback is provided."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(source=_PDF_PATH)
    assert result.document is not None


@pytest.mark.skipif(not _PDF_PATH.exists(), reason="Test PDF not available")
def test_callback_exception_does_not_break_conversion():
    """A failing callback should not prevent conversion from completing.

    All pipelines protect against callback exceptions via
    ``BasePipeline._emit_progress``.  The standard PDF pipeline additionally
    wraps the callback in a thread-safe wrapper for its worker threads.
    """
    from docling.document_converter import DocumentConverter

    call_count = 0

    def _bad_callback(event: ProgressEvent) -> None:
        nonlocal call_count
        call_count += 1
        # Only raise on PAGE_COMPLETE to test the thread-safe wrapper
        if event.event_type == ProgressEventType.PAGE_COMPLETE:
            raise RuntimeError("intentional test error")

    converter = DocumentConverter(progress_callback=_bad_callback)
    result = converter.convert(source=_PDF_PATH)
    # Conversion should still succeed
    assert result.document is not None
    assert call_count > 0


@pytest.mark.skipif(not _MULTI_PAGE_PDF.exists(), reason="Test PDF not available")
def test_multi_page_event_count():
    """Convert a multi-page PDF and verify we get one PAGE_COMPLETE per page."""
    from docling.document_converter import DocumentConverter

    events: list[ProgressEvent] = []
    converter = DocumentConverter(progress_callback=events.append)
    result = converter.convert(source=_MULTI_PAGE_PDF)

    page_events = [e for e in events if isinstance(e, PageProgressEvent)]
    assert len(page_events) == len(result.pages)
