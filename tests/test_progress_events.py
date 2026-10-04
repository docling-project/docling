# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
from collections.abc import Iterable
from io import BytesIO
from typing import Optional

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ConvertPipelineOptions
from docling.datamodel.progress import (
    ConversionPhase,
    ConversionProgressEvent,
    DocumentCompletedProgress,
    DocumentStartedProgress,
    EnrichmentProgress,
    PageCompletedProgress,
    PhaseStartedProgress,
    ProgressCallback,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, FormatOption
from docling.models.base_model import GenericEnrichmentModel
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.simple_pipeline import SimplePipeline


class _Recorder:
    def __init__(self) -> None:
        self.events: list[ConversionProgressEvent] = []
        self.threads: set[str] = set()
        self.threads_by_doc: dict[int, set[str]] = {}
        self._lock = threading.Lock()

    def __call__(self, event: ConversionProgressEvent) -> None:
        name = threading.current_thread().name
        with self._lock:
            self.events.append(event)
            self.threads.add(name)
            self.threads_by_doc.setdefault(event.document_index, set()).add(name)

    def of(self, event_type: type) -> list:
        return [ev for ev in self.events if isinstance(ev, event_type)]


def _md(name: str, text: str = "# Title\n\nSome text.") -> DocumentStream:
    return DocumentStream(name=name, stream=BytesIO(text.encode()))


class _TagModel(GenericEnrichmentModel[NodeItem]):
    """Marks the text items it processes; skips the ones it cannot prepare."""

    elements_batch_size = 2

    def __init__(self, skip_text: Optional[str] = None) -> None:
        self.skip_text = skip_text

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return isinstance(element, TextItem) and element.label == DocItemLabel.TEXT

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[NodeItem]:
        if not self.is_processable(conv_res.document, element):
            return None
        assert isinstance(element, TextItem)
        return None if element.text == self.skip_text else element

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[NodeItem]:
        for element in element_batch:
            assert isinstance(element, TextItem)
            element.label = DocItemLabel.PARAGRAPH
            yield element


class _ParagraphModel(_TagModel):
    """Only sees the items the previous step relabelled, like chart extraction
    only sees the pictures the classifier marked as charts."""

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return isinstance(element, TextItem) and element.label == DocItemLabel.PARAGRAPH

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[NodeItem]:
        yield from element_batch


class _EnrichedMarkdownPipeline(SimplePipeline):
    def __init__(self, pipeline_options: ConvertPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.enrichment_pipe = [_TagModel(skip_text="skip me"), _ParagraphModel()]


class _FailingMarkdownPipeline(SimplePipeline):
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        raise ValueError("boom")


def _converter_with(
    pipeline_cls: type[BasePipeline], recorder: ProgressCallback
) -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: FormatOption(
                pipeline_cls=pipeline_cls, backend=MarkdownDocumentBackend
            )
        },
        progress_callback=recorder,
    )


def test_every_input_is_bracketed_by_document_events_in_order():
    recorder = _Recorder()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD], progress_callback=recorder
    )
    sources = [_md("a.md"), _md("same.md"), _md("not_allowed.html"), _md("same.md")]

    results = list(converter.convert_all(sources, raises_on_error=False))

    assert [r.status for r in results] == [
        ConversionStatus.SUCCESS,
        ConversionStatus.SUCCESS,
        ConversionStatus.SKIPPED,
        ConversionStatus.SUCCESS,
    ]
    started = recorder.of(DocumentStartedProgress)
    completed = recorder.of(DocumentCompletedProgress)
    assert [ev.document_index for ev in started] == [1, 2, 3, 4]
    assert [(ev.document_index, ev.status) for ev in completed] == [
        (1, ConversionStatus.SUCCESS),
        (2, ConversionStatus.SUCCESS),
        (3, ConversionStatus.SKIPPED),
        (4, ConversionStatus.SUCCESS),
    ]
    # Non-paged formats still report the pipeline phases, but no pages.
    first_doc = [ev for ev in recorder.events if ev.document_index == 1]
    assert isinstance(first_doc[0], DocumentStartedProgress)
    assert all(isinstance(ev, PhaseStartedProgress) for ev in first_doc[1:-1])
    assert isinstance(first_doc[-1], DocumentCompletedProgress)
    phases = [ev.phase for ev in recorder.of(PhaseStartedProgress)]
    assert phases[: len(ConversionPhase)] == list(ConversionPhase)
    assert not recorder.of(PageCompletedProgress)


def test_enrichment_items_are_counted_per_step_and_reach_the_total():
    recorder = _Recorder()
    converter = _converter_with(_EnrichedMarkdownPipeline, recorder)
    text = "\n\n".join(["one", "two", "skip me", "four", "five"])

    converter.convert(_md("doc.md", text))

    by_step: dict[str, list[tuple[int, int]]] = {}
    for ev in recorder.of(EnrichmentProgress):
        by_step.setdefault(ev.step, []).append((ev.completed_items, ev.total_items))
    # Batches of 2 over the 4 preparable items, then the skipped one closes the step.
    assert by_step["_TagModel"] == [(0, 5), (2, 5), (4, 5), (5, 5)]
    # Counted after the first step ran, so only the 4 items it relabelled.
    assert by_step["_ParagraphModel"] == [(0, 4), (2, 4), (4, 4)]


def test_failing_document_still_gets_a_terminal_event():
    recorder = _Recorder()
    converter = _converter_with(_FailingMarkdownPipeline, recorder)

    with pytest.raises(RuntimeError):
        converter.convert(_md("doc.md"), raises_on_error=True)

    assert isinstance(recorder.events[-1], DocumentCompletedProgress)
    assert recorder.events[-1].status == ConversionStatus.FAILURE


def test_raising_callback_does_not_break_the_conversion():
    def callback(event: ConversionProgressEvent) -> None:
        raise ValueError("progress bar was closed")

    converter = DocumentConverter(progress_callback=callback)

    result = converter.convert(_md("doc.md"))

    assert result.status == ConversionStatus.SUCCESS


def test_concurrent_documents_each_report_in_order_from_one_thread(monkeypatch):
    monkeypatch.setattr(settings.perf, "doc_batch_size", 4)
    monkeypatch.setattr(settings.perf, "doc_batch_concurrency", 4)
    recorder = _Recorder()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD], progress_callback=recorder
    )

    list(converter.convert_all([_md(f"doc{i}.md") for i in range(1, 9)]))

    for index in range(1, 9):
        events = [ev for ev in recorder.events if ev.document_index == index]
        assert isinstance(events[0], DocumentStartedProgress)
        assert isinstance(events[-1], DocumentCompletedProgress)
        assert len(recorder.threads_by_doc[index]) == 1
    assert threading.current_thread().name not in recorder.threads
