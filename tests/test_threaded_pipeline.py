import logging
import time
from collections import deque
from pathlib import Path
from typing import List

import pytest

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ThreadedPdfPipelineOptions,
)
from docling.datamodel.settings import DocumentLimits
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import (
    _LEAKED_PDFIUM_BACKENDS,
    RunContext,
    RunOutcome,
    StandardPdfPipeline,
    ThreadedItem,
)
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
                        batch_polling_interval_seconds=1.0,
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


def test_pypdfium_threaded_pipeline():
    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                ),
            },
        )
    )

    test_file = "tests/data/pdf/2206.01062.pdf"
    for i in range(6):
        print(f"iteration {i=}")
        conv_result = doc_converter.convert(test_file)
        assert conv_result.status == ConversionStatus.SUCCESS
        print(f"[{i=}] Success")
    print("All done!")


def test_execute_skips_backend_unload_on_unsafe_shutdown(monkeypatch):
    pipeline = StandardPdfPipeline(ThreadedPdfPipelineOptions())
    input_doc = InputDocument(
        path_or_stream=Path("tests/data/pdf/2206.01062.pdf"),
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )

    original_unload = input_doc._backend.unload
    unload_calls: list[str] = []

    def fake_build_document_run(conv_res: ConversionResult) -> RunOutcome:
        conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        return RunOutcome(conv_res=conv_res, allow_backend_unload=False)

    try:
        monkeypatch.setattr(
            input_doc._backend, "unload", lambda: unload_calls.append("backend")
        )
        monkeypatch.setattr(pipeline, "_build_document_run", fake_build_document_run)
        monkeypatch.setattr(
            pipeline,
            "_assemble_document_run",
            lambda conv_res, allow_backend_loads: conv_res,
        )
        monkeypatch.setattr(pipeline, "_enrich_document", lambda conv_res: conv_res)

        leaked_before = len(_LEAKED_PDFIUM_BACKENDS)
        result = pipeline.execute(input_doc, raises_on_error=True)

        assert result.status == ConversionStatus.PARTIAL_SUCCESS
        assert unload_calls == []
        assert _LEAKED_PDFIUM_BACKENDS[leaked_before] is input_doc._backend
    finally:
        while input_doc._backend in _LEAKED_PDFIUM_BACKENDS:
            _LEAKED_PDFIUM_BACKENDS.remove(input_doc._backend)
        original_unload()


def test_build_document_drains_emitted_pages_after_timeout(monkeypatch):
    class DummyPageBackend:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    class FakeInputQueue:
        def __init__(self) -> None:
            self.closed = False
            self.items: list[ThreadedItem] = []

        def put(self, item: ThreadedItem, timeout: float | None = None) -> bool:
            self.items.append(item)
            return not self.closed

        def close(self) -> None:
            self.closed = True

    class FakeStage:
        def __init__(self, input_queue: FakeInputQueue) -> None:
            self.input_queue = input_queue

        def start(self) -> None:
            return

        def stop(self) -> bool:
            return True

    class FakeOutputQueue:
        def __init__(self, batches: list[list[ThreadedItem]]) -> None:
            self._batches = deque(batches)
            self.closed = False

        def get_batch(
            self, size: int, timeout: float | None = None
        ) -> list[ThreadedItem]:
            if self._batches:
                batch = self._batches.popleft()
                if not self._batches:
                    self.closed = True
                return batch
            self.closed = True
            return []

        def close(self) -> None:
            self.closed = True

    class MonotonicClock:
        def __init__(self, values: list[float]) -> None:
            self._values = deque(values)
            self._last = values[-1]

        def __call__(self) -> float:
            if self._values:
                self._last = self._values.popleft()
            return self._last

    pipeline = StandardPdfPipeline(ThreadedPdfPipelineOptions(document_timeout=1))
    input_doc = InputDocument(
        path_or_stream=Path("tests/data/pdf/2206.01062.pdf"),
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        limits=DocumentLimits(page_range=(1, 2)),
    )
    conv_res = ConversionResult(input=input_doc)

    successful_page = Page(page_no=1)
    successful_page._backend = DummyPageBackend()

    fake_input_queue = FakeInputQueue()
    fake_stage = FakeStage(fake_input_queue)
    fake_output_queue = FakeOutputQueue(
        [
            [],
            [
                ThreadedItem(
                    payload=successful_page,
                    run_id=1,
                    page_no=1,
                    conv_res=conv_res,
                )
            ],
            [],
        ]
    )

    monkeypatch.setattr(
        pipeline,
        "_create_run_ctx",
        lambda: RunContext(
            stages=[fake_stage],
            first_stage=fake_stage,
            output_queue=fake_output_queue,
            timed_out_run_ids=set(),
        ),
    )
    monkeypatch.setattr(
        time,
        "monotonic",
        MonotonicClock([0.0, 0.0, 1.1, 1.1, 1.2]),
    )

    try:
        outcome = pipeline._build_document_run(conv_res)

        assert outcome.allow_backend_unload is True
        assert outcome.conv_res.status == ConversionStatus.PARTIAL_SUCCESS
        assert len(outcome.conv_res.pages) == 1
        assert outcome.conv_res.pages[0].page_no == 1
        assert successful_page._backend.unload_calls == 1
    finally:
        input_doc._backend.unload()


if __name__ == "__main__":
    # Run basic performance test
    test_pipeline_comparison()
