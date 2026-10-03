# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
import time
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.backend_options import (
    PdfBackendOptions,
    ThreadedDoclingParseBackendOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import (
    StandardPdfPipeline,
    ThreadedItem,
    ThreadedPipelineStage,
    ThreadedQueue,
)

_TEST_FILES = [
    "tests/data/pdf/sources/2203.01017v2.pdf",
    "tests/data/pdf/sources/2206.01062.pdf",
    "tests/data/pdf/sources/2305.03393v1.pdf",
]
_SINGLE_FILE = "tests/data/pdf/sources/2206.01062.pdf"

pytestmark = pytest.mark.ml_pdf_model


def _make_threaded_converter(**kwargs) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=ThreadedDoclingParseDocumentBackend,
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_table_structure=False,
                    do_ocr=False,
                    **kwargs,
                ),
            )
        }
    )


def test_threaded_pipeline_multiple_documents():
    converter = _make_threaded_converter()
    converter.initialize_pipeline(InputFormat.PDF)

    results = list(converter.convert_all(_TEST_FILES, raises_on_error=True))

    assert len(results) == len(_TEST_FILES)
    assert all(r.status == ConversionStatus.SUCCESS for r in results)


def test_threaded_pipeline_with_deterministic_batching():
    """The option must convert the same document as the default does, and must not stall:
    waiting for a full batch relies on the queue being closed to release the last one."""
    default = _make_threaded_converter()
    default.initialize_pipeline(InputFormat.PDF)
    expected = default.convert(_SINGLE_FILE, raises_on_error=True)

    converter = _make_threaded_converter(deterministic_batching=True)
    converter.initialize_pipeline(InputFormat.PDF)
    result = converter.convert(_SINGLE_FILE, raises_on_error=True)

    assert result.status == ConversionStatus.SUCCESS
    assert [p.page_no for p in result.pages] == [p.page_no for p in expected.pages]


def test_threaded_pipeline_with_pypdfium_backend():
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_table_structure=False,
                    do_ocr=False,
                ),
            )
        }
    )
    converter.initialize_pipeline(InputFormat.PDF)

    for i in range(3):
        result = converter.convert(_SINGLE_FILE)
        assert result.status == ConversionStatus.SUCCESS, f"iteration {i} failed"


def test_threaded_docling_parse_table_matches_pypdfium(tmp_path: Path):
    """Both PDF backends must preserve the table from issue #3512."""
    pdf_path = tmp_path / "table_repro.pdf"
    rows = [
        ["Area of expertise", "Product Management", "Product Marketing"],
        ["Document Cloud", "Vamsi Vutukuru", "Nora Yau"],
        ["Acrobat", "Alex Chen", "Maria Lopez"],
        ["Sign", "Sam Patel", "Lena Frei"],
    ]

    figure = Figure(figsize=(8.27, 11.69))
    axis = figure.subplots()
    axis.axis("off")
    axis.text(
        0.5,
        0.95,
        "Contacts available for customer meetings",
        ha="center",
        fontsize=14,
    )
    table = axis.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    figure.savefig(pdf_path)

    documents = []
    backend_configs = [
        (PyPdfiumDocumentBackend, PdfBackendOptions()),
        (
            ThreadedDoclingParseDocumentBackend,
            ThreadedDoclingParseBackendOptions(parser_threads=1),
        ),
    ]
    for backend, backend_options in backend_configs:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=backend,
                    backend_options=backend_options,
                    pipeline_options=ThreadedPdfPipelineOptions(do_ocr=False),
                )
            }
        )
        result = converter.convert(pdf_path)

        assert result.status == ConversionStatus.SUCCESS
        documents.append(result.document)

    expected_cells = [cell for row in rows for cell in row]
    for document in documents:
        assert len(document.tables) == 1
        assert len(document.texts) == 1
        assert [
            cell.text.strip() for cell in document.tables[0].data.table_cells
        ] == expected_cells

    pypdfium_document, threaded_document = documents
    assert pypdfium_document.tables[0].data.num_rows == (
        threaded_document.tables[0].data.num_rows
    )
    assert pypdfium_document.tables[0].data.num_cols == (
        threaded_document.tables[0].data.num_cols
    )


def test_threaded_pipeline_page_range():
    converter = _make_threaded_converter()

    result = converter.convert(
        _SINGLE_FILE,
        raises_on_error=True,
        page_range=(2, 4),
    )

    assert result.status == ConversionStatus.SUCCESS
    assert [p.page_no for p in result.pages] == [2, 3, 4]


def test_threaded_pipeline_stage_shutdown_timeout():
    """A stage stuck in a blocking model call is abandoned after
    `shutdown_timeout`, not the hardcoded 15s the pipeline used to have."""
    entered = threading.Event()
    release = threading.Event()

    class SlowModel:
        def __call__(self, conv_res, pages):
            entered.set()
            release.wait()
            return pages

    stage = ThreadedPipelineStage(
        name="slow",
        model=SlowModel(),
        batch_size=1,
        batch_timeout=0.05,
        queue_max_size=10,
        shutdown_timeout=1.0,
    )
    stage.start()
    try:
        stage.input_queue.put(
            ThreadedItem(payload=Page(page_no=1), run_id=1, page_no=1, conv_res=None)
        )
        assert entered.wait(timeout=5.0), "stage never entered the blocking call"

        start = time.monotonic()
        stage.stop()
        elapsed = time.monotonic() - start

        assert 1.0 <= elapsed < 5.0
        assert stage._thread is not None and stage._thread.is_alive()
    finally:
        release.set()
        if stage._thread is not None:
            stage._thread.join(timeout=5.0)


def _item(page_no: int) -> ThreadedItem:
    return ThreadedItem(
        payload=Page(page_no=page_no), run_id=1, page_no=page_no, conv_res=None
    )


def test_get_batch_without_require_full_takes_what_is_queued():
    """The default: whatever arrived by the time the poll interval expires."""
    queue = ThreadedQueue(10)
    queue.put(_item(1))

    batch = queue.get_batch(4, timeout=0.05)

    assert [item.page_no for item in batch] == [1]


def test_get_batch_with_require_full_waits_for_the_whole_batch():
    """A slow producer must not decide the batch: without this, the size of a batch - and
    with it the model output - depends on how busy the machine is."""
    queue = ThreadedQueue(10)
    queue.put(_item(1))

    def feed_slowly() -> None:
        for page_no in (2, 3, 4):
            time.sleep(0.05)
            queue.put(_item(page_no))

    producer = threading.Thread(target=feed_slowly)
    producer.start()
    try:
        batch = queue.get_batch(4, timeout=0.01, require_full=True)
    finally:
        producer.join(timeout=5.0)

    assert [item.page_no for item in batch] == [1, 2, 3, 4]


def test_get_batch_with_require_full_returns_the_remainder_once_closed():
    """The last batch of a run is short, and closing the queue is what releases it."""
    queue = ThreadedQueue(10)
    queue.put(_item(1))
    queue.put(_item(2))
    released = threading.Event()

    def close_soon() -> None:
        time.sleep(0.05)
        released.set()
        queue.close()

    closer = threading.Thread(target=close_soon)
    closer.start()
    try:
        batch = queue.get_batch(4, require_full=True)
    finally:
        closer.join(timeout=5.0)

    assert released.is_set(), "the batch came back before the queue was closed"
    assert [item.page_no for item in batch] == [1, 2]


def test_get_batch_with_require_full_does_not_deadlock_on_a_short_queue():
    """A queue that cannot hold a full batch would otherwise stop both sides: the producer
    blocks at `max_size` while the consumer waits for more."""
    queue = ThreadedQueue(2)
    queue.put(_item(1))
    queue.put(_item(2))

    batch = queue.get_batch(4, require_full=True)

    assert [item.page_no for item in batch] == [1, 2]


def test_deterministic_batching_reaches_the_stage():
    """The option is off by default and is what the stage passes to its queue."""
    assert ThreadedPdfPipelineOptions().deterministic_batching is False

    stage = ThreadedPipelineStage(
        name="noop",
        model=lambda conv_res, pages: pages,
        batch_size=2,
        batch_timeout=0.05,
        queue_max_size=10,
        require_full_batches=True,
    )

    assert stage.require_full_batches is True
