# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import threading
from io import BytesIO

import pytest

from docling.datamodel.base_models import DocumentStream, InputFormat, Page
from docling.datamodel.document import ConversionResult, _DocumentConversionInput
from docling.document_converter import HTMLFormatOption
from docling.pipeline.standard_pdf_pipeline import (
    ThreadedItem,
    ThreadedPipelineStage,
    ThreadedQueue,
)


def test_url_input_with_none_limits_uses_default_limits(monkeypatch):
    calls = []

    def resolve(source, headers, *, max_file_size):
        calls.append(max_file_size)
        return DocumentStream(name="page.html", stream=BytesIO(b"<p>Hello</p>"))

    monkeypatch.setattr("docling.datamodel.document.resolve_source_to_stream", resolve)
    inputs = _DocumentConversionInput(
        path_or_stream_iterator=["https://example.com/page.html"], limits=None
    )
    documents = list(inputs.docs({InputFormat.HTML: HTMLFormatOption()}))
    assert len(documents) == 1 and documents[0].valid
    assert documents[0]._backend.convert().export_to_markdown() == "Hello"
    assert len(calls) == 1


@pytest.mark.parametrize("timeout", [None, 5.0])
def test_queue_close_wakes_blocked_producer(monkeypatch, timeout):
    queue = ThreadedQueue(max_size=0)
    waiting = threading.Event()
    original_wait = queue._not_full.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(queue._not_full, "wait", wait)
    accepted = []
    # A full zero-capacity queue never inspects the item payload.
    producer = threading.Thread(
        target=lambda: accepted.append(queue.put(None, timeout))
    )
    producer.start()
    reached_wait = waiting.wait(2.0)
    queue.close()
    producer.join(2.0)
    assert reached_wait
    assert not producer.is_alive()
    assert accepted == [False]


def test_log_level_change_during_stage_does_not_fail_page():
    inputs = _DocumentConversionInput(
        path_or_stream_iterator=[
            DocumentStream(name="page.html", stream=BytesIO(b"<p>Hello</p>"))
        ]
    )
    document = next(iter(inputs.docs({InputFormat.HTML: HTMLFormatOption()})))
    result = ConversionResult(input=document)
    item = ThreadedItem(payload=Page(page_no=1), run_id=1, page_no=1, conv_res=result)
    logger = logging.getLogger("docling.pipeline.standard_pdf_pipeline")
    original_level = logger.level

    def model(result, pages):
        logger.setLevel(logging.DEBUG)
        return pages

    stage = ThreadedPipelineStage(
        name="test",
        model=model,
        batch_size=1,
        batch_timeout=0.1,
        queue_max_size=1,
    )
    try:
        logger.setLevel(logging.INFO)
        assert stage._process_batch([item]) == [item]
        assert not item.is_failed
        assert item.error is None
    finally:
        logger.setLevel(original_level)
