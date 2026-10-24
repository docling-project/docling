# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
import time
from typing import cast

from docling.datamodel.document import ConversionResult
from docling.pipeline.standard_pdf_pipeline import ThreadedItem, ThreadedQueue


def _item(page_no: int) -> ThreadedItem:
    return ThreadedItem(
        payload=None,
        run_id=1,
        page_no=page_no,
        conv_res=cast(ConversionResult, None),
    )


def test_get_batch_waits_for_later_pages_until_the_timeout() -> None:
    """A stage must keep collecting pages for the configured interval.

    ``batch_polling_interval_seconds`` says each stage waits up to that long
    to accumulate items. Returning at the first page makes the layout batch
    depend on arrival timing.
    """
    queue = ThreadedQueue(max_size=8)
    queue.put(_item(1))

    def _arrive_later() -> None:
        time.sleep(0.15)
        queue.put(_item(2))
        queue.put(_item(3))

    producer = threading.Thread(target=_arrive_later)
    producer.start()
    started = time.monotonic()
    batch = queue.get_batch(4, timeout=0.6)
    elapsed = time.monotonic() - started
    producer.join(timeout=1.0)

    assert [item.page_no for item in batch] == [1, 2, 3]
    assert elapsed >= 0.15


def test_get_batch_returns_a_closed_partial_batch_without_waiting() -> None:
    queue = ThreadedQueue(max_size=8)
    queue.put(_item(1))
    queue.close()

    started = time.monotonic()
    batch = queue.get_batch(4, timeout=1.0)

    assert [item.page_no for item in batch] == [1]
    assert time.monotonic() - started < 0.2


def test_get_batch_without_timeout_does_not_wait_for_a_full_batch() -> None:
    queue = ThreadedQueue(max_size=8)
    queue.put(_item(1))
    queue.put(_item(2))

    started = time.monotonic()
    batch = queue.get_batch(4, timeout=None)

    assert [item.page_no for item in batch] == [1, 2]
    assert time.monotonic() - started < 0.2
