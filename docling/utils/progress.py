# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, TextIO

from tqdm import tqdm

from docling.datamodel.progress import (
    ConversionProgressEvent,
    DocumentCompletedProgress,
    DocumentStartedProgress,
    EnrichmentProgress,
    PageCompletedProgress,
)
from docling.datamodel.settings import settings

# The orange of the documentation theme (docs/stylesheets/extra.css).
DOCLING_ORANGE = "#ff4902"


@dataclass
class _Bar:
    """The pages or the enrichment step currently shown for one document."""

    key: str
    label: str
    bar: Optional[tqdm]
    count: str = ""


class ProgressPrinter:
    """Ready-made progress callback that prints to stderr.

    One line per document, then a progress bar for its pages and one for each
    enrichment step. On a terminal the bars fill up in place, in docling
    orange; elsewhere (a log file, a pipe) each bar is written once, as a plain
    `done/total` count. When documents convert concurrently
    (`settings.perf.doc_batch_concurrency > 1`), each document keeps its own
    bar and the labels carry the document index, e.g. `[2] pages 9/9`.

    Args:
        total_documents: Number of documents in the batch, to print `[3/12]`
            instead of `[3]`.
        stream: Where to print. Defaults to `sys.stderr`.
        color: Color the bars. Pass `False`, or set the `NO_COLOR` environment
            variable, for plain bars.
    """

    def __init__(
        self,
        total_documents: Optional[int] = None,
        stream: Optional[TextIO] = None,
        color: bool = True,
    ) -> None:
        self.total_documents = total_documents
        self._stream = stream
        self.color = color and not os.environ.get("NO_COLOR")
        self._bars: dict[int, _Bar] = {}
        self._lock = threading.Lock()

    @property
    def stream(self) -> Optional[TextIO]:
        return self._stream if self._stream is not None else sys.stderr

    @staticmethod
    def is_terminal(stream: Optional[TextIO]) -> bool:
        return stream is not None and stream.isatty()

    def __call__(self, event: ConversionProgressEvent) -> None:
        with self._lock:
            if isinstance(event, DocumentStartedProgress):
                position = str(event.document_index)
                if self.total_documents is not None:
                    position += f"/{self.total_documents}"
                self._write(f"[{position}] Converting {event.document_name}")
            elif isinstance(event, PageCompletedProgress):
                self._advance(
                    event.document_index,
                    "pages",
                    "page",
                    event.completed_pages,
                    event.total_pages,
                )
            elif isinstance(event, EnrichmentProgress):
                self._advance(
                    event.document_index,
                    event.label,
                    "item",
                    event.completed_items,
                    event.total_items,
                )
            elif isinstance(event, DocumentCompletedProgress):
                self._close(event.document_index)
                self._write(f"Finished {event.document_name}: {event.status.value}")

    def _write(self, text: str) -> None:
        stream = self.stream
        if stream is not None:
            # Clears the bars on the stream, writes the line, redraws the bars.
            tqdm.write(text, file=stream)
            stream.flush()

    @staticmethod
    def _concurrent() -> bool:
        return (
            settings.perf.doc_batch_concurrency > 1 and settings.perf.doc_batch_size > 1
        )

    def _advance(
        self, document_index: int, key: str, unit: str, done: int, total: int
    ) -> None:
        shown = self._bars.get(document_index)
        if shown is None or shown.key != key:
            self._close(document_index)
            concurrent = self._concurrent()
            label = f"[{document_index}] {key}" if concurrent else key
            bar = None
            stream = self.stream
            if self.is_terminal(stream):
                # Pages arrive in batches: start from the first batch, so the
                # rate and time left come from the batches that follow.
                # Concurrent documents share the screen, so their finished bars
                # give way to a one-line count instead of staying.
                bar = tqdm(
                    total=total,
                    initial=done,
                    desc=f"  {label}",
                    unit=unit,
                    file=stream,
                    disable=False,
                    dynamic_ncols=True,
                    leave=not concurrent,
                    colour=DOCLING_ORANGE if self.color else None,
                )
            shown = self._bars[document_index] = _Bar(key, label, bar)
        shown.count = f"{done}/{total}"
        if shown.bar is not None:
            shown.bar.update(done - shown.bar.n)

    def _close(self, document_index: int) -> None:
        shown = self._bars.pop(document_index, None)
        if shown is None:
            return
        if shown.bar is not None:
            shown.bar.close()
            if shown.bar.leave:
                return
        self._write(f"  {shown.label} {shown.count}")
