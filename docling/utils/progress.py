# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import os
import sys
import threading
from typing import Optional, TextIO

from tqdm import tqdm

from docling.datamodel.progress import (
    ConversionProgressEvent,
    DocumentCompletedProgress,
    DocumentStartedProgress,
    EnrichmentProgress,
    PageCompletedProgress,
)

# The orange of the documentation theme (docs/stylesheets/extra.css).
DOCLING_ORANGE = "#ff4902"


class ProgressPrinter:
    """Ready-made progress callback that prints to stderr.

    One line per document, then a progress bar for its pages and one for each
    enrichment step. On a terminal the bars fill up in place, in docling
    orange; elsewhere (a log file, a pipe) each bar is written once, as a plain
    `done/total` count.

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
        self._bar_key: Optional[str] = None
        self._bar: Optional[tqdm] = None
        self._count = ""
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
                self._close_bar()
                position = str(event.document_index)
                if self.total_documents is not None:
                    position += f"/{self.total_documents}"
                self._print(f"[{position}] Converting {event.document_name}")
            elif isinstance(event, PageCompletedProgress):
                self._advance("pages", "page", event.completed_pages, event.total_pages)
            elif isinstance(event, EnrichmentProgress):
                self._advance(
                    event.label, "item", event.completed_items, event.total_items
                )
            elif isinstance(event, DocumentCompletedProgress):
                self._close_bar()
                self._print(f"Finished {event.document_name}: {event.status.value}")

    def _print(self, text: str) -> None:
        stream = self.stream
        if stream is not None:
            print(text, file=stream, flush=True)

    def _advance(self, key: str, unit: str, done: int, total: int) -> None:
        if self._bar_key != key:
            self._close_bar()
            self._bar_key = key
            stream = self.stream
            if self.is_terminal(stream):
                # Pages arrive in batches: start from the first batch, so the
                # rate and time left come from the batches that follow.
                self._bar = tqdm(
                    total=total,
                    initial=done,
                    desc=f"  {key}",
                    unit=unit,
                    file=stream,
                    disable=False,
                    dynamic_ncols=True,
                    colour=DOCLING_ORANGE if self.color else None,
                )
        self._count = f"{done}/{total}"
        if self._bar is not None:
            self._bar.update(done - self._bar.n)

    def _close_bar(self) -> None:
        if self._bar_key is None:
            return
        if self._bar is not None:
            self._bar.close()
        else:
            self._print(f"  {self._bar_key} {self._count}")
        self._bar_key = None
        self._bar = None
