# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
import threading
from typing import Optional, TextIO

from docling.datamodel.progress import (
    ConversionProgressEvent,
    DocumentCompletedProgress,
    DocumentStartedProgress,
    EnrichmentProgress,
    PageCompletedProgress,
)


class ProgressPrinter:
    """Ready-made progress callback that prints to stderr.

    One line per document, with the page count and each enrichment step on a
    line of its own. On a terminal those lines update in place; elsewhere (a
    log file, a pipe) only their final value is printed.
    """

    def __init__(
        self, total_documents: Optional[int] = None, stream: Optional[TextIO] = None
    ) -> None:
        self.total_documents = total_documents
        self._stream = stream
        self._line_key: Optional[str] = None
        self._line_text = ""
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
                self._end_line()
                position = str(event.document_index)
                if self.total_documents is not None:
                    position += f"/{self.total_documents}"
                self._print(f"[{position}] Converting {event.document_name}")
            elif isinstance(event, PageCompletedProgress):
                self._update(
                    "pages", f"pages {event.completed_pages}/{event.total_pages}"
                )
            elif isinstance(event, EnrichmentProgress):
                self._update(
                    event.step,
                    f"{event.step} {event.completed_items}/{event.total_items}",
                )
            elif isinstance(event, DocumentCompletedProgress):
                self._end_line()
                self._print(f"Finished {event.document_name}: {event.status.value}")

    def _print(self, text: str, end: str = "\n") -> None:
        stream = self.stream
        if stream is not None:
            print(text, end=end, file=stream, flush=True)

    def _update(self, key: str, text: str) -> None:
        if self._line_key not in (None, key):
            self._end_line()
        self._line_key = key
        self._line_text = text
        if self.is_terminal(self.stream):
            self._print(f"\r  {text}", end="")

    def _end_line(self) -> None:
        if self._line_key is None:
            return
        if self.is_terminal(self.stream):
            self._print("")
        else:
            self._print(f"  {self._line_text}")
        self._line_key = None
