from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument


class ManagedPdfiumDocumentBackend(PdfDocumentBackend, ABC):
    """Shared lifecycle management for PDFium-backed document backends."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: PdfBackendOptions = PdfBackendOptions(),
    ) -> None:
        super().__init__(in_doc, path_or_stream, options)
        self._live_pages: set[ManagedPdfiumPageBackend] = set()
        self._live_pages_lock = threading.Lock()
        self._live_pages_cond = threading.Condition(self._live_pages_lock)
        self._closing = False
        self._closed = False

    def _register_live_page(self, page_backend: ManagedPdfiumPageBackend) -> None:
        with self._live_pages_cond:
            if self._closing or self._closed:
                raise RuntimeError(
                    "Cannot register a page while the document is closing."
                )
            self._live_pages.add(page_backend)

    def _release_live_page(self, page_backend: ManagedPdfiumPageBackend) -> None:
        with self._live_pages_cond:
            self._live_pages.discard(page_backend)
            self._live_pages_cond.notify_all()

    def _close_live_pages(self) -> None:
        while True:
            with self._live_pages_cond:
                live_pages = list(self._live_pages)
            if not live_pages:
                return
            for page_backend in live_pages:
                page_backend.unload()

    @abstractmethod
    def _close_native_document(self) -> None:
        pass

    def unload(self) -> None:
        with self._live_pages_cond:
            if self._closed:
                return
            self._closing = True

        try:
            self._close_live_pages()
            self._close_native_document()
        finally:
            with self._live_pages_cond:
                self._closed = True
                self._closing = False
                self._live_pages.clear()
                self._live_pages_cond.notify_all()

        super().unload()


class ManagedPdfiumPageBackend(PdfPageBackend, ABC):
    """Shared page lifecycle for PDFium-backed page backends."""

    def __init__(self, owner: ManagedPdfiumDocumentBackend) -> None:
        self._owner: ManagedPdfiumDocumentBackend | None = owner
        self._closed = False
        owner._register_live_page(self)

    @abstractmethod
    def _close_native_page(self) -> None:
        pass

    def unload(self) -> None:
        if self._closed:
            return

        owner = self._owner
        try:
            self._close_native_page()
        finally:
            self._closed = True
            self._owner = None
            if owner is not None:
                owner._release_live_page(self)
