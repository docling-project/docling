from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

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
        options: Optional[PdfBackendOptions] = None,
    ) -> None:
        if options is None:
            options = PdfBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self._closed = False

    @abstractmethod
    def _close_native_document(self) -> None:
        pass

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_native_document()
        super().unload()


class ManagedPdfiumPageBackend(PdfPageBackend, ABC):
    """Shared page lifecycle for PDFium-backed page backends."""

    def __init__(self) -> None:
        self._closed = False

    @abstractmethod
    def _close_native_page(self) -> None:
        pass

    def unload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_native_page()
