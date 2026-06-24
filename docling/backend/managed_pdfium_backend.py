from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import PdfOutlineItem
from docling.utils.pdf_outline import extract_outline_from_pdfium

if TYPE_CHECKING:
    import pypdfium2 as pdfium

    from docling.datamodel.document import InputDocument


class ManagedPdfiumDocumentBackend(PdfDocumentBackend, ABC):
    """Shared lifecycle management for PDFium-backed document backends."""

    # Set by concrete subclasses to the open PDFium document; reset to None on close.
    _pdoc: Optional[pdfium.PdfDocument]

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

    def get_document_outline(self) -> list[PdfOutlineItem]:
        """Extract the PDF outline from the open PDFium document.

        Shared by both the pypdfium2 and docling-parse backends (both hold a PDFium handle),
        so the bookmark signal is available under the default text backend too.
        """
        if self._pdoc is None:
            return []
        return extract_outline_from_pdfium(self._pdoc)

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
