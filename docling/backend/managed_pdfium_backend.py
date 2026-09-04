# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import PdfBackendOptions

if TYPE_CHECKING:
    import pypdfium2 as pdfium

    from docling.datamodel.document import InputDocument
    from docling.utils.form_utils import FormFieldInfo


class ManagedPdfiumDocumentBackend(PdfDocumentBackend, ABC):
    """Shared lifecycle management for PDFium-backed document backends."""

    # Concrete backends hold their native document here and set it to None on close.
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

    def get_form_fields(self) -> list[FormFieldInfo]:
        """Read AcroForm widget fields from the underlying pypdfium2 document."""
        from docling.utils.form_utils import extract_form_fields

        if self._pdoc is None:
            return []
        return extract_form_fields(self._pdoc)

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
