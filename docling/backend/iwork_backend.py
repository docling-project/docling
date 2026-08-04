"""Backends for Apple iWork documents.

Currently limited to Pages (``.pages``). A ``.pages`` file is a ZIP container whose
layout depends on when it was written:

* iWork '13 and later store the document in ``Index/*.iwa`` — Snappy-compressed
  protobuf streams whose schemas Apple has never published.
* iWork '09 stored it in a plain ``index.xml`` (optionally gzipped).

Rather than decode either representation, this backend converts the
``QuickLook/Preview.pdf`` that Pages embeds in the container and hands it to the
regular PDF pipeline. Layout analysis, table structure and OCR therefore behave
exactly as they do for a PDF, at the cost of losing the semantics Pages itself
knows about (real heading levels, list nesting, alt text).

The preview is written whenever "Include preview in document" is enabled, which is
the default, but it is not guaranteed: documents saved with that setting off carry
no preview and are rejected with an explanatory error.
"""

import logging
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

from typing_extensions import override

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import IWorkBackendOptions, PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError
from docling.utils.pdf_outline import _PdfOutlineItem

_log = logging.getLogger(__name__)

_PREVIEW_MEMBER = "QuickLook/Preview.pdf"

# Members that identify a container as a Pages document. Modern documents carry an
# Index/ directory of IWA archives; iWork '09 carried a single index.xml.
_MODERN_INDEX_PREFIX = "Index/"
_LEGACY_INDEX_MEMBERS = ("index.xml", "index.xml.gz")


class IWorkPagesDocumentBackend(PdfDocumentBackend):
    """Convert Apple Pages documents via their embedded QuickLook preview PDF.

    The backend validates the container, extracts ``QuickLook/Preview.pdf`` and
    delegates every paginated operation to a nested PDF backend, so the document
    flows through :class:`~docling.pipeline.standard_pdf_pipeline.StandardPdfPipeline`
    unchanged.

    Known limitations:
        * Documents saved without an embedded preview cannot be converted.
        * ``.pages`` bundles saved as a *directory* package rather than a single
          file are not recognised; the converter cannot address a directory as an
          input document.
        * Structure comes from layout analysis of the preview, not from the Pages
          document model, so heading levels and list nesting are inferred.
        * The converted document's ``origin.mimetype`` reads ``application/pdf``,
          because StandardPdfPipeline stamps that on everything it produces. The
          original filename is preserved. Image and METS-GBS inputs behave the
          same way.
        * The nested PDF backend is fixed; ``--pdf-backend`` does not apply to the
          preview.
    """

    #: PDF backend used for the extracted preview. Overridable by subclasses and
    #: swapped in tests; resolved lazily so importing this module does not require
    #: the PDF extras to be installed.
    pdf_backend_cls: Optional[type[PdfDocumentBackend]] = None

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[PdfBackendOptions] = None,
    ):
        # The CLI and any caller reusing PdfFormatOption hand this backend plain
        # PDF backend options; widen them so the archive limits always exist while
        # password and font settings still reach the nested PDF backend.
        if options is None:
            iwork_options = IWorkBackendOptions()
        elif isinstance(options, IWorkBackendOptions):
            iwork_options = options
        else:
            iwork_options = IWorkBackendOptions(**options.model_dump())

        super().__init__(in_doc, path_or_stream, iwork_options)
        self.options: IWorkBackendOptions = iwork_options

        self._archive: Optional[zipfile.ZipFile] = None
        self._preview_stream: Optional[BytesIO] = None
        self._inner: Optional[PdfDocumentBackend] = None

        try:
            self._archive = zipfile.ZipFile(path_or_stream)
            preview = self._read_preview(self._archive)
        except DocumentLoadError:
            self._close_archive()
            raise
        except (zipfile.BadZipFile, OSError) as exc:
            self._close_archive()
            raise DocumentLoadError(
                f"Could not open Pages document with hash {self.document_hash}: "
                "the file is not a readable ZIP container."
            ) from exc

        self._preview_stream = BytesIO(preview)
        self._inner = self._build_inner_backend(in_doc, self._preview_stream)

    @staticmethod
    def _resolve_pdf_backend_cls() -> type[PdfDocumentBackend]:
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

        return DoclingParseDocumentBackend

    def _read_preview(self, archive: zipfile.ZipFile) -> bytes:
        """Locate and read the QuickLook preview, enforcing archive limits."""
        infos = archive.infolist()
        if len(infos) > self.options.max_member_count:
            raise DocumentLoadError(
                f"Pages archive has {len(infos)} members, exceeding the "
                f"max_member_count limit of {self.options.max_member_count}."
            )

        names = {info.filename for info in infos}
        is_pages_container = any(
            name.startswith(_MODERN_INDEX_PREFIX) for name in names
        ) or any(name in names for name in _LEGACY_INDEX_MEMBERS)
        if not is_pages_container:
            raise DocumentLoadError(
                f"Document with hash {self.document_hash} is a ZIP archive but does "
                "not look like a Pages document: it has neither an Index/ directory "
                "nor an index.xml."
            )

        total_bytes = sum(info.file_size for info in infos)
        if total_bytes > self.options.max_total_bytes:
            raise DocumentLoadError(
                f"Pages archive expands to {total_bytes} bytes, exceeding the "
                f"max_total_bytes limit of {self.options.max_total_bytes}."
            )

        preview_info = next(
            (info for info in infos if info.filename == _PREVIEW_MEMBER), None
        )
        if preview_info is None:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} contains no "
                f"'{_PREVIEW_MEMBER}'. Docling converts Pages documents through "
                "their embedded preview, which Pages only writes when 'Include "
                "preview in document' is enabled. Re-save the document with that "
                "setting on, or export it to PDF or DOCX."
            )
        if preview_info.file_size > self.options.max_file_bytes:
            raise DocumentLoadError(
                f"Pages preview is {preview_info.file_size} bytes, exceeding the "
                f"max_file_bytes limit of {self.options.max_file_bytes}."
            )

        preview = archive.read(preview_info)
        if not preview:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} has an empty "
                f"'{_PREVIEW_MEMBER}'."
            )
        return preview

    def _build_inner_backend(
        self, in_doc: InputDocument, stream: BytesIO
    ) -> PdfDocumentBackend:
        backend_cls = self.pdf_backend_cls or self._resolve_pdf_backend_cls()

        # The nested backend must see a PDF InputDocument: PdfDocumentBackend
        # rejects any format outside its supported set. Reusing the outer limits
        # keeps page-count and size policy consistent across the two layers.
        pdf_options = PdfBackendOptions(
            enable_remote_fetch=self.options.enable_remote_fetch,
            enable_local_fetch=self.options.enable_local_fetch,
            password=self.options.password,
            enforce_same_font=self.options.enforce_same_font,
        )
        preview_doc = InputDocument(
            path_or_stream=stream,
            format=InputFormat.PDF,
            backend=backend_cls,
            filename=f"{in_doc.file.stem or 'document'}.pdf",
            backend_options=pdf_options,
            limits=in_doc.limits,
        )
        if not preview_doc.valid:
            raise DocumentLoadError(
                f"The preview PDF embedded in Pages document with hash "
                f"{self.document_hash} could not be loaded."
            )

        inner = preview_doc._backend
        if not isinstance(inner, PdfDocumentBackend):
            raise DocumentLoadError(
                f"{backend_cls.__name__} is not a PDF document backend."
            )
        return inner

    def _require_inner(self) -> PdfDocumentBackend:
        if self._inner is None:
            raise RuntimeError(
                f"Pages backend for document with hash {self.document_hash} has "
                "already been unloaded."
            )
        return self._inner

    @override
    def is_valid(self) -> bool:
        return self._inner is not None and self._inner.is_valid()

    @override
    def page_count(self) -> int:
        return self._require_inner().page_count()

    @override
    def load_page(self, page_no: int) -> PdfPageBackend:
        return self._require_inner().load_page(page_no)

    @override
    def iter_pages(self) -> Iterator[PdfPageBackend]:
        return self._require_inner().iter_pages()

    @override
    def get_document_outline(self) -> list[_PdfOutlineItem]:
        return self._require_inner().get_document_outline()

    @classmethod
    @override
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.IWORK_PAGES}

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    def _close_archive(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @override
    def unload(self):
        if self._inner is not None:
            self._inner.unload()
            self._inner = None
        self._close_archive()
        if self._preview_stream is not None:
            self._preview_stream.close()
            self._preview_stream = None
        super().unload()
