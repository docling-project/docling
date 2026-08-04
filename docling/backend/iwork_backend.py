"""Backends for Apple iWork documents.

Currently limited to Pages (``.pages``). A ``.pages`` file is a ZIP container, but
what is inside changed completely with Pages 5:

* **iWork '09 and earlier** stored the document as ``index.xml`` (optionally
  gzipped) and, when "Include preview in document" was enabled, a full
  ``QuickLook/Preview.pdf`` render alongside it.
* **Pages 5 and later (2013 onwards)** store the document as ``Index/*.iwa`` —
  Snappy-compressed protobuf whose schemas Apple has never published. The
  QuickLook PDF is gone; the only previews are root-level ``preview.jpg``,
  ``preview-micro.jpg`` and ``preview-web.jpg``, which cover just the top of the
  first page at roughly 720x552 and are useless as a document render.

This backend converts the '09-era preview PDF through the regular PDF pipeline.
Modern documents carry nothing convertible, so they are rejected with an error
that says why rather than offering advice the user cannot act on. Supporting them
requires decoding the IWA archives, which is not implemented here.
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
    """Convert iWork '09-era Pages documents via their embedded preview PDF.

    The backend validates the container, extracts ``QuickLook/Preview.pdf`` and
    delegates every paginated operation to a nested PDF backend, so the document
    flows through :class:`~docling.pipeline.standard_pdf_pipeline.StandardPdfPipeline`
    unchanged.

    Known limitations:
        * **Pages 5+ (2013 onwards) documents cannot be converted.** They store
          their content in ``Index/*.iwa`` and embed no PDF render. This covers
          effectively every Pages document written in the last decade.
        * '09 documents saved without an embedded preview cannot be converted.
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
            # Distinguish the two ways a preview can be absent. Telling a Pages 5+
            # user to enable "Include preview in document" would be useless advice:
            # that option, and the PDF it produced, no longer exist.
            if any(name.startswith(_MODERN_INDEX_PREFIX) for name in names):
                raise DocumentLoadError(
                    f"Pages document with hash {self.document_hash} was written by "
                    "Pages 5 or later (2013 onwards), which stores its content in "
                    "Index/*.iwa archives and embeds no PDF render. Docling cannot "
                    "decode that format yet. Export the document to PDF, DOCX or "
                    "EPUB from Pages and convert that instead."
                )
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} contains no "
                f"'{_PREVIEW_MEMBER}'. Docling converts iWork '09 Pages documents "
                "through their embedded preview, which Pages only wrote when "
                "'Include preview in document' was enabled. Re-save with that "
                "setting on, or export the document to PDF or DOCX."
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
