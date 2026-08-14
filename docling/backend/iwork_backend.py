"""Backends for Apple iWork documents.

Currently limited to Pages (``.pages``). A ``.pages`` file is a ZIP container, but
what is inside changed completely with Pages 5:

* **Pages 5 and later (2013 onwards)** store the document as ``Index/*.iwa`` —
  Snappy-framed protobuf whose schemas Apple has never published. This is what
  essentially every Pages document in circulation looks like.
* **iWork '09 and earlier** stored it as a plain ``index.xml``, optionally
  gzipped, alongside a ``QuickLook/Preview.pdf`` render that Apple stopped
  writing after that release.

Both generations are read for their text here, so the backend is declarative: it
builds a :class:`~docling_core.types.doc.DoclingDocument` directly rather than
rendering pages and running layout analysis over them.

Only body text is extracted. Heading levels, lists and tables are carried by the
paragraph style runs (IWA) and style attributes (XML) and are not yet mapped.
"""

import gzip
import logging
import mimetypes
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

import defusedxml.ElementTree as ET
from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.iwork_iwa import iter_objects, read_fields, read_reference
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_PAGES_MIMETYPE = "application/vnd.apple.pages"

# DocumentOrigin only accepts a mimetype that the stdlib knows or that
# docling-core allow-lists, and Python ships no mapping for ".pages". Teaching
# the stdlib the real Apple type keeps the origin honest without waiting on a
# docling-core release; it also makes mimetypes.guess_type() correct for callers.
mimetypes.add_type(_PAGES_MIMETYPE, ".pages")

_MODERN_INDEX_PREFIX = "Index/"
_LEGACY_INDEX_MEMBERS = ("index.xml", "index.xml.gz")

# Message type numbers from the reverse-engineered iWork schemas. Only these two
# are needed to reach the body text.
_TP_DOCUMENT_ARCHIVE = 10000
_TSWP_STORAGE_ARCHIVE = 2001

# TP.DocumentArchive field 4 references the body TSWP.StorageArchive, whose
# field 3 holds the text. Verified against Pages 5+ documents.
_DOCUMENT_BODY_FIELD = 4
_STORAGE_TEXT_FIELD = 3

_SF_NAMESPACE = "http://developer.apple.com/namespaces/sf"
_SF_PARAGRAPH = f"{{{_SF_NAMESPACE}}}p"
# iWork '09 placeholder text. It is what the template shows before the author
# types anything, so it must never be emitted as document content.
_SF_GHOST_TEXT = f"{{{_SF_NAMESPACE}}}ghost-text"

# Apple marks inline attachments (images, footnote anchors) with U+FFFC inside
# the text run. There is no text there to emit.
_OBJECT_REPLACEMENT = "￼"


class IWorkPagesDocumentBackend(DeclarativeDocumentBackend):
    """Extract text from Apple Pages documents of either generation.

    Known limitations:
        * Only body text is produced. Heading levels, lists and tables are not
          yet recovered, so the output is a flat sequence of paragraphs.
        * Text boxes, headers, footers, footnotes and comments are not included;
          only the main body storage is read.
        * Password-protected documents cannot be read.
        * ``.pages`` bundles saved as a *directory* package rather than a single
          file are not recognised; the converter cannot address a directory as an
          input document.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[IWorkBackendOptions] = None,
    ):
        if options is None:
            options = IWorkBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options: IWorkBackendOptions = options

        self._paragraphs: list[str] = []
        self._valid = False

        try:
            with zipfile.ZipFile(path_or_stream) as archive:
                self._paragraphs = self._read_paragraphs(archive)
        except DocumentLoadError:
            raise
        except (NotImplementedError, RuntimeError) as exc:
            # Pages encrypts members with a scheme zipfile cannot read, leaving a
            # nonsense compress_type rather than setting the standard encrypted
            # flag, so this is the only reliable signal that a document is
            # password-protected.
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} appears to be "
                "password-protected; Docling cannot read encrypted iWork "
                "documents. Remove the password in Pages and save again."
            ) from exc
        except (zipfile.BadZipFile, OSError) as exc:
            raise DocumentLoadError(
                f"Could not open Pages document with hash {self.document_hash}: "
                "the file is not a readable ZIP container."
            ) from exc

        self._valid = True

    def _read_paragraphs(self, archive: zipfile.ZipFile) -> list[str]:
        """Dispatch to the reader for whichever generation wrote the container."""
        infos = archive.infolist()
        if len(infos) > self.options.max_member_count:
            raise DocumentLoadError(
                f"Pages archive has {len(infos)} members, exceeding the "
                f"max_member_count limit of {self.options.max_member_count}."
            )
        total_bytes = sum(info.file_size for info in infos)
        if total_bytes > self.options.max_total_bytes:
            raise DocumentLoadError(
                f"Pages archive expands to {total_bytes} bytes, exceeding the "
                f"max_total_bytes limit of {self.options.max_total_bytes}."
            )

        names = {info.filename for info in infos}
        if any(name.startswith(_MODERN_INDEX_PREFIX) for name in names):
            return self._read_iwa_paragraphs(archive, infos)

        legacy = next((n for n in _LEGACY_INDEX_MEMBERS if n in names), None)
        if legacy is not None:
            return self._read_legacy_paragraphs(archive, legacy)

        raise DocumentLoadError(
            f"Document with hash {self.document_hash} is a ZIP archive but does "
            "not look like a Pages document: it has neither an Index/ directory "
            "nor an index.xml."
        )

    def _read_iwa_paragraphs(
        self, archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
    ) -> list[str]:
        """Read body text from the IWA object graph of a Pages 5+ document."""
        objects = {}
        for info in infos:
            if not info.filename.endswith(".iwa"):
                continue
            if info.file_size > self.options.max_file_bytes:
                raise DocumentLoadError(
                    f"Pages archive member {info.filename} is {info.file_size} "
                    f"bytes, exceeding the max_file_bytes limit of "
                    f"{self.options.max_file_bytes}."
                )
            for obj in iter_objects(archive.read(info)):
                objects[obj.identifier] = obj

        document = next(
            (o for o in objects.values() if o.message_type == _TP_DOCUMENT_ARCHIVE),
            None,
        )
        if document is None:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} has no "
                "TP.DocumentArchive; the container may be corrupt or "
                "password-protected."
            )

        body_ref = read_fields(document.payload).get(_DOCUMENT_BODY_FIELD, [None])[0]
        target = read_reference(body_ref) if isinstance(body_ref, bytes) else None
        storage = objects.get(target) if target is not None else None
        if storage is None or storage.message_type != _TSWP_STORAGE_ARCHIVE:
            raise DocumentLoadError(
                f"Pages document with hash {self.document_hash} does not reference "
                "a body text storage."
            )

        chunks = [
            value.decode("utf-8", errors="replace")
            for value in read_fields(storage.payload).get(_STORAGE_TEXT_FIELD, [])
            if isinstance(value, bytes)
        ]
        return _split_paragraphs("".join(chunks))

    def _read_legacy_paragraphs(
        self, archive: zipfile.ZipFile, member: str
    ) -> list[str]:
        """Read body text from the ``index.xml`` of an iWork '09 document."""
        raw = archive.read(member)
        if member.endswith(".gz"):
            try:
                raw = gzip.decompress(raw)
            except OSError as exc:
                raise DocumentLoadError(
                    f"Could not decompress '{member}' in Pages document with hash "
                    f"{self.document_hash}."
                ) from exc

        try:
            root = ET.fromstring(raw)
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not parse '{member}' in Pages document with hash "
                f"{self.document_hash}."
            ) from exc

        paragraphs = []
        for para in root.iter(_SF_PARAGRAPH):
            text = "".join(
                # itertext() would pull in the template placeholder text, which is
                # not document content.
                part
                for part in _iter_text_excluding_ghosts(para)
            )
            text = _clean(text)
            if text:
                paragraphs.append(text)
        return paragraphs

    @override
    def is_valid(self) -> bool:
        return self._valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.IWORK_PAGES}

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert Pages document with hash {self.document_hash} "
                "because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=_PAGES_MIMETYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        for paragraph in self._paragraphs:
            doc.add_text(label=DocItemLabel.TEXT, text=paragraph)
        return doc


def _iter_text_excluding_ghosts(element) -> list[str]:
    """Collect text under ``element``, skipping ``sf:ghost-text`` subtrees."""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if child.tag != _SF_GHOST_TEXT:
            parts.extend(_iter_text_excluding_ghosts(child))
        if child.tail:
            parts.append(child.tail)
    return parts


def _clean(text: str) -> str:
    return text.replace(_OBJECT_REPLACEMENT, "").strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split a TSWP text run into paragraphs.

    Apple separates paragraphs with newlines and pads empty ones, so blank
    results are dropped rather than emitted as empty text items.
    """
    paragraphs = []
    for line in text.split("\n"):
        cleaned = _clean(line)
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs
