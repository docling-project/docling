# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend for .cv resume files.

A .cv file (https://cvfile.org/spec/) is a PDF/A-3u that embeds its canonical
text as a PDF Associated File (ISO 32000-2, section 14.13) and names that
payload in a ``cv:`` XMP packet. The payload is Markdown or HTML. This backend
reads the payload and converts it with ``MarkdownDocumentBackend`` or
``HTMLDocumentBackend``. It does not parse the visual PDF layer, so the
conversion keeps the source text exactly and needs no layout or OCR model.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.backend_options import (
    DeclarativeBackendOptions,
    HTMLBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.base_models import FormatToMimeType, InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

# pypdf and defusedxml are only installed by the `format-cv` extra, but
# DocumentConverter imports every backend eagerly. Guard the imports like the
# email backend does, and raise only when a .cv file is actually parsed.
_CV_DEPENDENCIES_AVAILABLE: bool = False
_CV_DEPENDENCIES_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    import defusedxml.ElementTree as ET
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject, DictionaryObject, StreamObject

    _CV_DEPENDENCIES_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _CV_DEPENDENCIES_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'pypdf' and 'defusedxml' packages are required to process .cv files. "
    "Install them with `pip install 'docling-slim[format-cv]'`."
)

_MIME_TYPE = FormatToMimeType[InputFormat.CV][0]
_CV_NAMESPACE = "{http://ns.cvfile.org/cv/1.0/}"
_RDF_DESCRIPTION = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"

# The .cv specification caps each decoded payload at 16 MiB (section 7.3).
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_SUPPORTED_MAJOR_VERSION = 1
_SHA256_ALGORITHMS = {"sha-256", "sha256"}
_MARKDOWN_MIME_TYPE = "text/markdown"
_HTML_MIME_TYPE = "text/html"


@dataclass(frozen=True)
class _CvMetadata:
    """The ``cv:`` XMP properties that the backend uses."""

    version: str
    primary_payload: str
    sha256_digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CvPayload:
    """An embedded payload and its MIME type, without parameters."""

    data: bytes
    mime_type: str


class CvDocumentBackend(DeclarativeDocumentBackend):
    """Convert a .cv file from its embedded Markdown or HTML payload."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: DeclarativeBackendOptions | None = None,
    ):
        # Raised before super().__init__() so a missing optional dependency
        # gives an actionable message rather than a NameError below.
        if not _CV_DEPENDENCIES_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _CV_DEPENDENCIES_IMPORT_ERROR
        super().__init__(in_doc, path_or_stream, options)

        self.valid = False
        self.payload: _CvPayload | None = None
        try:
            self.payload = self._read_primary_payload(self._read_bytes())
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not initialize .cv backend for file with hash {self.document_hash}."
            ) from exc
        self.valid = True

    def _read_bytes(self) -> bytes:
        if isinstance(self.path_or_stream, BytesIO):
            return self.path_or_stream.getvalue()
        if isinstance(self.path_or_stream, Path):
            return self.path_or_stream.read_bytes()
        raise TypeError(f"Unsupported input type: {type(self.path_or_stream)}")

    def _read_primary_payload(self, data: bytes) -> _CvPayload:
        catalog = PdfReader(BytesIO(data)).root_object
        metadata = self._read_cv_metadata(catalog)
        major_version = int(metadata.version.split(".", 1)[0])
        if major_version > _SUPPORTED_MAJOR_VERSION:
            _log.warning(
                "The .cv file declares version %s, which is newer than the "
                "supported version %s.x. The embedded payload is converted anyway.",
                metadata.version,
                _SUPPORTED_MAJOR_VERSION,
            )
        payload = self._read_associated_file(catalog, metadata.primary_payload)
        if payload.mime_type not in (_MARKDOWN_MIME_TYPE, _HTML_MIME_TYPE):
            raise DocumentLoadError(
                f"The .cv primary payload {metadata.primary_payload!r} has the "
                f"unsupported MIME type {payload.mime_type!r}."
            )
        expected_digest = metadata.sha256_digests.get(metadata.primary_payload)
        if (
            expected_digest is not None
            and hashlib.sha256(payload.data).hexdigest() != expected_digest.lower()
        ):
            raise DocumentLoadError(
                f"The .cv payload {metadata.primary_payload!r} does not match its "
                "cv:integrity digest."
            )
        return payload

    @staticmethod
    def _read_cv_metadata(catalog: DictionaryObject) -> _CvMetadata:
        metadata_stream = catalog.get("/Metadata")
        if metadata_stream is None:
            raise DocumentLoadError("The .cv file has no XMP metadata stream.")
        xmp = ET.fromstring(metadata_stream.get_object().get_data())

        # RDF allows a property as a child element or as an attribute of
        # rdf:Description. Both forms are read.
        properties: dict[str, str] = {}
        for description in xmp.iter(_RDF_DESCRIPTION):
            for name, value in description.attrib.items():
                if name.startswith(_CV_NAMESPACE):
                    properties[name.removeprefix(_CV_NAMESPACE)] = value.strip()
            for child in description:
                if child.tag.startswith(_CV_NAMESPACE):
                    properties[child.tag.removeprefix(_CV_NAMESPACE)] = (
                        child.text or ""
                    ).strip()

        version = properties.get("version")
        primary_payload = properties.get("primaryPayload")
        if not version or not primary_payload:
            raise DocumentLoadError(
                "The .cv XMP metadata has no cv:version or cv:primaryPayload."
            )
        # cv:integrity is a JSON array of {payload, algorithm, digest} objects.
        integrity = json.loads(properties.get("integrity") or "[]")
        return _CvMetadata(
            version=version,
            primary_payload=primary_payload,
            sha256_digests={
                entry["payload"]: entry["digest"]
                for entry in integrity
                if entry["algorithm"].lower() in _SHA256_ALGORITHMS
            },
        )

    @staticmethod
    def _find_embedded_file(
        catalog: DictionaryObject, name: str
    ) -> tuple[DictionaryObject, StreamObject]:
        """Return the file specification and stream of an associated file."""
        associated_files = catalog.get("/AF")
        associated_files = (
            associated_files.get_object() if associated_files is not None else None
        )
        if not isinstance(associated_files, ArrayObject):
            raise DocumentLoadError("The .cv file has no /AF associated files.")

        for reference in associated_files:
            filespec = reference.get_object()
            if not isinstance(filespec, DictionaryObject):
                continue
            if str(filespec.get("/UF") or filespec.get("/F") or "") != name:
                continue
            embedded = filespec.get("/EF")
            embedded = embedded.get_object() if embedded is not None else None
            if not isinstance(embedded, DictionaryObject):
                continue
            stream = embedded.get("/UF") or embedded.get("/F")
            stream = stream.get_object() if stream is not None else None
            if isinstance(stream, StreamObject):
                return filespec, stream

        raise DocumentLoadError(f"The .cv file has no associated file named {name!r}.")

    @classmethod
    def _read_associated_file(cls, catalog: DictionaryObject, name: str) -> _CvPayload:
        filespec, stream = cls._find_embedded_file(catalog, name)
        # /Params /Size is the decoded size, which the specification requires.
        # Checking it first avoids decoding a declared oversized payload. pypdf
        # also bounds the output of its decoders.
        params = stream.get("/Params")
        params = params.get_object() if params is not None else None
        declared_size = (
            params.get("/Size") if isinstance(params, DictionaryObject) else None
        )
        if declared_size is not None and int(declared_size) > _MAX_PAYLOAD_BYTES:
            raise DocumentLoadError(
                f"The .cv payload {name!r} is larger than {_MAX_PAYLOAD_BYTES} bytes."
            )
        data = stream.get_data()
        if len(data) > _MAX_PAYLOAD_BYTES:
            raise DocumentLoadError(
                f"The .cv payload {name!r} is larger than {_MAX_PAYLOAD_BYTES} bytes."
            )
        # The specification puts /Subtype on the file specification, and the
        # PDF specification puts it on the embedded file stream.
        subtype = stream.get("/Subtype") or filespec.get("/Subtype") or ""
        mime_type = str(subtype).removeprefix("/").split(";", 1)[0].strip()
        return _CvPayload(data=data, mime_type=mime_type.lower())

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.CV}

    @classmethod
    def embeds_cv_payload(cls, path_or_stream: BytesIO | Path) -> bool:
        """Return whether a PDF is a .cv file with its primary payload.

        Format detection calls this to confirm a byte match of the .cv
        namespace, which can also occur in page content or in another stream.
        The catalog must carry the ``cv:`` XMP properties and embed the payload
        they name. A PDF tool can drop the attachments and keep the XMP, and
        such a file converts from its visual layer as a PDF.
        """
        if not _CV_DEPENDENCIES_AVAILABLE:
            return False
        position = path_or_stream.tell() if isinstance(path_or_stream, BytesIO) else 0
        try:
            catalog = PdfReader(path_or_stream).root_object
            metadata = cls._read_cv_metadata(catalog)
            cls._find_embedded_file(catalog, metadata.primary_payload)
        except Exception:
            return False
        finally:
            if isinstance(path_or_stream, BytesIO):
                path_or_stream.seek(position)
        return True

    @classmethod
    def dependencies_available(cls) -> bool:
        return (
            _CV_DEPENDENCIES_AVAILABLE
            and MarkdownDocumentBackend.dependencies_available()
            and HTMLDocumentBackend.dependencies_available()
        )

    def _convert_markdown(self, markdown: bytes) -> DoclingDocument:
        # The .cv specification requires UTF-8 text payloads.
        options = MarkdownBackendOptions(encoding="utf-8")
        stream = BytesIO(markdown)
        in_doc = InputDocument(
            path_or_stream=stream,
            format=InputFormat.MD,
            filename=self.file.name,
            backend=MarkdownDocumentBackend,
            backend_options=options,
        )
        stream.seek(0)
        return MarkdownDocumentBackend(
            in_doc=in_doc, path_or_stream=stream, options=options
        ).convert()

    def _convert_html(self, html: bytes) -> DoclingDocument:
        # The payload is UTF-8. The byte order mark makes the HTML parser use
        # UTF-8 and ignore a different charset declared in a <meta> tag.
        if not html.startswith(codecs.BOM_UTF8):
            html = codecs.BOM_UTF8 + html
        options = HTMLBackendOptions()
        stream = BytesIO(html)
        in_doc = InputDocument(
            path_or_stream=stream,
            format=InputFormat.HTML,
            filename=self.file.name,
            backend=HTMLDocumentBackend,
            backend_options=options,
        )
        stream.seek(0)
        return HTMLDocumentBackend(
            in_doc=in_doc, path_or_stream=stream, options=options
        ).convert()

    def convert(self) -> DoclingDocument:
        if not self.is_valid() or self.payload is None:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        doc = (
            self._convert_markdown(self.payload.data)
            if self.payload.mime_type == _MARKDOWN_MIME_TYPE
            else self._convert_html(self.payload.data)
        )
        doc.origin = DocumentOrigin(
            filename=self.file.name or "file.cv",
            mimetype=_MIME_TYPE,
            binary_hash=self.document_hash,
        )
        return doc
