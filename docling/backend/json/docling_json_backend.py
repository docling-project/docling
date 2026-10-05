# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import codecs
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import DoclingDocument, ImageRef
from pydantic import AnyUrl, BaseModel
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import BackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

_KEPT_IMAGE_URI_SCHEMES = frozenset({"data", "http", "https"})


def _is_kept_image_uri(uri: AnyUrl | Path) -> bool:
    return isinstance(uri, AnyUrl) and uri.scheme.lower() in _KEPT_IMAGE_URI_SCHEMES


def _clear_local_image_refs(node: object) -> int:
    """Set every ``ImageRef`` field below ``node`` whose URI is local to ``None``.

    The whole model tree is walked, rather than a fixed list of item
    collections, so image fields on any item type (pictures, tables, code,
    forms, pages, ...) are covered, including ones added in future
    docling-core releases. Returns the number of references cleared.
    """
    cleared = 0
    if isinstance(node, BaseModel):
        for name, value in node:
            if isinstance(value, ImageRef):
                if not _is_kept_image_uri(value.uri):
                    setattr(node, name, None)
                    cleared += 1
            else:
                cleared += _clear_local_image_refs(value)
    elif isinstance(node, (list, tuple)):
        for child in node:
            cleared += _clear_local_image_refs(child)
    elif isinstance(node, dict):
        for child in node.values():
            cleared += _clear_local_image_refs(child)
    return cleared


class DoclingJSONBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[BackendOptions] = None,
    ) -> None:
        super().__init__(in_doc, path_or_stream, options)

        # given we need to store any actual conversion exception for raising it from
        # convert(), this captures the successful result or the actual error in a
        # mutually exclusive way:
        self._doc_or_err = self._get_doc_or_err()

    @override
    def is_valid(self) -> bool:
        return isinstance(self._doc_or_err, DoclingDocument)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.JSON_DOCLING}

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        # A leading BOM is rejected by model_validate_json as an unexpected
        # character, failing the whole load. utf-8-sig drops it when decoding,
        # and is equivalent to utf-8 when no BOM is present; the stream branch
        # never decodes, so the bytes are stripped directly instead.
        try:
            json_data: Union[str, bytes]
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8-sig") as f:
                    json_data = f.read()
            elif isinstance(self.path_or_stream, BytesIO):
                json_data = self.path_or_stream.getvalue().removeprefix(codecs.BOM_UTF8)
            else:
                raise RuntimeError(f"Unexpected: {type(self.path_or_stream)=}")
            doc = DoclingDocument.model_validate_json(json_data=json_data)
            if not self.options.enable_local_fetch:
                self._drop_local_image_refs(doc)
            return doc
        except Exception as e:
            return e

    def _drop_local_image_refs(self, doc: DoclingDocument) -> None:
        """Remove image references that point at the local filesystem.

        ``ImageRef.uri`` is document content, so a JSON input can name any
        file on the converting host, which later stages (picture enrichment,
        embedded-image export) would open. Only ``data:`` URIs and ``http(s)``
        URLs (never fetched by docling-core) are kept; bare paths, ``file:``
        URIs and any other scheme are dropped unless the caller sets
        ``enable_local_fetch`` on the backend options.

        A dropped reference is replaced by ``None``, so the image's size, dpi
        and mimetype are lost along with its URI.
        """
        dropped = _clear_local_image_refs(doc)
        if dropped:
            _log.warning(
                "%s: ignored %d image reference(s) pointing at local files. "
                "To load them from a trusted JSON input, pass "
                "DoclingJSONFormatOption(backend_options="
                "DeclarativeBackendOptions(enable_local_fetch=True)) for "
                "InputFormat.JSON_DOCLING in DocumentConverter(format_options=...).",
                self.file.name,
                dropped,
            )

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err
        else:
            raise self._doc_or_err
