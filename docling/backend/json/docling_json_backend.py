# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import codecs
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import DoclingDocument
from pydantic import AnyUrl
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import BackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


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

        ``ImageRef.uri`` is document content. A JSON document supplied by an
        untrusted party can name any path on the converting host, and later
        stages (picture enrichment, embedded-image export) would open it and
        return its bytes. Embedded ``data:`` URIs are kept; remote URLs are kept
        because docling-core never fetches them; ``file://`` URIs and bare paths
        are dropped unless the caller opts in with ``enable_local_fetch``.
        """
        dropped = 0
        holders = [
            *doc.pictures,
            *doc.tables,
            *doc.key_value_items,
            *doc.form_items,
            *doc.pages.values(),
        ]
        for holder in holders:
            image = getattr(holder, "image", None)
            if image is None:
                continue
            uri = image.uri
            is_local = isinstance(uri, Path) or (
                isinstance(uri, AnyUrl) and uri.scheme == "file"
            )
            if is_local:
                holder.image = None
                dropped += 1
        if dropped:
            _log.warning(
                "%s: dropped %d local image reference(s) from the JSON input; "
                "set enable_local_fetch=True on the backend options to keep them.",
                self.file.name,
                dropped,
            )

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err
        else:
            raise self._doc_or_err
