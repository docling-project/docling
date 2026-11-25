# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Convert MediaWiki XML exports and wikitext files."""

import tarfile
import zlib
from collections.abc import Iterator
from html import escape
from io import BytesIO
from pathlib import Path, PurePosixPath
from xml.etree.ElementTree import ParseError

from docling_core.types.doc import DoclingDocument, DocumentOrigin
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.xml.mediawiki._tokenizer_state import ParserError
from docling.backend.xml.mediawiki.export import read_wiki_pages
from docling.backend.xml.mediawiki.render import render_wikitext
from docling.datamodel.backend_options import (
    BackendOptions,
    HTMLBackendOptions,
    MediaWikiBackendOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument, get_input_rejection_cause
from docling.exceptions import DocumentLoadError
from docling.utils.mediawiki_input import (
    MediaWikiSource,
    MediaWikiSourceKind,
    open_mediawiki_sources,
)
from docling.utils.text_decoding import decode_text

_MAX_WIKITEXT_CHARS = 4 * 1024 * 1024
_MAX_WIKITEXT_BYTES = 4 * _MAX_WIKITEXT_CHARS


class MediaWikiDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Path | BytesIO,
        options: BackendOptions | None = None,
    ) -> None:
        super().__init__(in_doc, path_or_stream, options)
        self.mediawiki_options = (
            options
            if isinstance(options, MediaWikiBackendOptions)
            else MediaWikiBackendOptions()
        )

    @override
    def is_valid(self) -> bool:
        return self.path_or_stream is not None

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.MEDIAWIKI}

    @override
    def convert(self) -> DoclingDocument:
        stream = self.path_or_stream
        if stream is None:
            raise DocumentLoadError("MediaWiki input is closed.")
        name = self.file.name
        parts: list[DoclingDocument] = []
        origin: DocumentOrigin | None = None
        try:
            with open_mediawiki_sources(stream, name) as sources:
                for source in sources:
                    if origin is None:
                        origin = DocumentOrigin.model_validate(
                            {
                                "filename": name,
                                "mimetype": source.mimetype,
                                "binary_hash": self.document_hash,
                            }
                        )
                    parts.extend(self._convert_source(source))
        except (
            EOFError,
            OSError,
            ParseError,
            ParserError,
            tarfile.TarError,
            zlib.error,
        ) as exc:
            raise DocumentLoadError(
                f"Could not read MediaWiki input {name!r}: {exc}"
            ) from exc
        if origin is None:
            raise DocumentLoadError(f"No MediaWiki content found in {name!r}.")
        doc = parts[0] if len(parts) == 1 else DoclingDocument.concatenate(parts)
        doc.name = self.file.stem or "MediaWiki"
        doc.origin = origin
        return doc

    def _convert_source(self, source: MediaWikiSource) -> Iterator[DoclingDocument]:
        if source.kind is MediaWikiSourceKind.XML:
            for page in read_wiki_pages(source.stream):
                yield self._convert_article(page.wikitext, page.title, page.wiki_base)
            return

        data = source.stream.read(_MAX_WIKITEXT_BYTES + 1)
        if len(data) > _MAX_WIKITEXT_BYTES:
            raise DocumentLoadError(
                f"Wikitext input {source.name!r} exceeds the size limit."
            )
        title = PurePosixPath(source.name).stem if source.archive_member else ""
        yield self._convert_article(decode_text(BytesIO(data)), title)

    def _convert_article(
        self,
        wikitext: str,
        title: str,
        wiki_base: str | None = None,
    ) -> DoclingDocument:
        if len(wikitext) > _MAX_WIKITEXT_CHARS:
            raise DocumentLoadError(f"Wikitext page {title!r} exceeds the size limit.")
        wikitext = wikitext.replace("\r\n", "\n").replace("\r", "\n")
        html = render_wikitext(
            wikitext, wiki_base, self.mediawiki_options.reference_mode
        )
        if title:
            html = f"<h1>{escape(title)}</h1>{html}"
        input_doc = InputDocument(
            path_or_stream=BytesIO(
                f"<html><body>{html}</body></html>".encode("utf-8-sig")
            ),
            format=InputFormat.HTML,
            filename="wiki-page.html",
            backend=HTMLDocumentBackend,
            backend_options=HTMLBackendOptions(add_title=False, infer_furniture=False),
        )
        backend = input_doc._backend
        if not input_doc.valid or not isinstance(backend, HTMLDocumentBackend):
            raise DocumentLoadError("Could not parse rendered wikitext.") from (
                get_input_rejection_cause(input_doc)
            )
        try:
            return backend.convert()
        finally:
            input_doc.release()
