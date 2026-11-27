# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Convert MediaWiki XML exports and wikitext files."""

import zlib
from collections.abc import Iterator
from html import escape
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError

from docling_core.types.doc import DoclingDocument, DocumentOrigin
from docling_core.types.doc.document import Formatting
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.xml.mediawiki._tokenizer_state import ParserError
from docling.backend.xml.mediawiki.export import read_wiki_pages
from docling.backend.xml.mediawiki.render import InlineReferenceBudget, render_wikitext
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
    open_mediawiki_source,
)
from docling.utils.text_decoding import decode_text


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
        options = self.mediawiki_options
        reference_budget = InlineReferenceBudget(options.max_inline_reference_bytes)
        try:
            with open_mediawiki_source(stream, name, options.max_total_bytes) as source:
                parts = list(self._convert_source(source, reference_budget))
                origin = DocumentOrigin.model_validate(
                    {
                        "filename": name,
                        "mimetype": source.kind.value,
                        "binary_hash": self.document_hash,
                    }
                )
        except (
            EOFError,
            OSError,
            ParseError,
            ParserError,
            zlib.error,
        ) as exc:
            raise DocumentLoadError(
                f"Could not read MediaWiki input {name!r}: {exc}"
            ) from exc
        if not parts:
            raise DocumentLoadError(f"No MediaWiki content found in {name!r}.")
        doc = parts[0] if len(parts) == 1 else DoclingDocument.concatenate(parts)
        doc.name = self.file.stem or "MediaWiki"
        doc.origin = origin
        if not options.preserve_bold or not options.preserve_italic:
            self._filter_inferred_formatting(doc)
        return doc

    def _filter_inferred_formatting(self, doc: DoclingDocument) -> None:
        """Apply the emphasis options to HTML-derived styles such as bold list terms."""
        options = self.mediawiki_options
        for item in doc.texts:
            if item.formatting is not None:
                formatting = item.formatting.model_copy(
                    update={
                        "bold": item.formatting.bold and options.preserve_bold,
                        "italic": item.formatting.italic and options.preserve_italic,
                    }
                )
                item.formatting = None if formatting == Formatting() else formatting

    def _convert_source(
        self, source: MediaWikiSource, reference_budget: InlineReferenceBudget
    ) -> Iterator[DoclingDocument]:
        options = self.mediawiki_options
        if source.kind is MediaWikiSourceKind.XML:
            for page in read_wiki_pages(source.stream, options.max_page_chars):
                yield self._convert_article(
                    page.wikitext, reference_budget, page.title, page.wiki_base
                )
            return

        data = source.stream.read(options.max_file_bytes + 1)
        if len(data) > options.max_file_bytes:
            raise DocumentLoadError(
                f"Wikitext input {source.name!r} exceeds max_file_bytes ({options.max_file_bytes})."
            )
        yield self._convert_article(decode_text(BytesIO(data)), reference_budget)

    def _convert_article(
        self,
        wikitext: str,
        reference_budget: InlineReferenceBudget,
        title: str = "",
        wiki_base: str | None = None,
    ) -> DoclingDocument:
        options = self.mediawiki_options
        if len(wikitext) > options.max_page_chars:
            raise DocumentLoadError(
                f"Wikitext page {title!r} exceeds max_page_chars ({options.max_page_chars})."
            )
        wikitext = wikitext.replace("\r\n", "\n").replace("\r", "\n")
        html = render_wikitext(
            wikitext,
            wiki_base,
            options.reference_mode,
            preserve_bold=options.preserve_bold,
            preserve_italic=options.preserve_italic,
            reference_budget=reference_budget,
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
            backend_options=HTMLBackendOptions(
                add_title=False, infer_furniture=False, fetch_images=False
            ),
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
