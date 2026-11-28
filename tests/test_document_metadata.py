# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Offline unit tests for structured document metadata extraction."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import BaseMeta
from odfdo import Cell, Document as OdfDocument, DrawPage, Frame, Paragraph, Row, Table

from docling.backend.epub_backend import EpubDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.opendocument_backend import (
    OdpDocumentBackend,
    OdsDocumentBackend,
    OdtDocumentBackend,
)
from docling.datamodel.backend_options import (
    DeclarativeBackendOptions,
    EpubBackendOptions,
    HTMLBackendOptions,
    MarkdownBackendOptions,
    OdsBackendOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.document_metadata import (
    DOCUMENT_METADATA_META_KEY,
    DocumentMetadata,
)


def _get_doc_metadata(doc: DoclingDocument) -> dict | None:
    if doc.body.meta is None:
        return None
    dumped = doc.body.meta.model_dump(exclude_none=True)
    return dumped.get(DOCUMENT_METADATA_META_KEY)


def _build_epub_bytes(metadata_xml: str) -> BytesIO:
    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            "<rootfiles>"
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles>"
            "</container>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f"{metadata_xml}"
            "</metadata>"
            "<manifest>"
            '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine>'
            "</package>",
        )
        zf.writestr(
            "OEBPS/ch1.xhtml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<body><h1>Chapter 1</h1><p>Body text.</p></body>"
            "</html>",
        )
    buf.seek(0)
    return buf


def _build_odf_bytes(kind: str, *, with_meta: bool = True) -> BytesIO:
    odf_doc = OdfDocument(kind)
    odf_doc.body.clear()
    if kind == "text":
        odf_doc.body.append(Paragraph("Hello ODT"))
    elif kind == "presentation":
        page = DrawPage("slide1", name="Slide 1")
        page.append(
            Frame.text_frame("Hello ODP", size=("10cm", "2cm"), position=("1cm", "1cm"))
        )
        odf_doc.body.append(page)
    elif kind == "spreadsheet":
        table = Table("Sheet1")
        row = Row()
        row.append_cell(Cell("Header"))
        table.append_row(row)
        odf_doc.body.append(table)

    if with_meta:
        meta = odf_doc.meta
        meta.set_title(f"Sample {kind.upper()}")
        meta.set_creator("Alice Smith")
        meta.set_language("en-US")
        meta.set_description("ODF description")
        meta.set_keywords("odf, docling")

    buf = BytesIO()
    odf_doc.save(buf)
    buf.seek(0)
    return buf


def test_document_metadata_model_attach_and_preserve():
    doc = DoclingDocument(name="test")
    empty = DocumentMetadata()
    assert empty.is_empty()
    empty.attach_to_document(doc)
    assert doc.body.meta is None

    doc.body.meta = BaseMeta.model_validate({"docling__existing": {"k": "v"}})
    meta = DocumentMetadata(
        title="Doc Title",
        authors=["Alice", "Bob"],
        date="2026-11-23",
        language="en",
        keywords=["rag", "pdf"],
    )
    meta.attach_to_document(doc)

    assert doc.body.meta is not None
    dumped = doc.body.meta.model_dump(exclude_none=True)
    assert dumped["docling__existing"] == {"k": "v"}
    assert dumped[DOCUMENT_METADATA_META_KEY] == {
        "title": "Doc Title",
        "authors": ["Alice", "Bob"],
        "date": "2026-11-23",
        "language": "en",
        "keywords": ["rag", "pdf"],
    }

    reloaded = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert _get_doc_metadata(reloaded) == dumped[DOCUMENT_METADATA_META_KEY]


def test_markdown_metadata_extraction_and_opt_in():
    content = (
        b"---\n"
        b"title: Research Note\n"
        b"authors:\n"
        b"  - Alice\n"
        b"  - Bob\n"
        b"date: 2026-11-23\n"
        b"lang: en\n"
        b"description: Summary text\n"
        b"tags: [ai, parsing]\n"
        b"doi: 10.1000/xyz\n"
        b"status: draft\n"
        b"---\n\n"
        b"# Heading\n\nBody paragraph.\n"
    )

    # Disabled by default -> body.meta stays None
    stream_default = BytesIO(content)
    in_default = InputDocument(
        path_or_stream=stream_default,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="note.md",
    )
    doc_default = MarkdownDocumentBackend(in_default, stream_default).convert()
    assert _get_doc_metadata(doc_default) is None

    # Enabled -> structured metadata populated
    stream_enabled = BytesIO(content)
    opts = MarkdownBackendOptions(extract_metadata=True)
    in_enabled = InputDocument(
        path_or_stream=stream_enabled,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="note.md",
        backend_options=opts,
    )
    doc_enabled = MarkdownDocumentBackend(in_enabled, stream_enabled, opts).convert()
    meta = _get_doc_metadata(doc_enabled)
    assert meta == {
        "title": "Research Note",
        "authors": ["Alice", "Bob"],
        "date": "2026-11-23",
        "language": "en",
        "description": "Summary text",
        "keywords": ["ai", "parsing"],
        "identifier": "10.1000/xyz",
        "custom": {"status": "draft"},
    }


def test_markdown_metadata_with_html_block_and_malformed_frontmatter():
    md_with_html = (
        b"---\ntitle: HTML Block Doc\nauthor: Carol\n---\n\n"
        b"<div><p>Inline HTML block</p></div>\n"
    )
    stream = BytesIO(md_with_html)
    opts = MarkdownBackendOptions(extract_metadata=True)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="html_block.md",
        backend_options=opts,
    )
    doc = MarkdownDocumentBackend(in_doc, stream, opts).convert()
    assert _get_doc_metadata(doc) == {
        "title": "HTML Block Doc",
        "authors": ["Carol"],
    }

    malformed = b"---\ntitle: [unclosed\n---\n\n# Safe Body\n"
    stream_bad = BytesIO(malformed)
    in_bad = InputDocument(
        path_or_stream=stream_bad,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="bad.md",
        backend_options=opts,
    )
    doc_bad = MarkdownDocumentBackend(in_bad, stream_bad, opts).convert()
    assert _get_doc_metadata(doc_bad) is None


def test_html_metadata_extraction():
    html = (
        b'<!DOCTYPE html><html lang="zh-CN"><head>'
        b"<title>Fallback Title</title>"
        b'<meta property="og:title" content="Primary HTML Title">'
        b'<meta name="author" content="Alice, Bob">'
        b'<meta name="citation_author" content="Bob">'
        b'<meta name="date" content="2026-11-23">'
        b'<meta name="description" content="Page summary">'
        b'<meta name="keywords" content="docling, html">'
        b'<meta property="og:site_name" content="Example Press">'
        b'<link rel="canonical" href="https://example.org/doc">'
        b'<meta name="empty" content="   ">'
        b"</head><body><h1>Main</h1><p>Content</p></body></html>"
    )

    stream_default = BytesIO(html)
    in_default = InputDocument(
        path_or_stream=stream_default,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="page.html",
    )
    assert (
        _get_doc_metadata(HTMLDocumentBackend(in_default, stream_default).convert())
        is None
    )

    stream_enabled = BytesIO(html)
    opts = HTMLBackendOptions(extract_metadata=True)
    in_enabled = InputDocument(
        path_or_stream=stream_enabled,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="page.html",
        backend_options=opts,
    )
    doc = HTMLDocumentBackend(in_enabled, stream_enabled, opts).convert()
    assert _get_doc_metadata(doc) == {
        "title": "Primary HTML Title",
        "authors": ["Alice", "Bob"],
        "date": "2026-11-23",
        "language": "zh-CN",
        "description": "Page summary",
        "keywords": ["docling", "html"],
        "publisher": "Example Press",
        "identifier": "https://example.org/doc",
    }


def test_epub_metadata_extraction():
    epub_stream = _build_epub_bytes(
        "<dc:title>EPUB Guide</dc:title>"
        "<dc:creator>Alice</dc:creator>"
        "<dc:creator>Bob</dc:creator>"
        "<dc:publisher>Open Press</dc:publisher>"
        "<dc:date>2026-05-01</dc:date>"
        "<dc:language>en</dc:language>"
        "<dc:description>E-book summary</dc:description>"
        "<dc:subject>ebooks</dc:subject>"
        "<dc:subject>parsing</dc:subject>"
        '<dc:identifier id="uid">urn:isbn:9780000000001</dc:identifier>'
    )
    opts = EpubBackendOptions(extract_metadata=True)
    in_doc = InputDocument(
        path_or_stream=epub_stream,
        format=InputFormat.EPUB,
        backend=EpubDocumentBackend,
        filename="guide.epub",
        backend_options=opts,
    )
    doc = EpubDocumentBackend(in_doc, epub_stream, opts).convert()
    assert _get_doc_metadata(doc) == {
        "title": "EPUB Guide",
        "authors": ["Alice", "Bob"],
        "date": "2026-05-01",
        "language": "en",
        "description": "E-book summary",
        "keywords": ["ebooks", "parsing"],
        "publisher": "Open Press",
        "identifier": "urn:isbn:9780000000001",
    }

    # Empty metadata block stays clean
    empty_stream = _build_epub_bytes("")
    in_empty = InputDocument(
        path_or_stream=empty_stream,
        format=InputFormat.EPUB,
        backend=EpubDocumentBackend,
        filename="empty.epub",
        backend_options=opts,
    )
    assert (
        _get_doc_metadata(EpubDocumentBackend(in_empty, empty_stream, opts).convert())
        is None
    )


def test_opendocument_metadata_extraction():
    cases = [
        ("text", InputFormat.ODT, OdtDocumentBackend, "doc.odt"),
        ("presentation", InputFormat.ODP, OdpDocumentBackend, "deck.odp"),
        ("spreadsheet", InputFormat.ODS, OdsDocumentBackend, "sheet.ods"),
    ]
    for kind, fmt, backend_cls, fname in cases:
        stream = _build_odf_bytes(kind, with_meta=True)
        opts = (
            OdsBackendOptions(extract_metadata=True)
            if fmt == InputFormat.ODS
            else DeclarativeBackendOptions(extract_metadata=True)
        )
        in_doc = InputDocument(
            path_or_stream=stream,
            format=fmt,
            backend=backend_cls,
            filename=fname,
            backend_options=opts,
        )
        assert isinstance(in_doc._backend, backend_cls)
        meta = _get_doc_metadata(in_doc._backend.convert())
        assert meta is not None
        assert meta["title"] == f"Sample {kind.upper()}"
        assert meta["authors"] == ["Alice Smith"]
        assert meta["language"] == "en-US"
        assert meta["description"] == "ODF description"
        assert meta["keywords"] == ["odf", "docling"]
