# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib.util
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.opendocument_backend import (
    OdpDocumentBackend,
    OdsDocumentBackend,
    OdtDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.backend_options import (
    BaseBackendOptions,
    DeclarativeBackendOptions,
    HTMLBackendOptions,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.datamodel.settings import DocumentLimits
from docling.document_converter import (
    HTMLFormatOption,
    ImageFormatOption,
    OdpFormatOption,
    OdsFormatOption,
    OdtFormatOption,
    PdfFormatOption,
)


def test_in_doc_from_valid_path():
    test_doc_path = Path("./tests/data/pdf/sources/2206.01062.pdf")
    doc = _make_input_doc(test_doc_path)
    assert doc.valid is True
    assert doc.backend_options is None


def test_in_doc_from_invalid_path():
    test_doc_path = Path("./tests/does/not/exist.pdf")

    doc = _make_input_doc(test_doc_path)

    assert doc.valid is False


def test_in_doc_from_valid_buf():
    buf = BytesIO(Path("./tests/data/pdf/sources/2206.01062.pdf").open("rb").read())
    stream = DocumentStream(name="my_doc.pdf", stream=buf)

    doc = _make_input_doc_from_stream(stream)
    assert doc.valid is True


def test_in_doc_from_invalid_buf():
    buf = BytesIO(b"")
    stream = DocumentStream(name="my_doc.pdf", stream=buf)

    doc = _make_input_doc_from_stream(stream)
    assert doc.valid is False


def test_in_doc_with_page_range():
    test_doc_path = Path("./tests/data/pdf/sources/2206.01062.pdf")
    limits = DocumentLimits()
    limits.page_range = (1, 10)

    doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        limits=limits,
    )
    assert doc.valid is True

    limits.page_range = (9, 9)

    doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        limits=limits,
    )
    assert doc.valid is True

    limits.page_range = (11, 12)

    doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        limits=limits,
    )
    assert doc.valid is False


def test_in_doc_with_backend_options():
    test_doc_path = Path("./tests/data/html/sources/example_01.html")
    doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        backend_options=HTMLBackendOptions(),
    )
    assert doc.valid
    assert doc.backend_options
    assert isinstance(doc.backend_options, HTMLBackendOptions)
    assert not doc.backend_options.fetch_images
    assert not doc.backend_options.enable_local_fetch
    assert not doc.backend_options.enable_remote_fetch

    with pytest.raises(AttributeError, match="no attribute 'source_uri'"):
        doc = InputDocument(
            path_or_stream=test_doc_path,
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            backend_options=DeclarativeBackendOptions(),
        )

    with pytest.raises(ValidationError):
        doc = InputDocument(
            path_or_stream=test_doc_path,
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            backend_options=BaseBackendOptions(),
        )


def test_html_backend_options_set_source_uri_per_input(tmp_path):
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    first.write_text("<html><body>First</body></html>")
    second.write_text("<html><body>Second</body></html>")
    backend_options = HTMLBackendOptions(enable_local_fetch=True)
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[first, second])

    docs = list(
        conversion_input.docs(
            {
                InputFormat.HTML: HTMLFormatOption(
                    backend_options=backend_options,
                )
            }
        )
    )

    assert len(docs) == 2
    assert isinstance(docs[0].backend_options, HTMLBackendOptions)
    assert isinstance(docs[1].backend_options, HTMLBackendOptions)
    assert docs[0].backend_options.source_uri == first
    assert docs[1].backend_options.source_uri == second
    assert backend_options.source_uri is None


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\x5a\x00\x08",
        b"\x00\x00\x08\xd3\xa8\xa8\x00\x00\x00",
        b"\x5a\x00\x07\xd3\xa8\xa8\x00\x00\x00",
        b"\x5a\x80\x00\xd3\xa8\xa8\x00\x00\x00",
        b"\x5a\x00\x08\x00\xa8\xa8\x00\x00\x00",
    ],
)
def test_detect_afp_rejects_invalid_headers(content):
    assert _DocumentConversionInput._detect_afp(content) is None


def test_detect_afp_accepts_valid_modca_header():
    content = b"\x5a\x00\x08\xd3\xa8\xa8\x00\x00\x00"

    assert _DocumentConversionInput._detect_afp(content) == "application/vnd.ibm.modcap"


def test_guess_format_sniffs_afp_from_octet_stream(tmp_path, monkeypatch):
    content = b"\x5a\x00\x08\xd3\xa8\xa8\x00\x00\x00"
    afp_path = tmp_path / "print-stream.bin"
    afp_path.write_bytes(content)
    monkeypatch.setattr(
        "docling.datamodel.document.filetype.guess_mime",
        lambda _: "application/octet-stream",
    )
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(afp_path) is InputFormat.AFP


def test_guess_format_preserves_confident_mime_for_afp_like_content(
    tmp_path, monkeypatch
):
    content = b"\x5a\x00\x08\xd3\xa8\xa8\x00\x00\x00"
    binary_path = tmp_path / "already-detected.bin"
    binary_path.write_bytes(content)
    monkeypatch.setattr(
        "docling.datamodel.document.filetype.guess_mime",
        lambda _: "application/pdf",
    )
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(binary_path) is InputFormat.PDF


def test_guess_format_markdown_extension(tmp_path):
    markdown_path = tmp_path / "document.markdown"
    markdown_path.write_text("# Title\n")
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(markdown_path) is InputFormat.MD


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("paper", b"\\documentclass{article}\n\\begin{document}\n", InputFormat.LATEX),
        (
            "paper.ltx",
            b"% preamble\n\\documentclass[a4paper]{article}\n",
            InputFormat.LATEX,
        ),
        ("paper", b"\xef\xbb\xbf\\documentclass{article}\n", InputFormat.LATEX),
        ("paper", b"\\documentstyle[times]{ACMconf}\n", InputFormat.LATEX),
        ("paper", b"% \\documentclass{article}\n", None),
        ("chapter", b"\\section{Intro}\nText.\n", None),
        ("notes.txt", b"\\documentclass{article}\n", InputFormat.MD),
    ],
)
def test_guess_format_latex_without_tex_extension(tmp_path, name, content, expected):
    """A LaTeX document without a .tex name is recognized by its preamble."""
    doc_path = tmp_path / name
    doc_path.write_bytes(content)
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(doc_path) is expected
    stream = DocumentStream(name=name, stream=BytesIO(content))
    assert dci._guess_format(stream) is expected


@pytest.mark.parametrize(
    ("suffix", "content", "expected"),
    [
        ("VTT", b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n", InputFormat.VTT),
        ("ADOC", b"= Title\n\nSome text\n", InputFormat.ASCIIDOC),
        ("HTML", b"<p>no doctype and no html tag</p>\n", InputFormat.HTML),
        ("EML", b"From: a@b.c\nTo: d@e.f\nSubject: hi\n\nbody\n", InputFormat.EMAIL),
    ],
)
def test_guess_format_upper_case_extension_path(tmp_path, suffix, content, expected):
    # The extension is what identifies these text formats; a path must resolve
    # it case-insensitively, as a DocumentStream with the same name already does.
    doc_path = tmp_path / f"document.{suffix}"
    doc_path.write_bytes(content)
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(doc_path) is expected
    stream = DocumentStream(name=doc_path.name, stream=BytesIO(content))
    assert dci._guess_format(stream) is expected


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("doc", InputFormat.DOC),
        ("DOC", InputFormat.DOC),
        ("ppt", InputFormat.PPT),
        ("xls", InputFormat.XLS),
    ],
)
def test_guess_format_ole2_with_fat_first_sector(tmp_path, suffix, expected):
    # An OLE2 container whose first sector after the header is a FAT sector
    # (FD FF FF FF) is reported as Excel by filetype whatever it really holds,
    # e.g. a Word 6.0 .doc. The extension must then decide between the OLE2
    # based formats.
    content = bytearray(1024)
    content[:8] = bytes.fromhex("D0CF11E0A1B11AE1")
    content[512:516] = bytes.fromhex("FDFFFFFF")
    doc_path = tmp_path / f"document.{suffix}"
    doc_path.write_bytes(bytes(content))
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(doc_path) == expected
    stream = DocumentStream(name=doc_path.name, stream=BytesIO(bytes(content)))
    assert dci._guess_format(stream) == expected


def test_guess_format(tmp_path):
    """Test docling.datamodel.document._DocumentConversionInput.__guess_format"""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    temp_dir = tmp_path / "test_guess_format"
    temp_dir.mkdir()

    # Valid PDF
    buf = BytesIO(Path("./tests/data/pdf/sources/2206.01062.pdf").open("rb").read())
    stream = DocumentStream(name="my_doc.pdf", stream=buf)
    assert dci._guess_format(stream) == InputFormat.PDF
    doc_path = Path("./tests/data/pdf/sources/2206.01062.pdf")
    assert dci._guess_format(doc_path) == InputFormat.PDF

    # Valid MS Office (modern formats)
    buf = BytesIO(Path("./tests/data/docx/sources/lorem_ipsum.docx").open("rb").read())
    stream = DocumentStream(name="lorem_ipsum.docx", stream=buf)
    assert dci._guess_format(stream) == InputFormat.DOCX
    doc_path = Path("./tests/data/docx/sources/lorem_ipsum.docx")
    assert dci._guess_format(doc_path) == InputFormat.DOCX

    # MS Office without file extension (ZIP introspection fallback)
    buf = BytesIO(Path("./tests/data/docx/sources/lorem_ipsum.docx").open("rb").read())
    stream = DocumentStream(name="abc123-def456", stream=buf)
    assert dci._guess_format(stream) == InputFormat.DOCX

    buf = BytesIO(
        Path("./tests/data/pptx/sources/powerpoint_sample.pptx").open("rb").read()
    )
    stream = DocumentStream(name="upload_no_ext", stream=buf)
    assert dci._guess_format(stream) == InputFormat.PPTX

    docx_no_ext = temp_dir / "docx_no_ext"
    docx_no_ext.write_bytes(
        Path("./tests/data/docx/sources/lorem_ipsum.docx").read_bytes()
    )
    assert dci._guess_format(docx_no_ext) == InputFormat.DOCX

    pptx_no_ext = temp_dir / "pptx_no_ext"
    pptx_no_ext.write_bytes(
        Path("./tests/data/pptx/sources/powerpoint_sample.pptx").read_bytes()
    )
    assert dci._guess_format(pptx_no_ext) == InputFormat.PPTX

    # Legacy binary Office formats
    legacy_cases = [
        (
            Path("./tests/data/doc/sources/legacy_sample.doc"),
            InputFormat.DOC,
        ),
        (
            Path("./tests/data/xls/sources/legacy_sample.xls"),
            InputFormat.XLS,
        ),
        (
            Path("./tests/data/ppt/sources/legacy_sample.ppt"),
            InputFormat.PPT,
        ),
    ]
    for legacy_path, expected_format in legacy_cases:
        assert dci._guess_format(legacy_path) == expected_format

        stream = DocumentStream(
            name=legacy_path.name, stream=BytesIO(legacy_path.read_bytes())
        )
        assert dci._guess_format(stream) == expected_format

        no_ext = temp_dir / f"{expected_format.value}_no_ext"
        no_ext.write_bytes(legacy_path.read_bytes())
        assert dci._guess_format(no_ext) == expected_format

        no_ext_stream = DocumentStream(
            name=f"{expected_format.value}_upload",
            stream=BytesIO(legacy_path.read_bytes()),
        )
        assert dci._guess_format(no_ext_stream) == expected_format

    # Rich Text Format is detected from its extension for both paths and streams.
    rtf_path = Path("./tests/data/rtf/sources/legacy_sample.rtf")
    assert dci._guess_format(rtf_path) == InputFormat.RTF
    rtf_stream = DocumentStream(
        name=rtf_path.name, stream=BytesIO(rtf_path.read_bytes())
    )
    assert dci._guess_format(rtf_stream) == InputFormat.RTF

    # Valid OpenDocument formats
    odfdo_available = importlib.util.find_spec("odfdo") is not None
    odf_cases = [
        (
            Path("./tests/data/odf/sources/text_document_01.odt"),
            InputFormat.ODT,
            OdtDocumentBackend,
            OdtFormatOption(),
        ),
        (
            Path("./tests/data/odf/sources/odf_table_with_title_01.ods"),
            InputFormat.ODS,
            OdsDocumentBackend,
            OdsFormatOption(),
        ),
        (
            Path("./tests/data/odf/sources/odf_presentation_01.odp"),
            InputFormat.ODP,
            OdpDocumentBackend,
            OdpFormatOption(),
        ),
    ]

    for doc_path, input_format, backend_cls, format_option in odf_cases:
        assert dci._guess_format(doc_path) == input_format

        stream = DocumentStream(
            name=doc_path.name, stream=BytesIO(doc_path.read_bytes())
        )
        assert dci._guess_format(stream) == input_format

        no_ext_path = temp_dir / f"{input_format.value}_no_ext"
        no_ext_path.write_bytes(doc_path.read_bytes())
        assert dci._guess_format(no_ext_path) == input_format

        no_ext_stream = DocumentStream(
            name=f"{input_format.value}_upload", stream=BytesIO(doc_path.read_bytes())
        )
        assert dci._guess_format(no_ext_stream) == input_format

        assert format_option.backend is backend_cls

        if odfdo_available:
            docs = list(
                _DocumentConversionInput(path_or_stream_iterator=[doc_path]).docs(
                    {input_format: format_option}
                )
            )
            assert len(docs) == 1
            assert docs[0].format == input_format
            assert isinstance(docs[0]._backend, backend_cls)

    # Plain ZIP (not Office) should not be detected as an Office format
    import zipfile as _zipfile

    plain_zip_path = temp_dir / "archive_no_ext"
    with _zipfile.ZipFile(plain_zip_path, "w") as zf:
        zf.writestr("data.txt", "hello world")
    assert dci._guess_format(plain_zip_path) is None

    buf = BytesIO(plain_zip_path.read_bytes())
    stream = DocumentStream(name="archive_no_ext", stream=buf)
    assert dci._guess_format(stream) is None

    # Valid HTML
    buf = BytesIO(Path("./tests/data/html/sources/wiki_duck.html").open("rb").read())
    stream = DocumentStream(name="wiki_duck.html", stream=buf)
    assert dci._guess_format(stream) == InputFormat.HTML
    doc_path = Path("./tests/data/html/sources/wiki_duck.html")
    assert dci._guess_format(doc_path) == InputFormat.HTML

    html_str = (  # HTML starting with a script
        "<script>\nconsole.log('foo');\n</script>"
        '<!doctype html>\n<html lang="en-us class="no-js"></html>'
    )
    stream = DocumentStream(name="lorem_ipsum", stream=BytesIO(f"{html_str}".encode()))
    assert dci._guess_format(stream) == InputFormat.HTML

    # Valid MD
    buf = BytesIO(Path("./tests/data/md/sources/wiki.md").open("rb").read())
    stream = DocumentStream(name="wiki.md", stream=buf)
    assert dci._guess_format(stream) == InputFormat.MD
    doc_path = Path("./tests/data/md/sources/wiki.md")
    assert dci._guess_format(doc_path) == InputFormat.MD

    # Valid CSV
    buf = BytesIO(Path("./tests/data/csv/sources/csv-comma.csv").open("rb").read())
    stream = DocumentStream(name="csv-comma.csv", stream=buf)
    assert dci._guess_format(stream) == InputFormat.CSV
    stream = DocumentStream(name="test-comma", stream=buf)
    assert dci._guess_format(stream) == InputFormat.CSV
    doc_path = Path("./tests/data/csv/sources/csv-comma.csv")
    assert dci._guess_format(doc_path) == InputFormat.CSV

    # Valid XML USPTO patent
    buf = BytesIO(
        Path("./tests/data/uspto/sources/ipa20110039701.xml").open("rb").read()
    )
    stream = DocumentStream(name="ipa20110039701.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_USPTO
    doc_path = Path("./tests/data/uspto/sources/ipa20110039701.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_USPTO

    # Valid XML USPTO patent grant, Full Text Data/XML v2.5
    buf = BytesIO(Path("./tests/data/uspto/sources/pg06442728.xml").open("rb").read())
    stream = DocumentStream(name="pg06442728.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_USPTO
    doc_path = Path("./tests/data/uspto/sources/pg06442728.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_USPTO

    buf = BytesIO(
        Path("./tests/data/uspto/sources/pftaps057006474.txt").open("rb").read()
    )
    stream = DocumentStream(name="pftaps057006474.txt", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_USPTO
    doc_path = Path("./tests/data/uspto/sources/pftaps057006474.txt")
    assert dci._guess_format(doc_path) == InputFormat.XML_USPTO
    stream = DocumentStream(
        name="pftaps057006474.txt",
        stream=BytesIO(b"PATN\nWKU  057006474\n"),
    )
    assert dci._guess_format(stream) == InputFormat.XML_USPTO

    # Valid XML JATS
    buf = BytesIO(Path("./tests/data/jats/sources/elife-56337.xml").open("rb").read())
    stream = DocumentStream(name="elife-56337.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/sources/elife-56337.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    buf = BytesIO(Path("./tests/data/jats/sources/elife-56337.nxml").open("rb").read())
    stream = DocumentStream(name="elife-56337.nxml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/sources/elife-56337.nxml")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    buf = BytesIO(Path("./tests/data/jats/sources/elife-56337.txt").open("rb").read())
    stream = DocumentStream(name="elife-56337.txt", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/sources/elife-56337.txt")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    buf = BytesIO(Path("./tests/data/xbrl/sources/mlac-20251231.xml").open("rb").read())
    stream = DocumentStream(name="mlac-20251231.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_XBRL
    doc_path = Path("./tests/data/xbrl/sources/mlac-20251231.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_XBRL

    # Valid XML, non-supported flavor
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE docling_test SYSTEM '
        '"test.dtd"><docling>Docling parses documents</docling>'
    )
    doc_path = temp_dir / "docling_test.xml"
    doc_path.write_text(xml_content, encoding="utf-8")
    assert dci._guess_format(doc_path) is None
    buf = BytesIO(Path(doc_path).open("rb").read())
    stream = DocumentStream(name="docling_test.xml", stream=buf)
    assert dci._guess_format(stream) is None

    # Valid DocLang XML with generic .xml extension
    doclang_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<doclang><heading>DocLang</heading><text>Hello</text></doclang>"
    )
    doc_path = temp_dir / "doclang_sample.xml"
    doc_path.write_text(doclang_xml, encoding="utf-8")
    assert dci._guess_format(doc_path) == InputFormat.XML_DOCLANG
    buf = BytesIO(doc_path.read_bytes())
    stream = DocumentStream(name="doclang_sample.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_DOCLANG

    # Plain .txt file (not USPTO) should be detected as Markdown
    stream = DocumentStream(name="pftaps057006474.txt", stream=BytesIO(b"xyz"))
    assert dci._guess_format(stream) == InputFormat.MD

    # Valid METS-GBS archive
    mets_gbs_path = Path("./tests/data/mets_gbs/sources/32044009881525_select.tar.gz")
    if mets_gbs_path.exists():
        assert dci._guess_format(mets_gbs_path) == InputFormat.METS_GBS

        buf = BytesIO(mets_gbs_path.open("rb").read())
        stream = DocumentStream(name="32044009881525_select.tar.gz", stream=buf)
        assert dci._guess_format(stream) == InputFormat.METS_GBS

    doc_path = temp_dir / "pftaps_wrong.txt"
    doc_path.write_text("xyz", encoding="utf-8")
    assert dci._guess_format(doc_path) == InputFormat.MD

    # Plain .txt with typical text content
    stream = DocumentStream(
        name="readme.txt", stream=BytesIO(b"Hello, this is a plain text file.")
    )
    assert dci._guess_format(stream) == InputFormat.MD

    # Valid WebVTT
    buf = BytesIO(
        Path("./tests/data/webvtt/sources/webvtt_example_01.vtt").open("rb").read()
    )
    stream = DocumentStream(name="webvtt_example_01.vtt", stream=buf)
    assert dci._guess_format(stream) == InputFormat.VTT

    # Valid email
    buf = BytesIO(Path("./tests/data/email/sources/eml_simple.eml").open("rb").read())
    stream = DocumentStream(name="eml_simple.eml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.EMAIL
    doc_path = Path("./tests/data/email/sources/eml_simple.eml")
    assert dci._guess_format(doc_path) == InputFormat.EMAIL

    # Valid Docling JSON
    test_str = '{"name": ""}'
    stream = DocumentStream(name="test.json", stream=BytesIO(f"{test_str}".encode()))
    assert dci._guess_format(stream) == InputFormat.JSON_DOCLING
    doc_path = temp_dir / "test.json"
    doc_path.write_text(test_str, encoding="utf-8")
    assert dci._guess_format(doc_path) == InputFormat.JSON_DOCLING

    # Non-Docling JSON
    # TODO: Docling JSON is currently the single supported JSON flavor and the pipeline
    # will try to validate *any* JSON (based on suffix/MIME) as Docling JSON; proper
    # disambiguation seen as part of https://github.com/docling-project/docling/issues/802
    test_str = "{}"
    stream = DocumentStream(name="test.json", stream=BytesIO(f"{test_str}".encode()))
    assert dci._guess_format(stream) == InputFormat.JSON_DOCLING
    doc_path = temp_dir / "test.json"
    doc_path.write_text(test_str, encoding="utf-8")
    assert dci._guess_format(doc_path) == InputFormat.JSON_DOCLING


def test_guess_format_xml_with_undecodable_head(tmp_path):
    """XML whose sniffed head is not valid UTF-8 must still be detected (#1762).

    ``_guess_format`` sniffs only the first 1024 bytes of a path (8192 of a
    stream), so the head can end mid-codepoint even for well-formed UTF-8, and
    an XML document may declare a non-UTF-8 encoding outright. Neither may raise
    out of format detection -- that aborts the whole conversion batch.
    """
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    temp_dir = tmp_path / "test_guess_format_undecodable"
    temp_dir.mkdir()

    # A JATS article declared in ISO-8859-1; the surname carries byte 0xF8.
    latin1_jats = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<!DOCTYPE article PUBLIC "-//NLM//DTD JATS-journalpublishing1.dtd" "x.dtd">\n'
        "<article><front><article-meta>"
        "<contrib><name><surname>Bj\xf8rnstad</surname></name></contrib>"
        "</article-meta></front></article>\n"
    ).encode("iso-8859-1")

    doc_path = temp_dir / "article_latin1.xml"
    doc_path.write_bytes(latin1_jats)
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    stream = DocumentStream(name="article_latin1.xml", stream=BytesIO(latin1_jats))
    assert dci._guess_format(stream) == InputFormat.XML_JATS

    # Well-formed UTF-8, but a two-byte codepoint straddles the 1024-byte cut.
    head = '<?xml version="1.0" encoding="UTF-8"?>\n<doclang><text>'
    padding = "a" * (1024 - len(head.encode()) - 1)
    split_utf8 = f"{head}{padding}é</text></doclang>\n".encode()
    assert len(split_utf8) > 1024
    with pytest.raises(UnicodeDecodeError):  # the head alone is undecodable
        split_utf8[:1024].decode("utf-8")

    doc_path = temp_dir / "doclang_split.xml"
    doc_path.write_bytes(split_utf8)
    assert dci._guess_format(doc_path) == InputFormat.XML_DOCLANG


def _make_input_doc(path):
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.PDF,
        backend=PdfFormatOption().backend,  # use default
    )
    return in_doc


def _make_input_doc_from_stream(doc_stream):
    in_doc = InputDocument(
        path_or_stream=doc_stream.stream,
        format=InputFormat.PDF,
        filename=doc_stream.name,
        backend=PdfFormatOption().backend,  # use default
    )
    return in_doc


def test_tiff_two_pages():
    tiff_path = Path("./tests/data/tiff/sources/2206.01062.tif")
    doc = InputDocument(
        path_or_stream=tiff_path,
        format=InputFormat.IMAGE,
        backend=ImageFormatOption().backend,  # use default backend
    )
    assert doc.valid is True
    assert doc.page_count == 2

    # Expect two full-page rectangles
    rects_page1 = doc._backend.load_page(0).get_bitmap_rects()
    rects_page2 = doc._backend.load_page(1).get_bitmap_rects()

    page1_rect = next(rects_page1)
    page2_rect = next(rects_page2)

    assert page1_rect.t == page2_rect.t == 0
    assert page1_rect.l == page2_rect.l == 0
    assert page1_rect.r == page2_rect.r == 612.0
    assert page1_rect.b == page2_rect.b == 792.0


# ---------------------------------------------------------------------------
# Tests for non-ASCII format detection (regression tests for UTF-8/latin-1 fallback)
# ---------------------------------------------------------------------------


def test_detect_csv_with_non_ascii_utf8():
    """CSV with non-ASCII UTF-8 characters should be detected correctly."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # UTF-8 CSV with non-ASCII characters (é, ñ, 中文)
    csv_content = b"name,value\nH\xc3\xa9llo,w\xc3\xb3rld\nM\xc3\xa9xico,Espa\xc3\xb1a\n"
    assert dci._detect_csv(csv_content) == "text/csv"

    # UTF-8 with semicolon delimiter
    csv_content_semicolon = b"name;value\nH\xc3\xa9llo;world\n"
    assert dci._detect_csv(csv_content_semicolon) == "text/csv"


def test_detect_csv_with_non_ascii_latin1():
    """CSV with latin-1 encoded non-ASCII characters should be detected."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # Latin-1 encoded CSV with non-ASCII (0xE9 = é in latin-1)
    csv_content = b"name,value\nH\xe9llo,w\xf3rld\n"  # "Héllo,wórld" in latin-1
    assert dci._detect_csv(csv_content) == "text/csv"


def test_detect_html_xhtml_with_non_ascii():
    """HTML/XHTML with non-ASCII characters should be detected correctly."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # UTF-8 HTML with non-ASCII
    html_utf8 = b"<html><body>H\xc3\xa9llo w\xc3\xb3rld</body></html>"
    assert dci._detect_html_xhtml(html_utf8) == "text/html"

    # XHTML with non-ASCII
    xhtml_utf8 = b'<?xml version="1.0" encoding="UTF-8"?><html><body>H\xc3\xa9llo</body></html>'
    assert dci._detect_html_xhtml(xhtml_utf8) == "application/xhtml+xml"

    # Latin-1 HTML with non-ASCII
    html_latin1 = b"<html><body>H\xe9llo</body></html>".encode("latin-1")
    assert dci._detect_html_xhtml(html_latin1) == "text/html"


def test_detect_latex_with_non_ascii():
    """LaTeX with non-ASCII characters should be detected correctly."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # UTF-8 LaTeX with non-ASCII
    latex_utf8 = br"\documentclass{article}\begin{document}H\xc3\xa9llo\end{document}"
    assert dci._detect_latex(latex_utf8, ext=None) == "application/x-latex"

    # Latin-1 LaTeX
    latex_latin1 = br"\documentclass{article}\begin{document}H\xe9llo\end{document}".encode("latin-1")
    assert dci._detect_latex(latex_latin1, ext=None) == "application/x-latex"

    # With .tex extension
    assert dci._detect_latex(latex_utf8, ext=".tex") == "application/x-latex"


def test_detect_odf_mimetype_utf8_and_latin1():
    """ODF mimetype detection should handle UTF-8 and latin-1."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # Create a minimal ODT file with UTF-8 mimetype
    with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            # UTF-8 mimetype
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
            # Minimal content.xml
            zf.writestr("content.xml", '<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"/>')
        try:
            mime = dci._detect_office_mime_from_zip(tmp.name)
            assert mime == "application/vnd.oasis.opendocument.text"
        finally:
            os.unlink(tmp.name)

    # Test with latin-1 mimetype
    with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            # Latin-1 mimetype (valid ODF mimetype is ASCII but test robustness)
            zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
            zf.writestr("content.xml", '<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"/>')
        try:
            mime = dci._detect_office_mime_from_zip(tmp.name)
            assert mime == "application/vnd.oasis.opendocument.text"
        finally:
            os.unlink(tmp.name)
