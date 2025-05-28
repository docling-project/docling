from io import BytesIO
from pathlib import Path

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.datamodel.settings import DocumentLimits
from docling.document_converter import PdfFormatOption


def test_in_doc_from_valid_path():
    test_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    doc = _make_input_doc(test_doc_path)
    assert doc.valid is True


def test_in_doc_from_invalid_path():
    test_doc_path = Path("./tests/does/not/exist.pdf")

    doc = _make_input_doc(test_doc_path)

    assert doc.valid is False


def test_in_doc_from_valid_buf():
    buf = BytesIO(Path("./tests/data/pdf/2206.01062.pdf").open("rb").read())
    stream = DocumentStream(name="my_doc.pdf", stream=buf)

    doc = _make_input_doc_from_stream(stream)
    assert doc.valid is True


def test_in_doc_from_invalid_buf():
    buf = BytesIO(b"")
    stream = DocumentStream(name="my_doc.pdf", stream=buf)

    doc = _make_input_doc_from_stream(stream)
    assert doc.valid is False


def test_image_in_pdf_backend():
    in_doc = InputDocument(
        path_or_stream=Path("tests/data/2305.03393v1-pg9-img.png"),
        format=InputFormat.IMAGE,
        backend=PyPdfiumDocumentBackend,
    )

    assert in_doc.valid
    in_doc = InputDocument(
        path_or_stream=Path("tests/data/2305.03393v1-pg9-img.png"),
        format=InputFormat.IMAGE,
        backend=DoclingParseDocumentBackend,
    )
    assert in_doc.valid

    in_doc = InputDocument(
        path_or_stream=Path("tests/data/2305.03393v1-pg9-img.png"),
        format=InputFormat.IMAGE,
        backend=DoclingParseV2DocumentBackend,
    )
    assert in_doc.valid

    in_doc = InputDocument(
        path_or_stream=Path("tests/data/2305.03393v1-pg9-img.png"),
        format=InputFormat.IMAGE,
        backend=DoclingParseV4DocumentBackend,
    )
    assert in_doc.valid


def test_in_doc_with_page_range():
    test_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
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


def test_guess_format(tmp_path):
    """Test docling.datamodel.document._DocumentConversionInput.__guess_format"""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    temp_dir = tmp_path / "test_guess_format"
    temp_dir.mkdir()

    # Valid PDF
    buf = BytesIO(Path("./tests/data/pdf/2206.01062.pdf").open("rb").read())
    stream = DocumentStream(name="my_doc.pdf", stream=buf)
    assert dci._guess_format(stream) == InputFormat.PDF
    doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    assert dci._guess_format(doc_path) == InputFormat.PDF

    # Valid MS Office
    buf = BytesIO(Path("./tests/data/docx/lorem_ipsum.docx").open("rb").read())
    stream = DocumentStream(name="lorem_ipsum.docx", stream=buf)
    assert dci._guess_format(stream) == InputFormat.DOCX
    doc_path = Path("./tests/data/docx/lorem_ipsum.docx")
    assert dci._guess_format(doc_path) == InputFormat.DOCX

    # Valid HTML
    buf = BytesIO(Path("./tests/data/html/wiki_duck.html").open("rb").read())
    stream = DocumentStream(name="wiki_duck.html", stream=buf)
    assert dci._guess_format(stream) == InputFormat.HTML
    doc_path = Path("./tests/data/html/wiki_duck.html")
    assert dci._guess_format(doc_path) == InputFormat.HTML

    html_str = (  # HTML starting with a script
        "<script>\nconsole.log('foo');\n</script>"
        '<!doctype html>\n<html lang="en-us class="no-js"></html>'
    )
    stream = DocumentStream(name="lorem_ipsum", stream=BytesIO(f"{html_str}".encode()))
    assert dci._guess_format(stream) == InputFormat.HTML

    # Valid MD
    buf = BytesIO(Path("./tests/data/md/wiki.md").open("rb").read())
    stream = DocumentStream(name="wiki.md", stream=buf)
    assert dci._guess_format(stream) == InputFormat.MD
    doc_path = Path("./tests/data/md/wiki.md")
    assert dci._guess_format(doc_path) == InputFormat.MD

    # Valid CSV
    buf = BytesIO(Path("./tests/data/csv/csv-comma.csv").open("rb").read())
    stream = DocumentStream(name="csv-comma.csv", stream=buf)
    assert dci._guess_format(stream) == InputFormat.CSV
    stream = DocumentStream(name="test-comma", stream=buf)
    assert dci._guess_format(stream) == InputFormat.CSV
    doc_path = Path("./tests/data/csv/csv-comma.csv")
    assert dci._guess_format(doc_path) == InputFormat.CSV

    # Valid XML USPTO patent
    buf = BytesIO(Path("./tests/data/uspto/ipa20110039701.xml").open("rb").read())
    stream = DocumentStream(name="ipa20110039701.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_USPTO
    doc_path = Path("./tests/data/uspto/ipa20110039701.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_USPTO

    buf = BytesIO(Path("./tests/data/uspto/pftaps057006474.txt").open("rb").read())
    stream = DocumentStream(name="pftaps057006474.txt", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_USPTO
    doc_path = Path("./tests/data/uspto/pftaps057006474.txt")
    assert dci._guess_format(doc_path) == InputFormat.XML_USPTO

    # Valid XML JATS
    buf = BytesIO(Path("./tests/data/jats/elife-56337.xml").open("rb").read())
    stream = DocumentStream(name="elife-56337.xml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/elife-56337.xml")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    buf = BytesIO(Path("./tests/data/jats/elife-56337.nxml").open("rb").read())
    stream = DocumentStream(name="elife-56337.nxml", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/elife-56337.nxml")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

    buf = BytesIO(Path("./tests/data/jats/elife-56337.txt").open("rb").read())
    stream = DocumentStream(name="elife-56337.txt", stream=buf)
    assert dci._guess_format(stream) == InputFormat.XML_JATS
    doc_path = Path("./tests/data/jats/elife-56337.txt")
    assert dci._guess_format(doc_path) == InputFormat.XML_JATS

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

    # Invalid USPTO patent (as plain text)
    stream = DocumentStream(name="pftaps057006474.txt", stream=BytesIO(b"xyz"))
    assert dci._guess_format(stream) is None
    doc_path = temp_dir / "pftaps_wrong.txt"
    doc_path.write_text("xyz", encoding="utf-8")
    assert dci._guess_format(doc_path) is None

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
