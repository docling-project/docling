from io import BytesIO
from pathlib import Path

from docling.backend.email_backend import EmailDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter


def test_convert_email_backend_from_path():
    in_path = Path("tests/data/email/simple.eml")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    assert backend.is_valid()

    doc = backend.convert()
    markdown = doc.export_to_markdown()

    assert "Simple Email" in markdown
    assert "From: Alice Example &lt;alice@example.com&gt;" in markdown
    assert "To: Bob Example &lt;bob@example.com&gt;" in markdown
    assert "Hello Bob," in markdown
    assert "This is a simple email body." in markdown
    assert backend._extract_attachments() == []


def test_convert_email_backend_from_stream():
    raw_email = Path("tests/data/email/simple.eml").read_bytes()
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="simple.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_email),
    )

    assert backend.is_valid()
    assert "Simple Email" in backend.convert().export_to_markdown()


def test_email_document_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.EMAIL])
    result = converter.convert(Path("tests/data/email/simple.eml"))

    markdown = result.document.export_to_markdown()
    assert "Simple Email" in markdown
    assert "This is a simple email body." in markdown
