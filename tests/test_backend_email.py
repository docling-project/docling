from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DocItemLabel, TextItem

from docling.backend.email_backend import _OLE2_MAGIC, EmailDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter


def test_convert_email_backend_from_path():
    in_path = Path("tests/data/email/sources/eml_simple.eml")
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


def test_convert_email_backend_from_stream():
    raw_email = Path("tests/data/email/sources/eml_simple.eml").read_bytes()
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="eml_simple.eml",
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
    result = converter.convert(Path("tests/data/email/sources/eml_simple.eml"))

    markdown = result.document.export_to_markdown()
    assert "Simple Email" in markdown
    assert "This is a simple email body." in markdown


def test_email_with_attachment_excludes_encoded_content():
    """Test that base64-encoded attachment content is not included in the converted document."""
    in_path = Path("tests/data/email/sources/eml_with_attachment.eml")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    assert backend.is_valid()

    doc = backend.convert()
    markdown = doc.export_to_markdown()

    # Verify email metadata and body are present
    assert "Email with Attachment" in markdown
    assert "From: Alice Example &lt;alice@example.com&gt;" in markdown
    assert "To: Bob Example &lt;bob@example.com&gt;" in markdown
    assert "This email contains an attachment." in markdown

    # Verify base64-encoded attachment content is NOT in the document
    assert (
        "VGhpcyBpcyBhIHRlc3QgYXR0YWNobWVudCBmaWxlLgpJdCBjb250YWlucyBzb21lIGR1bW15IGNv"
        not in markdown
    )
    assert "bnRlbnQuCg==" not in markdown

    # Verify decoded attachment content is also NOT in the document
    assert "This is a test attachment file." not in markdown
    assert "It contains some dummy content." not in markdown


def test_email_attachments_listed_when_enabled():
    """With include_attachments=True, attachment metadata is surfaced (no payload)."""
    in_path = Path("tests/data/email/sources/eml_with_attachment.eml")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
        options=EmailBackendOptions(include_attachments=True),
    )

    markdown = backend.convert().export_to_markdown()

    # Body is still present, plus an attachments section listing the file.
    assert "This email contains an attachment." in markdown
    assert "Attachments" in markdown
    assert "test.txt (text/plain" in markdown

    # The attachment payload itself (encoded or decoded) is still never inlined.
    assert "bnRlbnQuCg==" not in markdown
    assert "This is a test attachment file." not in markdown


def test_email_attachments_excluded_by_default():
    """Without the opt-in flag, no attachment section is added."""
    in_path = Path("tests/data/email/sources/eml_with_attachment.eml")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    markdown = backend.convert().export_to_markdown()

    assert "This email contains an attachment." in markdown
    assert "Attachments" not in markdown
    assert "test.txt" not in markdown


def test_msg_extension_is_detected_as_email():
    """Outlook `.msg` files (OLE2 container) route to the EMAIL format by extension."""
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    # OLE2 magic alone is not recognized by `filetype`, so routing relies on the
    # newly registered `.msg` extension / `application/vnd.ms-outlook` mime type.
    stream = DocumentStream(
        name="message.msg", stream=BytesIO(_OLE2_MAGIC + b"\x00" * 512)
    )
    assert dci._guess_format(stream) == InputFormat.EMAIL


def test_email_backend_preserves_body_paragraphs_and_date():
    raw_email = b"""From: Alice Example <alice@example.com>
To: Bob Example <bob@example.com>
Subject: Paragraph Email
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Hello Bob,

This is a second paragraph.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="paragraph.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    doc = backend.convert()
    text_items = [item for item in doc.texts if isinstance(item, TextItem)]

    assert [item.text for item in text_items] == [
        "Paragraph Email",
        "From: Alice Example <alice@example.com>",
        "To: Bob Example <bob@example.com>",
        "Date: 2026-05-20T10:30:00+00:00",
        "Hello Bob,",
        "This is a second paragraph.",
    ]
    assert [item.label for item in text_items[1:]] == [DocItemLabel.TEXT] * 5


def test_email_backend_converts_html_body_to_text_paragraphs():
    raw_email = b"""From: Alice Example <alice@example.com>
To: Bob Example <bob@example.com>
Subject: HTML Email
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html><body><p>Hello <strong>Bob</strong>,</p><p>This is HTML.</p></body></html>
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="html.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    markdown = backend.convert().export_to_markdown()

    assert "Hello **Bob** ," in markdown
    assert "This is HTML." in markdown
    assert "<strong>" not in markdown
