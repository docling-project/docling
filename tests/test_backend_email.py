# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from email.utils import getaddresses
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DocItemLabel, TextItem

from docling.backend.email_backend import EmailDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter, EmailFormatOption


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


def test_email_backend_quotes_display_names_holding_specials():
    raw_email = b"""From: "Doe, John" <john@example.com>
To: "Smith, Jane" <jane@example.com>, "Roe, Richard" <rich@example.com>
Subject: Quarterly Report
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Numbers attached.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="specials.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]
    to_line = next(text for text in rendered if text.startswith("To: "))

    assert 'From: "Doe, John" <john@example.com>' in rendered
    # Names and addresses both survive the round trip; unquoted, each comma
    # inside a name reads as another recipient.
    assert getaddresses([to_line.removeprefix("To: ")]) == [
        ("Smith, Jane", "jane@example.com"),
        ("Roe, Richard", "rich@example.com"),
    ]


def test_email_backend_leaves_plain_display_names_unquoted():
    # "." and "'" parse back intact unquoted, and are common enough in real
    # names that quoting them would only add noise to a document meant to be
    # read.
    raw_email = b"""From: John A. Smith <john@example.com>
To: O'Brien <obrien@example.com>
Subject: Plain Names
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Body.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="plain_names.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]

    assert "From: John A. Smith <john@example.com>" in rendered
    assert "To: O'Brien <obrien@example.com>" in rendered


def test_email_backend_escapes_quotes_inside_display_names():
    raw_email = b"""From: "John \\"JD\\" Doe" <jd@example.com>
To: bob@example.com
Subject: Embedded Quotes
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Body.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="embedded_quotes.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]
    from_line = next(text for text in rendered if text.startswith("From: "))

    # Without escaping, the inner quotes close the quoted-string early and the
    # name parses back mangled.
    assert getaddresses([from_line.removeprefix("From: ")]) == [
        ('John "JD" Doe', "jd@example.com")
    ]


def test_email_backend_collapses_crlf_in_headers():
    # An RFC 2047 encoded-word decodes to arbitrary text, CR/LF included, so a
    # crafted name would otherwise spread across lines and forge headers of its
    # own in the rendered document.
    raw_email = b"""From: =?utf-8?q?Attacker=0D=0ADate=3A_1999-01-01=0D=0ATo=3A_ceo=40corp?= <evil@example.com>
To: real@example.com
Subject: =?utf-8?q?Hi=0D=0AFrom=3A_boss=40corp?=
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Body.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="crlf.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]

    assert not any("\r" in text or "\n" in text for text in rendered)
    assert (
        'From: "Attacker  Date: 1999-01-01  To: ceo@corp" <evil@example.com>'
        in rendered
    )
    assert "Hi  From: boss@corp" in rendered


def test_email_backend_keeps_non_ascii_display_names_readable():
    raw_email = b"""From: =?utf-8?b?5byg5LiJ?= <zhang@example.com>
To: =?utf-8?q?M=C3=BCller=2C_Anna?= <anna@example.com>
Subject: Non-ASCII Names
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Body.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="non_ascii.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]

    # A name is quoted when it holds specials but is never re-encoded: the
    # document is read by people, not parsed off the wire. Swapping in
    # formataddr or headerregistry.Address would put "=?utf-8?" back here.
    assert "From: \u5f20\u4e09 <zhang@example.com>" in rendered
    assert 'To: "M\xfcller, Anna" <anna@example.com>' in rendered


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


def test_convert_msg_backend_from_path():
    in_path = Path("tests/data/email/sources/msg_simple.msg")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    assert backend.is_valid()
    assert backend.is_msg

    markdown = backend.convert().export_to_markdown()

    assert "Simple Email" in markdown
    assert "From: Alice Example &lt;alice@example.com&gt;" in markdown
    assert "To: Bob Example &lt;bob@example.com&gt;" in markdown
    assert "Hello Bob," in markdown
    assert "This is a simple email body." in markdown


def test_convert_msg_backend_from_stream():
    raw_msg = Path("tests/data/email/sources/msg_simple.msg").read_bytes()
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_msg),
        format=InputFormat.EMAIL,
        filename="msg_simple.msg",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_msg),
    )

    assert backend.is_valid()
    assert backend.is_msg
    assert "Simple Email" in backend.convert().export_to_markdown()


def test_msg_document_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.EMAIL])
    result = converter.convert(Path("tests/data/email/sources/msg_simple.msg"))

    markdown = result.document.export_to_markdown()
    assert "Simple Email" in markdown
    assert "This is a simple email body." in markdown


def test_msg_with_attachment_excludes_content_and_names_by_default():
    in_path = Path("tests/data/email/sources/msg_with_attachment.msg")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    markdown = backend.convert().export_to_markdown()

    assert "Email with Attachment" in markdown
    assert "This email contains an attachment." in markdown

    # Decoded attachment content is never embedded.
    assert "This is a test attachment file." not in markdown
    # Attachment names are only listed when the option is enabled.
    assert "test.txt" not in markdown
    assert "report.pdf" not in markdown
    assert "Attachments" not in markdown


def test_email_backend_lists_attachments_when_enabled():
    in_path = Path("tests/data/email/sources/eml_with_attachment.eml")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
        options=EmailBackendOptions(list_attachments=True),
    )

    markdown = backend.convert().export_to_markdown()

    assert "Attachments" in markdown
    assert "test.txt" in markdown
    # Listing names must not pull in the encoded or decoded payload.
    assert "This is a test attachment file." not in markdown
    assert (
        "VGhpcyBpcyBhIHRlc3QgYXR0YWNobWVudCBmaWxlLgpJdCBjb250YWlucyBzb21lIGR1bW15IGNv"
        not in markdown
    )


def test_msg_backend_lists_attachments_when_enabled():
    in_path = Path("tests/data/email/sources/msg_with_attachment.msg")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
        options=EmailBackendOptions(list_attachments=True),
    )

    markdown = backend.convert().export_to_markdown()

    assert "Attachments" in markdown
    assert "test.txt (text/plain)" in markdown
    assert "report.pdf (application/pdf)" in markdown
    assert "This is a test attachment file." not in markdown


def test_msg_document_converter_lists_attachments_via_format_option():
    converter = DocumentConverter(
        allowed_formats=[InputFormat.EMAIL],
        format_options={
            InputFormat.EMAIL: EmailFormatOption(
                backend_options=EmailBackendOptions(list_attachments=True)
            )
        },
    )
    result = converter.convert(Path("tests/data/email/sources/msg_with_attachment.msg"))

    markdown = result.document.export_to_markdown()
    assert "Attachments" in markdown
    assert "test.txt" in markdown
    assert "report.pdf" in markdown
