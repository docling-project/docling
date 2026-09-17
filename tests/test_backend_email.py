# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from base64 import b64encode
from email.headerregistry import HeaderRegistry
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
    """A name holding specials is quoted so the list parses back as itself."""
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
    assert getaddresses([to_line.removeprefix("To: ")]) == [
        ("Smith, Jane", "jane@example.com"),
        ("Roe, Richard", "rich@example.com"),
    ]


def test_email_backend_leaves_plain_display_names_unquoted():
    """A period or an apostrophe parses back unquoted, so the name is left alone."""
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
    """A quote inside a name is escaped instead of closing the string early."""
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

    assert getaddresses([from_line.removeprefix("From: ")]) == [
        ('John "JD" Doe', "jd@example.com")
    ]


def test_email_backend_collapses_line_breaks_in_headers():
    """A crafted name, subject or attachment filename cannot forge a header line.

    Attachment listing is off by default, so the invariant only covers the
    filename path with it enabled.
    """
    raw_email = (
        b"From: =?utf-8?q?Attacker=0D=0ADate=3A_1999-01-01=0D=0ATo=3A_ceo=40corp?="
        b" <evil@example.com>\r\n"
        b"To: real@example.com\r\n"
        b"Subject: =?utf-8?q?Hi=0D=0AFrom=3A_boss=40corp?=\r\n"
        b"Date: Tue, 20 May 2026 10:30:00 +0000\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/mixed; boundary="BOUNDARY"\r\n'
        b"\r\n"
        b"--BOUNDARY\r\n"
        b'Content-Type: text/plain; charset="utf-8"\r\n'
        b"\r\n"
        b"Body.\r\n"
        b"--BOUNDARY\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Disposition: attachment;"
        b' filename="=?utf-8?q?evil=0D=0AFrom=3A_boss=40corp?=.txt"\r\n'
        b"\r\n"
        b"payload\r\n"
        b"--BOUNDARY--\r\n"
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="crlf.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_email),
        options=EmailBackendOptions(list_attachments=True),
    )

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]

    assert not any("\r" in text or "\n" in text for text in rendered)
    assert (
        'From: "Attacker Date: 1999-01-01 To: ceo@corp" <evil@example.com>' in rendered
    )
    assert "Hi From: boss@corp" in rendered
    assert "evil From: boss@corp.txt (text/plain)" in rendered


def test_email_backend_collapses_non_crlf_line_breaks_in_headers():
    """Every break ``str.splitlines()`` recognises is collapsed, not only CR/LF."""
    for break_char in ("\x0b", "\x0c", "\x1c", "\x85", "\u2028"):
        encoded = b64encode(f"Hi{break_char}From: boss@corp".encode()).decode()
        raw_email = (
            b"From: sender@example.com\r\n"
            b"To: real@example.com\r\n"
            b"Subject: =?utf-8?b?" + encoded.encode() + b"?=\r\n"
            b"MIME-Version: 1.0\r\n"
            b'Content-Type: text/plain; charset="utf-8"\r\n'
            b"\r\n"
            b"Body.\r\n"
        )
        in_doc = InputDocument(
            path_or_stream=BytesIO(raw_email),
            format=InputFormat.EMAIL,
            filename="breaks.eml",
            backend=EmailDocumentBackend,
        )
        backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

        title = next(
            item.text
            for item in backend.convert().texts
            if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
        )

        assert title == "Hi From: boss@corp"
        assert title.splitlines() == [title]


def test_email_backend_unfolds_folded_headers_to_single_spaces():
    """A folded header renders with one space, not a run of three."""
    raw_email = (
        b"From: sender@example.com\r\n"
        b"To: real@example.com\r\n"
        b"Subject: A very long subject that is\r\n folded across lines\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: text/plain; charset="utf-8"\r\n'
        b"\r\n"
        b"Body.\r\n"
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="folded.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    title = next(
        item.text
        for item in backend.convert().texts
        if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
    )

    assert title == "A very long subject that is folded across lines"


def test_email_backend_quotes_display_names_holding_a_backslash():
    """A Windows-style name is quoted; unquoted the parser loses the address."""
    raw_email = b"""From: =?utf-8?q?CORP=5Cjsmith?= <jsmith@example.com>
To: real@example.com
Subject: Backslash Name
Date: Tue, 20 May 2026 10:30:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Body.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_email),
        format=InputFormat.EMAIL,
        filename="backslash.eml",
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

    rendered = [
        item.text for item in backend.convert().texts if isinstance(item, TextItem)
    ]
    from_line = next(text for text in rendered if text.startswith("From: "))

    header = HeaderRegistry()("To", from_line.removeprefix("From: "))
    assert [
        (address.display_name, address.addr_spec)
        for group in header.groups
        for address in group.addresses
    ] == [("CORP\\jsmith", "jsmith@example.com")]


def test_email_backend_keeps_non_ascii_display_names_readable():
    """A non-ASCII name is quoted when needed but never RFC 2047 re-encoded."""
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


def test_email_backend_normalises_body_line_endings():
    """A CRLF or lone-CR body leaves no carriage return inside a paragraph."""
    for terminator in ("\r\n", "\r"):
        body = (
            f"First paragraph line one.{terminator}First paragraph line two."
            f"{terminator}{terminator}Second paragraph.{terminator}"
        )
        raw_email = (
            "From: alice@example.com\r\n"
            "Subject: Line Endings\r\n"
            "MIME-Version: 1.0\r\n"
            'Content-Type: text/plain; charset="utf-8"\r\n'
            "\r\n" + body
        ).encode()
        in_doc = InputDocument(
            path_or_stream=BytesIO(raw_email),
            format=InputFormat.EMAIL,
            filename="line_endings.eml",
            backend=EmailDocumentBackend,
        )
        backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw_email))

        text_items = [
            item for item in backend.convert().texts if isinstance(item, TextItem)
        ]

        assert [item.text for item in text_items] == [
            "Line Endings",
            "From: alice@example.com",
            "First paragraph line one.\nFirst paragraph line two.",
            "Second paragraph.",
        ]


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


def test_header_safe_skips_non_string_values():
    assert EmailDocumentBackend._header_safe(None) == ""
    assert EmailDocumentBackend._header_safe(123) == ""
    assert EmailDocumentBackend._header_safe("Alice Example") == "Alice Example"


def test_split_paragraphs_skips_non_string_bodies():
    backend = EmailDocumentBackend.__new__(EmailDocumentBackend)
    assert backend._split_paragraphs(None) == []
    assert backend._split_paragraphs(b"hello") == []
    assert backend._split_paragraphs("Hello\n\nWorld") == ["Hello", "World"]
