# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Outlook .msg files whose 8-bit string streams end with NUL bytes (#5535)."""

from io import BytesIO
from pathlib import Path

from docling.backend.email_backend import EmailDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter, EmailFormatOption


def test_msg_with_nul_terminated_8bit_strings():
    """8-bit string streams that end with NUL bytes do not leak into the document.

    The fixture is msg_with_attachment.msg with every PtypString property stored
    as PtypString8: the ASCII text followed by NUL bytes, message code page 1252.
    python-oxmsg keeps the NUL bytes, and puts the sender's inside the quotes and
    angle brackets of `"name" <address>`. See #5535.
    """
    from oxmsg import Message as OxMsgMessage

    in_path = Path("tests/data/email/sources/msg_8bit_nul_terminated.msg")
    assert OxMsgMessage.load(in_path.read_bytes()).subject.endswith("\x00")

    converter = DocumentConverter(
        allowed_formats=[InputFormat.EMAIL],
        format_options={
            InputFormat.EMAIL: EmailFormatOption(
                backend_options=EmailBackendOptions(list_attachments=True)
            )
        },
    )
    doc = converter.convert(in_path).document
    markdown = doc.export_to_markdown()

    assert [item.text for item in doc.texts if "\x00" in item.text] == []
    assert "\x00" not in markdown
    assert "# Email with Attachment\n" in markdown
    assert "From: Alice Example &lt;alice@example.com&gt;" in markdown
    assert "To: Bob Example &lt;bob@example.com&gt;" in markdown
    assert "This email contains an attachment." in markdown
    assert "test.txt (text/plain)" in markdown
    assert "report.pdf (application/pdf)" in markdown


def test_msg_strips_trailing_nul_from_properties_and_body():
    """Verify that trailing NUL characters (\x00) from 8-bit .msg streams are stripped.

    Non-Unicode .msg files can have PtypString8 streams ending with null terminators.
    Ensure subject, sender, recipients, attachments, and bodies have \x00 stripped
    so that From: header and text items do not leak null bytes. See #5535.
    """
    from unittest.mock import MagicMock, PropertyMock, patch

    from oxmsg import Message as OxMsgMessage

    data = Path("tests/data/email/sources/msg_simple.msg").read_bytes()

    recipient_mock = MagicMock()
    recipient_mock.name = "Bob\x00"
    recipient_mock.email_address = "bob@example.com\x00"
    recipient_mock.properties.int_prop_value.return_value = 1

    attachment_mock = MagicMock()
    attachment_mock.file_name = "test\x00.pdf\x00"
    attachment_mock.mime_type = "application/pdf\x00"

    subject_mock = PropertyMock(return_value="Subject With Null\x00")
    sender_mock = PropertyMock(return_value='"Alice\x00" <alice@example.com\x00>')
    body_mock = PropertyMock(return_value="Body text with trailing null\x00")
    html_body_mock = PropertyMock(
        return_value="<p>HTML body with trailing null\x00</p>"
    )
    recipients_mock = PropertyMock(return_value=[recipient_mock])
    attachments_mock = PropertyMock(return_value=[attachment_mock])

    with (
        patch.object(OxMsgMessage, "subject", subject_mock),
        patch.object(OxMsgMessage, "sender", sender_mock),
        patch.object(OxMsgMessage, "body", body_mock),
        patch.object(OxMsgMessage, "html_body", html_body_mock),
        patch.object(OxMsgMessage, "recipients", recipients_mock),
        patch.object(OxMsgMessage, "attachments", attachments_mock),
    ):
        rfc822, _ = EmailDocumentBackend._msg_to_rfc822_bytes(data)

    backend = EmailDocumentBackend(
        in_doc=InputDocument(
            path_or_stream=BytesIO(rfc822),
            format=InputFormat.EMAIL,
            filename="test_nul.eml",
            backend=EmailDocumentBackend,
        ),
        path_or_stream=BytesIO(rfc822),
        options=EmailBackendOptions(list_attachments=True),
    )
    assert backend.is_valid()
    doc = backend.convert()
    markdown = doc.export_to_markdown()

    # Verify no null character survives in text items or markdown
    for item in doc.texts:
        assert "\x00" not in item.text
    assert "\x00" not in markdown
    assert "From:" in markdown
    assert "Alice" in markdown
    assert "To:" in markdown
    assert "Bob" in markdown
    assert "Subject With Null" in markdown
    assert "Body text with trailing null" in markdown
    assert "test.pdf" in markdown
    assert "application/pdf" in markdown

    # Verify OLE stream fallback stripping in _recover_msg_body
    mock_msg = MagicMock()
    mock_msg.properties.int_prop_value.return_value = 1252
    mock_ole = MagicMock()
    mock_ole.exists.return_value = True
    mock_ole.openstream.return_value.read.return_value = b"Recovered body with null\x00"
    mock_ole_cm = MagicMock()
    mock_ole_cm.__enter__.return_value = mock_ole
    with patch("olefile.OleFileIO", return_value=mock_ole_cm):
        result = EmailDocumentBackend._recover_msg_body(b"dummy", mock_msg)
    assert result == "Recovered body with null"
