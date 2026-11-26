# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Sender handling of Outlook .msg files: Exchange senders without an SMTP address."""

from pathlib import Path

from docling.backend.email_backend import EmailDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def test_msg_exchange_sender_keeps_display_name():
    """An internal Exchange sender (address type EX) keeps its display name.

    python-oxmsg returns such a sender as the display name with the legacy
    Exchange DN in the angle brackets, e.g.
    ``Display Name </O=ORG/OU=.../CN=RECIPIENTS/CN=...>``, which is not a
    valid addr-spec: mailparser drops the whole mailbox and the converted
    document loses its ``From:`` line. Regression test for #5415.

    The fixture is msg_simple.msg with only the sender rewritten: the
    sender name is "Doe, Jonathan" and the sender address is the legacy DN
    "/O=EX/OU=HQ/CN=U1" (same byte lengths, so the .msg streams are
    untouched otherwise). The recipient keeps its SMTP address, so the
    ``To:`` line is unaffected. PidTagSenderAddressType is left as "SMTP":
    neither oxmsg's sender assembly nor the backend reads it.
    """
    in_path = Path("tests/data/email/sources/msg_exchange_sender.msg")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.EMAIL,
        backend=EmailDocumentBackend,
    )
    backend = EmailDocumentBackend(in_doc=in_doc, path_or_stream=in_path)

    assert backend.is_valid()
    assert backend.is_msg

    markdown = backend.convert().export_to_markdown()

    assert 'From: "Doe, Jonathan"' in markdown
    assert "To: Bob Example &lt;bob@example.com&gt;" in markdown
    assert "Date: 2026-05-20T10:30:00+00:00" in markdown
    assert "Hello Bob," in markdown
    assert "This is a simple email body." in markdown

    # The projected message carries the name in its From header, and the
    # fallback is what convert() renders since mailparser drops a bare name.
    raw, fallback = EmailDocumentBackend._msg_to_rfc822_bytes(in_path.read_bytes())
    assert b'From: "Doe, Jonathan"' in raw
    assert fallback == '"Doe, Jonathan"'


def test_msg_sender_header_keeps_smtp_sender_unchanged():
    """A sender with an SMTP address is passed through exactly as oxmsg gave it."""
    header = EmailDocumentBackend._msg_sender_header

    assert header('"Alice Example" <alice@example.com>', "Alice Example", None) == (
        '"Alice Example" <alice@example.com>',
        "",
    )
    assert header("alice@example.com", None, None) == ("alice@example.com", "")


def test_msg_sender_header_keeps_display_name_for_exchange_sender():
    """A legacy Exchange DN in the angle brackets keeps the display name."""
    header = EmailDocumentBackend._msg_sender_header

    dn_sender = "Display Name </O=ORG/OU=SITE/CN=RECIPIENTS/CN=USER>"
    assert header(dn_sender, None, None) == ("Display Name", "Display Name")
    # A name holding specials is quoted, as for regular addresses.
    assert header('"Doe, Jonathan" </O=EX/OU=HQ/CN=U1>', "Doe, Jonathan", None) == (
        '"Doe, Jonathan"',
        '"Doe, Jonathan"',
    )
    # Without a name in the sender string, PidTagSenderName fills in.
    assert header("</O=EX/OU=HQ/CN=U1>", "Sender Name", None) == (
        "Sender Name",
        "Sender Name",
    )
    # Without any name at all, the From header is omitted.
    assert header("</O=EX/OU=HQ/CN=U1>", None, None) == (None, "")


def test_msg_sender_header_prefers_recorded_smtp_address():
    """A stamped PidTagSenderSmtpAddress beats the legacy DN oxmsg shows."""
    header = EmailDocumentBackend._msg_sender_header

    assert header(
        '"Doe, Jonathan" </O=EX/OU=HQ/CN=U1>', "Doe, Jonathan", "jdoe@corp.example"
    ) == ('"Doe, Jonathan" <jdoe@corp.example>', "")
    assert header("</O=EX/OU=HQ/CN=U1>", None, "jdoe@corp.example") == (
        "jdoe@corp.example",
        "",
    )


def test_msg_sender_header_falls_back_to_bare_sender_name():
    """oxmsg falls back to the sent-representing name when no address exists."""
    header = EmailDocumentBackend._msg_sender_header

    assert header("Alice Example", None, None) == ("Alice Example", "Alice Example")
    assert header('"Doe, Jonathan"', None, None) == (
        '"Doe, Jonathan"',
        '"Doe, Jonathan"',
    )
    # Several addresses stay exactly as oxmsg gave them.
    multi = "Alice <alice@example.com>, Bob <bob@example.com>"
    assert header(multi, "Alice", None) == (multi, "")


def test_msg_sender_header_ignores_parentheses_in_legacy_dn():
    """A real legacy DN holds "(FYDIBOHF23SPDLT)"; it must not reach the name.

    getaddresses() reads the group in parentheses as a comment and appends
    it to the display name, so the name comes from PidTagSenderName.
    """
    header = EmailDocumentBackend._msg_sender_header
    dn = (
        "/O=EXCHANGELABS/OU=EXCHANGE ADMINISTRATIVE GROUP (FYDIBOHF23SPDLT)"
        "/CN=RECIPIENTS/CN=4F5E6A7B8C9D-JDOE"
    )
    sender = f'"Doe, Jonathan" <{dn}>'

    assert header(sender, "Doe, Jonathan", None) == (
        '"Doe, Jonathan"',
        '"Doe, Jonathan"',
    )
    assert header(sender, "Doe, Jonathan", "jdoe@corp.example") == (
        '"Doe, Jonathan" <jdoe@corp.example>',
        "",
    )
    # The property keeps quotes in a name that the parse would drop.
    assert header(f'"Jonathan \\"JD\\" Doe" <{dn}>', 'Jonathan "JD" Doe', None) == (
        '"Jonathan \\"JD\\" Doe"',
        '"Jonathan \\"JD\\" Doe"',
    )


def test_msg_sender_properties_that_fail_to_decode_keep_the_from_line():
    """The sender properties are guarded like every other .msg string read.

    python-oxmsg maps some code pages to codec names that Python does not
    know (e.g. "windows-950"), and it can pick the wrong codec for 8-bit
    strings. A failed PidTagSenderName or PidTagSenderSmtpAddress read must
    not stop the conversion: the From header then comes from the sender
    string alone.
    """
    from unittest.mock import PropertyMock, patch

    from oxmsg import Message as OxMsgMessage
    from oxmsg.properties import Properties

    original = Properties.str_prop_value

    def str_prop_value(self: Properties, pid: int) -> str | None:
        if pid in (0x0C1A, 0x5D01):
            raise LookupError("unknown encoding: windows-950")
        return original(self, pid)

    data = Path("tests/data/email/sources/msg_exchange_sender.msg").read_bytes()
    sender = PropertyMock(return_value='"Doe, Jonathan" </O=EX/OU=HQ/CN=U1>')
    with (
        patch.object(OxMsgMessage, "sender", sender),
        patch.object(Properties, "str_prop_value", str_prop_value),
    ):
        rfc822, fallback = EmailDocumentBackend._msg_to_rfc822_bytes(data)

    assert b'From: "Doe, Jonathan"' in rfc822
    assert fallback == '"Doe, Jonathan"'


def test_msg_sender_header_keeps_bare_name_with_comma():
    """oxmsg returns PidTagSentRepresentingName unquoted, e.g. "Doe, Jonathan"."""
    header = EmailDocumentBackend._msg_sender_header

    assert header("Doe, Jonathan", None, None) == (
        '"Doe, Jonathan"',
        '"Doe, Jonathan"',
    )
