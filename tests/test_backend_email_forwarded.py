# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Email body selection with forwarded messages and digests."""

from io import BytesIO

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter


def _convert_raw_email(raw_email: bytes) -> DoclingDocument:
    return (
        DocumentConverter(allowed_formats=[InputFormat.EMAIL])
        .convert(DocumentStream(name="test.eml", stream=BytesIO(raw_email)))
        .document
    )


def _forward_email(body_part: bytes, forwarded_headers: bytes) -> bytes:
    return (
        b"From: alice@example.com\nSubject: Forward\nMIME-Version: 1.0\n"
        b'Content-Type: multipart/mixed; boundary="B"\n\n'
        b"--B\n"
        + body_part
        + b"--B\nContent-Type: message/rfc822\n"
        + forwarded_headers
        + b"\nFrom: bob@example.com\nSubject: Original\n"
        b'Content-Type: text/plain; charset="utf-8"\n\n'
        b"Original text.\n"
        b"--B--\n"
    )


@pytest.mark.parametrize(
    "body_part",
    [
        pytest.param(
            b'Content-Type: text/html; charset="utf-8"\n\n'
            b"<html><body><p>See the forwarded message.</p></body></html>\n",
            id="html-body",
        ),
        pytest.param(
            b'Content-Type: text/plain; charset="utf-8"\n\n'
            b"See the forwarded message.\n",
            id="plain-body",
        ),
    ],
)
def test_email_backend_keeps_body_apart_from_attached_message(
    body_part: bytes,
) -> None:
    """The text of a message/rfc822 attachment is not read as the body."""
    raw_email = _forward_email(
        body_part, b'Content-Disposition: attachment; filename="forwarded.eml"\n'
    )

    doc = _convert_raw_email(raw_email)

    assert [item.text for item in doc.texts] == [
        "Forward",
        "From: alice@example.com",
        "See the forwarded message.",
    ]


@pytest.mark.parametrize(
    "forwarded_headers",
    [
        pytest.param(b"", id="no-disposition"),
        pytest.param(b"Content-Disposition: inline\n", id="inline"),
    ],
)
def test_email_backend_keeps_inline_message_in_body(forwarded_headers: bytes) -> None:
    """A message/rfc822 part that is not an attachment stays in the body."""
    raw_email = _forward_email(
        b'Content-Type: text/plain; charset="utf-8"\n\nSee the forwarded message.\n',
        forwarded_headers,
    )

    doc = _convert_raw_email(raw_email)

    assert [item.text for item in doc.texts] == [
        "Forward",
        "From: alice@example.com",
        "See the forwarded message.",
        "Original text.",
    ]


def test_email_backend_keeps_digest_messages_in_body() -> None:
    """The messages of a multipart/digest are part of the body."""
    raw_email = b"""From: list@example.com
Subject: Dev Digest
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="O"

--O
Content-Type: text/plain

Today's Topics: shipping
--O
Content-Type: multipart/digest; boundary="D"

--D

From: a@example.com
Subject: Ship
Content-Type: text/plain

We ship on Friday.
--D

From: b@example.com
Subject: Re: Ship
Content-Type: text/plain

Friday works for me.
--D--
--O
Content-Type: text/plain

End of Dev Digest
--O--
"""

    doc = _convert_raw_email(raw_email)

    assert [item.text for item in doc.texts] == [
        "Dev Digest",
        "From: list@example.com",
        "Today's Topics: shipping",
        "We ship on Friday.",
        "Friday works for me.",
        "End of Dev Digest",
    ]
