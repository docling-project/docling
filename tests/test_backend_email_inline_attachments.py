# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import struct
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest
from docling_core.types.doc import ListItem
from oxmsg import Message as OxMsgMessage
from oxmsg.attachment import Attachment
from oxmsg.storage import Storage, Stream

from docling.backend.email_backend import EmailDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def _property_storage(
    path: str,
    header_size: int,
    properties: dict[int, str | bytes | bool],
    children: tuple[Storage, ...] = (),
) -> Storage:
    """Build real MAPI properties without a private Outlook message fixture."""
    segments = bytearray(header_size)
    streams = []
    for pid, value in properties.items():
        if isinstance(value, bool):
            segments.extend(
                struct.pack("<HHI8s", 0x000B, pid, 0, bytes([value]) + b"\0" * 7)
            )
        else:
            ptyp = 0x001F if isinstance(value, str) else 0x0102
            payload = value.encode("utf-16-le") if isinstance(value, str) else value
            segments.extend(struct.pack("<HHIII", ptyp, pid, 0, len(payload), 0))
            streams.append(Stream(f"{path}/__substg1.0_{pid:04X}{ptyp:04X}", payload))
    streams.append(Stream(f"{path}/__properties_version1.0", bytes(segments)))
    return Storage(path, tuple(streams), children)


@pytest.mark.parametrize(
    ("html", "content_id", "hidden", "listed"),
    [
        (
            '<p>Body.</p><img src="cid:logo@example.com">',
            "logo@example.com",
            None,
            False,
        ),
        (
            '<p>Body.</p><img src="CID:logo%40example.com">',
            "logo@example.com",
            False,
            False,
        ),
        (
            '<p>Body.</p><img src="cid:logo@example.com">',
            "<logo@example.com>",
            None,
            False,
        ),
        (
            '<p>Body.</p><img src="cid:logo@example.com.extra">',
            "logo@example.com",
            None,
            True,
        ),
        (
            '<p>Body.</p><img src="cid:other@example.com">',
            "logo@example.com",
            None,
            True,
        ),
        ("<p>Body.</p>", "logo@example.com", None, True),
        (None, "logo@example.com", None, True),
        (
            '<p>Body.</p><a href="cid:logo@example.com">Logo</a>',
            "logo@example.com",
            None,
            False,
        ),
        (
            '<body background="cid:logo@example.com">Body.</body>',
            "logo@example.com",
            None,
            False,
        ),
        (
            '<object data="cid:logo@example.com"></object>',
            "logo@example.com",
            None,
            False,
        ),
        ('<img src="cid:logo%2B1%40example.com">', "logo+1@example.com", None, False),
        ('<img src="cid:logo&amp;1@example.com">', "logo&1@example.com", None, False),
        ('<img src="cid:LOGO@example.com">', "logo@example.com", None, True),
        (None, None, True, False),
        (None, None, False, True),
        (None, None, None, True),
        (
            "<p>Body. cid:logo@example.com is a literal example.</p>",
            "logo@example.com",
            None,
            True,
        ),
        ('<p>Body.</p><img src="cid:logo@example.com">', None, None, True),
    ],
)
def test_msg_inline_attachment_metadata(
    monkeypatch: pytest.MonkeyPatch,
    html: str | None,
    content_id: str | None,
    hidden: bool | None,
    listed: bool,
) -> None:
    properties: dict[int, str | bytes | bool] = {
        0x3707: "logo.png",
        0x370E: "image/png",
        0x3701: b"private-image-payload-not-for-export",
    }
    if content_id is not None:
        properties[0x3712] = content_id
    if hidden is not None:
        properties[0x7FFE] = hidden
    logo = _property_storage("__attach_version1.0_#00000000", 8, properties)
    report = _property_storage(
        "__attach_version1.0_#00000001",
        8,
        {
            0x3707: "report.pdf",
            0x370E: "application/pdf",
            0x3701: b"private-report-payload",
        },
    )
    message_properties: dict[int, str | bytes | bool] = {
        0x0037: "Inline attachment example",
        0x1000: "Body.",
    }
    if html is not None:
        message_properties[0x1013] = html.encode("utf-8")
    message = OxMsgMessage(
        _property_storage("", 32, message_properties, (logo, report))
    )
    # Only the CFB file loader is replaced. Real oxmsg property decoding, MIME
    # projection, mail-parser, document structure and Markdown export still run.
    monkeypatch.setattr(OxMsgMessage, "load", lambda _: message)

    def reject_payload_read(self: Attachment) -> bytes:
        raise AssertionError(
            "Attachment payloads must not be read for metadata listing"
        )

    monkeypatch.setattr(Attachment, "file_bytes", property(reject_payload_read))
    path = Path("tests/data/email/sources/msg_with_attachment.msg")
    raw = EmailDocumentBackend._msg_to_rfc822_bytes(path.read_bytes())
    mime = BytesParser(policy=policy.default).parsebytes(raw)
    names = [part.get_filename() for part in mime.iter_attachments()]
    expected = (["logo.png"] if listed else []) + ["report.pdf"]
    assert names == expected
    assert b"private-image-payload" not in raw
    assert b"private-report-payload" not in raw

    in_doc = InputDocument(
        path_or_stream=path, format=InputFormat.EMAIL, backend=EmailDocumentBackend
    )
    backend = EmailDocumentBackend(
        in_doc=in_doc,
        path_or_stream=path,
        options=EmailBackendOptions(list_attachments=True),
    )
    doc = backend.convert()
    items = [item for item in doc.texts if isinstance(item, ListItem)]
    assert [item.text for item in items] == (
        (["logo.png (image/png)"] if listed else []) + ["report.pdf (application/pdf)"]
    )
    assert all(
        item.parent is not None and item.parent.resolve(doc).name == "attachments"
        for item in items
    )
    markdown = doc.export_to_markdown()
    assert "Body." in markdown
    assert "report.pdf (application/pdf)" in markdown
    assert ("logo.png" in markdown) is listed
