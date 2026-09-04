# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import binascii
import logging
import quopri
import re
from collections.abc import Callable
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime, formataddr
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions, HTMLBackendOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# mail-parser is only installed by the `format-email` extra, but
# DocumentConverter imports every backend eagerly. Importing it at module load
# would therefore break `import docling` on installs that omit the extra (the
# slim packages in particular). Guard the import like the opendocument and xbrl
# backends do, and surface the failure only when an email is actually parsed.
# See https://github.com/docling-project/docling/issues/3613.
_MAILPARSER_AVAILABLE: bool = False
_MAILPARSER_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    import mailparser

    _MAILPARSER_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _MAILPARSER_IMPORT_ERROR = e

# python-oxmsg (MIT, from the python-docx/python-pptx author, built on olefile)
# reads Outlook .msg files. We assemble a standard email.message.EmailMessage
# from the parsed message, which mailparser then parses through the same path as
# .eml input. It ships in the `format-email` extra, so guard the import for the
# same slim-install reason as mailparser above.
_OXMSG_AVAILABLE: bool = False
_OXMSG_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from oxmsg import Message as OxMsgMessage

    _OXMSG_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _OXMSG_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

# DocumentConverter lives in the entrypoints layer, which the backend (core)
# layer must not import (enforced by Tach). Instead, docling.document_converter
# injects a factory here at import time, inverting the dependency. When the
# factory is unset (e.g. the converter module was never imported), attachment
# processing gracefully degrades to listing.
_attachment_converter_factory: Callable[[], DocumentConverter] | None = None


def register_attachment_converter_factory(
    factory: Callable[[], DocumentConverter],
) -> None:
    """Register the factory used to build a converter for email attachments."""
    global _attachment_converter_factory
    _attachment_converter_factory = factory


# OLE2 / Compound File Binary signature that prefixes every Outlook .msg file.
_MSG_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# MAPI PidTagRecipientType (0x0C15) values on each .msg recipient row.
_RECIPIENT_TYPE_PROP_ID = 0x0C15
_RECIPIENT_TYPE_CC = 2
_RECIPIENT_TYPE_BCC = 3

_INSTALL_HINT = (
    "The 'mail-parser' package is required to process email files. "
    "Install it with `pip install 'docling-slim[format-email]'`."
)

_MSG_INSTALL_HINT = (
    "The 'python-oxmsg' package is required to process Outlook .msg files. "
    "Install it with `pip install 'docling-slim[format-email]'`."
)


class EmailDocumentBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: EmailBackendOptions | None = None,
    ):
        # Raised before super().__init__() so a missing optional dependency
        # gives an actionable message rather than a NameError when mailparser
        # is dereferenced below.
        if not _MAILPARSER_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _MAILPARSER_IMPORT_ERROR
        if options is None:
            options = EmailBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        self.options: EmailBackendOptions = options
        self.valid = False
        self.is_msg = False
        self.mail: mailparser.MailParser | None = None

        try:
            raw = self._read_bytes()
            self.is_msg = raw.startswith(_MSG_MAGIC)
            if self.is_msg:
                raw = self._msg_to_rfc822_bytes(raw)
            self.mail = mailparser.parse_from_bytes(raw)

            self.valid = self.mail is not None
        except ImportError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not initialize email backend for file with hash {self.document_hash}."
            ) from exc

    def _read_bytes(self) -> bytes:
        if isinstance(self.path_or_stream, BytesIO):
            return self.path_or_stream.getvalue()
        if isinstance(self.path_or_stream, Path):
            return self.path_or_stream.read_bytes()
        raise TypeError(f"Unsupported input type: {type(self.path_or_stream)}")

    @staticmethod
    def _header_safe(value: str) -> str:
        # Email header values must be single-line; collapse CR/LF to spaces so a
        # crafted .msg cannot inject headers and EmailMessage does not reject it.
        return value.replace("\r", " ").replace("\n", " ").strip()

    @staticmethod
    def _msg_to_rfc822_bytes(data: bytes) -> bytes:
        """Project an Outlook ``.msg`` (OLE2/CFB) onto RFC 822 bytes.

        python-oxmsg reads the MAPI message; we assemble a standard
        ``email.message.EmailMessage`` from it so the ``.msg`` path shares the
        exact body, HTML, address, and attachment handling used for ``.eml``
        input.
        """
        if not _OXMSG_AVAILABLE:
            raise ImportError(_MSG_INSTALL_HINT) from _OXMSG_IMPORT_ERROR

        message = OxMsgMessage.load(data)
        email_message = EmailMessage()

        if message.subject:
            email_message["Subject"] = EmailDocumentBackend._header_safe(
                message.subject
            )
        if message.sender:
            email_message["From"] = EmailDocumentBackend._header_safe(message.sender)

        # Preserve the To/Cc/Bcc split from the recipient rows so downstream
        # rendering (which shows only "To") matches the .eml behavior.
        grouped: dict[str, list[str]] = {}
        for recipient in message.recipients:
            formatted = formataddr(
                (recipient.name or "", recipient.email_address or "")
            )
            if not formatted:
                continue
            rtype = recipient.properties.int_prop_value(_RECIPIENT_TYPE_PROP_ID)
            if rtype == _RECIPIENT_TYPE_CC:
                header = "Cc"
            elif rtype == _RECIPIENT_TYPE_BCC:
                header = "Bcc"
            else:
                header = "To"
            grouped.setdefault(header, []).append(formatted)
        for header in ("To", "Cc", "Bcc"):
            if grouped.get(header):
                email_message[header] = EmailDocumentBackend._header_safe(
                    ", ".join(grouped[header])
                )

        if isinstance(message.sent_date, datetime):
            email_message["Date"] = format_datetime(message.sent_date)

        plain_body = message.body
        html_body = message.html_body
        if plain_body:
            email_message.set_content(plain_body)
            if html_body:
                email_message.add_alternative(html_body, subtype="html")
        elif html_body:
            email_message.set_content(html_body, subtype="html")
        else:
            email_message.set_content("")

        for attachment in message.attachments:
            mime_type = attachment.mime_type or "application/octet-stream"
            maintype, _, subtype = mime_type.partition("/")
            email_message.add_attachment(
                attachment.file_bytes or b"",
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=attachment.file_name,
            )

        return email_message.as_bytes()

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.EMAIL}

    def _format_addresses(
        self, addresses: list[tuple[str, str]] | None, fallback: str
    ) -> str:
        if not addresses:
            return fallback

        formatted = []
        for name, email in addresses:
            if name:
                formatted.append(f"{name} <{email}>")
            else:
                formatted.append(email)

        return ", ".join(formatted)

    def _split_paragraphs(self, text: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text.strip())
            if paragraph.strip()
        ]

    def _convert_html_part(self, html: str) -> DoclingDocument:
        html_stream = BytesIO(html.encode("utf-8"))
        in_doc = InputDocument(
            path_or_stream=html_stream,
            format=InputFormat.HTML,
            filename="email-body.html",
            backend=HTMLDocumentBackend,
        )
        html_stream.seek(0)
        backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=html_stream,
            options=HTMLBackendOptions(add_title=False, infer_furniture=False),
        )
        return backend.convert()

    def _get_body_paragraphs(self) -> list[str]:
        assert self.mail is not None

        if self.mail.text_plain:
            paragraphs: list[str] = []
            for part in self.mail.text_plain:
                paragraphs.extend(self._split_paragraphs(part))
            return paragraphs

        if self.mail.text_html:
            paragraphs = []
            for part in self.mail.text_html:
                html_doc = self._convert_html_part(part)
                paragraphs.extend(self._split_paragraphs(html_doc.export_to_markdown()))
            return paragraphs

        return self._split_paragraphs(self.mail.body)

    def _get_date_text(self) -> str:
        assert self.mail is not None

        mail_date = self.mail.date
        if isinstance(mail_date, datetime):
            return mail_date.isoformat()
        if isinstance(mail_date, str):
            return mail_date.strip()
        return ""

    @staticmethod
    def _attachment_filename(index: int, attachment: dict[str, Any]) -> str:
        filename = (attachment.get("filename") or "").strip()
        return filename or f"attachment-{index + 1}"

    @staticmethod
    def _attachment_label(index: int, attachment: dict[str, Any]) -> str:
        filename = EmailDocumentBackend._attachment_filename(index, attachment)
        content_type = (attachment.get("mail_content_type") or "").strip()
        return f"{filename} ({content_type})" if content_type else filename

    @staticmethod
    def _get_attachment_bytes(attachment: dict[str, Any]) -> bytes | None:
        """Decode an attachment's payload into raw bytes.

        mailparser exposes the payload as it appeared in the MIME part together
        with its transfer encoding, so we reverse that encoding here. Returns
        ``None`` when the payload is missing or cannot be decoded.
        """
        payload = attachment.get("payload")
        if payload is None:
            return None

        encoding = (attachment.get("content_transfer_encoding") or "").lower()
        try:
            if encoding == "base64":
                return base64.b64decode(payload)
            if encoding == "quoted-printable":
                data = payload.encode("utf-8") if isinstance(payload, str) else payload
                return quopri.decodestring(data)
        except (ValueError, binascii.Error):
            return None

        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            charset = attachment.get("charset") or "utf-8"
            try:
                return payload.encode(charset, errors="replace")
            except LookupError:
                return payload.encode("utf-8", errors="replace")
        return None

    @staticmethod
    def _build_attachment_converter() -> DocumentConverter | None:
        # The converter (entrypoints layer) is injected via
        # register_attachment_converter_factory so this core-layer backend never
        # imports it. Returns None when no factory has been registered, in which
        # case attachments are listed instead of parsed. One converter is built
        # per email and reused across its attachments (pipelines are cached per
        # converter instance); it uses default options, so a nested email
        # attachment is parsed but its own attachments are not recursed.
        if _attachment_converter_factory is None:
            return None
        return _attachment_converter_factory()

    def _convert_attachment(
        self, converter: DocumentConverter, filename: str, data: bytes
    ) -> DoclingDocument | None:
        """Convert one attachment via Docling, or ``None`` if it cannot be parsed."""
        stream = DocumentStream(name=filename, stream=BytesIO(data))
        try:
            result = converter.convert(stream, raises_on_error=False)
        except Exception as exc:  # unsupported format, missing extra, bad bytes
            _log.info("Could not convert email attachment %r: %s", filename, exc)
            return None
        if result.status not in (
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        ):
            _log.info(
                "Skipped email attachment %r (conversion status: %s)",
                filename,
                result.status,
            )
            return None
        return result.document

    def _list_attachments(
        self, doc: DoclingDocument, attachments: list[dict[str, Any]]
    ) -> None:
        attachments_group = doc.add_list_group(name="attachments")
        for index, attachment in enumerate(attachments):
            doc.add_list_item(
                text=self._attachment_label(index, attachment),
                parent=attachments_group,
            )

    def _process_attachments(
        self,
        doc: DoclingDocument,
        attachments: list[dict[str, Any]],
        converter: DocumentConverter,
    ) -> None:
        total_bytes = 0
        for index, attachment in enumerate(attachments):
            if index >= self.options.max_attachments:
                break
            filename = self._attachment_filename(index, attachment)
            label = self._attachment_label(index, attachment)
            section = doc.add_heading(text=filename, level=3)

            data = self._get_attachment_bytes(attachment)
            if not data:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"{label}: attachment content could not be read.",
                )
                continue
            if len(data) > self.options.max_attachment_bytes:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"{label}: skipped (exceeds max_attachment_bytes).",
                )
                continue
            if total_bytes + len(data) > self.options.max_attachment_total_bytes:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"{label}: skipped (attachment size budget exceeded).",
                )
                continue
            total_bytes += len(data)

            child = self._convert_attachment(converter, filename, data)
            if child is None:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"{label}: could not be parsed by Docling.",
                )
            else:
                doc.add_document(child, parent=section)

    def _add_attachments(self, doc: DoclingDocument) -> None:
        """Append the attachments section, listing or parsing per the options."""
        assert self.mail is not None

        attachments = self.mail.attachments or []
        if not attachments:
            return
        if not (self.options.process_attachments or self.options.list_attachments):
            return

        # Fall back to listing when parsing is off, or when no converter factory
        # was registered (the entrypoints layer was never imported).
        converter = (
            self._build_attachment_converter()
            if self.options.process_attachments
            else None
        )

        doc.add_heading(text="Attachments", level=2)
        if converter is None:
            self._list_attachments(doc, attachments)
        else:
            self._process_attachments(doc, attachments, converter)

    def convert(self) -> DoclingDocument:
        if not self.is_valid() or self.mail is None:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        # A .msg is projected onto RFC 822 for conversion, so the origin uses
        # the message/rfc822 mimetype for both inputs (DocumentOrigin only
        # accepts registered MIME types); the .msg filename fallback preserves
        # the distinction when no filename is available.
        origin = DocumentOrigin(
            filename=self.file.name or ("file.msg" if self.is_msg else "file.eml"),
            mimetype="message/rfc822",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        subject = (
            self.mail.subject.strip() if isinstance(self.mail.subject, str) else ""
        )
        from_text = self._format_addresses(self.mail.from_, fallback="")
        to_text = self._format_addresses(self.mail.to, fallback="")
        date_text = self._get_date_text()
        body_paragraphs = self._get_body_paragraphs()

        if subject:
            doc.add_title(text=subject)
        if from_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"From: {from_text}")
        if to_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"To: {to_text}")
        if date_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"Date: {date_text}")
        for body_paragraph in body_paragraphs:
            doc.add_text(label=DocItemLabel.TEXT, text=body_paragraph)

        self._add_attachments(doc)

        return doc
