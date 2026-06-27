from __future__ import annotations

import base64
import binascii
import logging
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import (
    BackendOptions,
    EmailBackendOptions,
    HTMLBackendOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

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

_log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'mail-parser' package is required to process email files. "
    "Install it with `pip install 'docling[format-email]'`."
)

# Outlook `.msg` parsing needs the optional 'extract-msg' package. It is only
# imported when an actual `.msg` (OLE2) file is encountered, so `.eml` parsing
# and `import docling` keep working without it.
_MSG_INSTALL_HINT = (
    "The 'extract-msg' package is required to process Outlook '.msg' files. "
    "Install it with `pip install 'docling[format-email]'` or `pip install extract-msg`."
)

# OLE2 / Compound File Binary Format magic bytes. Outlook `.msg` files are stored
# in this binary container, unlike RFC822 `.eml` files which are plain MIME text.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass(frozen=True)
class _EmailAttachment:
    filename: str
    media_type: str
    size: int | None = None


class EmailDocumentBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: BackendOptions | None = None,
    ):
        # Raised before super().__init__() so a missing optional dependency
        # gives an actionable message rather than a NameError when mailparser
        # is dereferenced below.
        if not _MAILPARSER_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _MAILPARSER_IMPORT_ERROR
        super().__init__(in_doc, path_or_stream, options)

        self.options: EmailBackendOptions = (
            options
            if isinstance(options, EmailBackendOptions)
            else EmailBackendOptions()
        )

        self.valid = False
        self.mimetype = "message/rfc822"
        self.subject = ""
        self.from_text = ""
        self.to_text = ""
        self.date_text = ""
        self.body_paragraphs: list[str] = []
        self.attachments: list[_EmailAttachment] = []

        try:
            if isinstance(self.path_or_stream, BytesIO):
                data = self.path_or_stream.getvalue()
            elif isinstance(self.path_or_stream, Path):
                data = self.path_or_stream.read_bytes()
            else:
                raise TypeError(f"Unsupported input type: {type(self.path_or_stream)}")

            if data[:8] == _OLE2_MAGIC:
                self._parse_msg(data)
            else:
                self._parse_eml(data)

            self.valid = True
        except ImportError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not initialize email backend for file with hash {self.document_hash}."
            ) from exc

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.EMAIL}

    # --- Shared helpers -----------------------------------------------------

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

    def _format_date(self, value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value.strip()
        return ""

    def _format_size(self, size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB"):
            if value < 1024:
                return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} GB"

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

    # --- RFC822 / .eml ------------------------------------------------------

    def _parse_eml(self, data: bytes) -> None:
        mail = mailparser.parse_from_bytes(data)
        self.mimetype = "message/rfc822"
        self.subject = mail.subject.strip() if isinstance(mail.subject, str) else ""
        self.from_text = self._format_addresses(mail.from_, fallback="")
        self.to_text = self._format_addresses(mail.to, fallback="")
        self.date_text = self._format_date(mail.date)
        self.body_paragraphs = self._eml_body_paragraphs(mail)
        self.attachments = self._eml_attachments(mail)

    def _eml_body_paragraphs(self, mail: mailparser.MailParser) -> list[str]:
        if mail.text_plain:
            paragraphs: list[str] = []
            for part in mail.text_plain:
                paragraphs.extend(self._split_paragraphs(part))
            return paragraphs

        if mail.text_html:
            paragraphs = []
            for part in mail.text_html:
                html_doc = self._convert_html_part(part)
                paragraphs.extend(self._split_paragraphs(html_doc.export_to_markdown()))
            return paragraphs

        return self._split_paragraphs(mail.body)

    def _eml_attachments(self, mail: mailparser.MailParser) -> list[_EmailAttachment]:
        attachments: list[_EmailAttachment] = []
        for att in mail.attachments or []:
            filename = str(att.get("filename") or "").strip() or "attachment"
            media_type = (
                str(att.get("mail_content_type") or "").strip()
                or "application/octet-stream"
            )
            attachments.append(
                _EmailAttachment(
                    filename=filename,
                    media_type=media_type,
                    size=self._eml_attachment_size(att),
                )
            )
        return attachments

    def _eml_attachment_size(self, att: dict) -> int | None:
        payload = att.get("payload")
        if not isinstance(payload, str):
            return None
        encoding = str(att.get("content_transfer_encoding") or "").lower()
        if encoding == "base64":
            try:
                return len(base64.b64decode(payload))
            except (binascii.Error, ValueError):
                return None
        return len(payload.encode("utf-8", errors="ignore"))

    # --- Outlook / .msg -----------------------------------------------------

    def _parse_msg(self, data: bytes) -> None:
        try:
            import extract_msg
        except ImportError as exc:
            raise ImportError(_MSG_INSTALL_HINT) from exc

        self.mimetype = "application/vnd.ms-outlook"
        # `openMsg` decodes a path argument as text; pass a file-like object so the
        # raw bytes are read as the OLE2 message content instead. The returned
        # message is dynamically typed (extract-msg ships only partial type info),
        # so treat it as `Any` to keep the attribute access readable.
        opened: Any = extract_msg.openMsg(BytesIO(data))
        with opened as msg:
            self.subject = (msg.subject or "").strip()
            self.from_text = (msg.sender or "").strip()
            self.to_text = (msg.to or "").strip()
            self.date_text = self._format_date(msg.date)

            plain_body = msg.body
            if plain_body:
                self.body_paragraphs = self._split_paragraphs(plain_body)
            else:
                html_body = msg.htmlBody
                if html_body:
                    if isinstance(html_body, (bytes, bytearray)):
                        html_body = html_body.decode("utf-8", errors="replace")
                    html_doc = self._convert_html_part(html_body)
                    self.body_paragraphs = self._split_paragraphs(
                        html_doc.export_to_markdown()
                    )

            attachments: list[_EmailAttachment] = []
            for att in msg.attachments:
                filename = str(att.getFilename() or "").strip() or "attachment"
                payload = att.data
                size = len(payload) if isinstance(payload, (bytes, bytearray)) else None
                media_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )
                attachments.append(
                    _EmailAttachment(
                        filename=filename, media_type=media_type, size=size
                    )
                )
            self.attachments = attachments

    # --- Document assembly --------------------------------------------------

    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file.eml",
            mimetype=self.mimetype,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        if self.subject:
            doc.add_title(text=self.subject)
        if self.from_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"From: {self.from_text}")
        if self.to_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"To: {self.to_text}")
        if self.date_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"Date: {self.date_text}")
        for body_paragraph in self.body_paragraphs:
            doc.add_text(label=DocItemLabel.TEXT, text=body_paragraph)

        if self.options.include_attachments and self.attachments:
            doc.add_heading(text="Attachments", level=1)
            for att in self.attachments:
                detail = att.media_type
                if att.size is not None:
                    detail = f"{att.media_type}, {self._format_size(att.size)}"
                doc.add_text(label=DocItemLabel.TEXT, text=f"{att.filename} ({detail})")

        return doc
