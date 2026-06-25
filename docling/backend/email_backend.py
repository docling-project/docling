import logging
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import mailparser
from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)


class EmailDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path):
        super().__init__(in_doc, path_or_stream)

        self.valid = False
        self.mail: mailparser.MailParser | None = None

        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.mail = mailparser.parse_from_bytes(self.path_or_stream.getvalue())
            elif isinstance(self.path_or_stream, Path):
                self.mail = mailparser.parse_from_file(str(self.path_or_stream))
            else:
                raise TypeError(f"Unsupported input type: {type(self.path_or_stream)}")

            self.valid = self.mail is not None
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

    def convert(self) -> DoclingDocument:
        if not self.is_valid() or self.mail is None:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file.eml",
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

        return doc
