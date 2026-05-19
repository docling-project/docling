import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import mailparser
from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class EmailDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
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
            raise RuntimeError(
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

    def _extract_attachments(self) -> list[dict[str, Any]]:
        """Stub for future attachment extraction support."""
        return []

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

    def _get_body_text(self) -> str:
        assert self.mail is not None

        if self.mail.text_plain:
            return "\n\n".join(
                part.strip() for part in self.mail.text_plain if part.strip()
            )

        if self.mail.text_html:
            return "\n\n".join(
                part.strip() for part in self.mail.text_html if part.strip()
            )

        return self.mail.body.strip()

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
        body_text = self._get_body_text()

        if subject:
            doc.add_title(text=subject)
        if from_text:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=f"From: {from_text}")
        if to_text:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=f"To: {to_text}")
        if body_text:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=body_text)

        _ = self._extract_attachments()

        return doc
