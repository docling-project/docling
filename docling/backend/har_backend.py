import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Union

from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

# ponytail: hard cap on inline body previews; raise if downstream needs full bodies
_MAX_BODY_CHARS = 2000


def _is_text_mime(mime: str) -> bool:
    return mime.startswith(("text/", "application/json", "application/xml"))


class HarDocumentBackend(DeclarativeDocumentBackend):
    """Backend for HTTP Archive (HAR) files.

    Converts HAR request/response entries into a structured DoclingDocument
    suitable for downstream AI workflows.
    """

    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        try:
            if isinstance(self.path_or_stream, BytesIO):
                raw = self.path_or_stream.getvalue().decode("utf-8")
            else:
                raw = Path(self.path_or_stream).read_text("utf-8")
            self._har: dict = json.loads(raw)
            self.valid = True
        except Exception as e:
            raise DocumentLoadError(
                f"HarDocumentBackend could not load document with hash {self.document_hash}"
            ) from e

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self) -> None:
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.HAR}

    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file.har",
            mimetype="application/json",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        try:
            log = self._har["log"]
        except (KeyError, TypeError):
            _log.warning("HAR file missing 'log' key, returning empty document.")
            return doc

        entries: list[dict] = log.get("entries", [])
        if not entries:
            _log.warning("HAR file contains no entries.")
            return doc

        doc.add_title(text=self.file.name or "HTTP Archive")

        for entry in entries:
            req: dict = entry.get("request", {})
            resp: dict = entry.get("response", {})

            method = req.get("method", "")
            url = req.get("url", "")
            status = resp.get("status", "")
            status_text = resp.get("statusText", "")

            heading = f"{method} {url}".strip()
            if heading:
                doc.add_heading(text=heading, level=1)

            if status:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"Response: {status} {status_text}".strip(),
                )

            post_data: dict = req.get("postData") or {}
            req_body = post_data.get("text", "")
            if req_body:
                doc.add_code(text=req_body[:_MAX_BODY_CHARS])

            content: dict = resp.get("content") or {}
            resp_body = content.get("text", "")
            if resp_body and _is_text_mime(content.get("mimeType", "")):
                doc.add_code(text=resp_body[:_MAX_BODY_CHARS])

        return doc
