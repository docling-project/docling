import logging
import warnings
from io import BytesIO, StringIO
from pathlib import Path
from typing import Set, Union

from docling_core.types.doc import (
    DoclingDocument,
    DocumentOrigin,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class AudioBackend(DeclarativeDocumentBackend):
    # content: StringIO

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        _log.info(f"path: {path_or_stream}")

        # Load content
        try:
            if isinstance(self.path_or_stream, BytesIO):
                _log.info(f"reading streaming: {self.path_or_stream}")
                # self.content = StringIO(self.path_or_stream.getvalue().decode("utf-8"))
            elif isinstance(self.path_or_stream, Path):
                _log.info(f"reading file: {self.path_or_stream}")
                # self.content = StringIO(self.path_or_stream.read())
            self.valid = True
        except Exception as e:
            raise RuntimeError(
                f"AudioBackend could not load document with hash {self.document_hash}"
            ) from e
        return

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.AUDIO_WAV}

    def convert(self) -> DoclingDocument:
        """
        Parses the audio file into a structured document model.
        """

        # Parse the CSV into a structured document model
        origin = DocumentOrigin(
            filename=self.file.name or "audio.wav",
            mimetype="audio/wav",
            binary_hash=self.document_hash,
        )
        _log.info(f"origin: {origin}")

        doc = DoclingDocument(name=self.file.stem or "audio.wav", origin=origin)

        if self.is_valid():
            _log.error("time to get going ...")
        else:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the audio backend failed to init."
            )

        return doc
