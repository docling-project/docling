import logging
from io import BytesIO
from pathlib import Path
from typing import Set, Union

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class NoOpBackend(AbstractDocumentBackend):
    """
    A no-op backend that only validates input existence.
    Used e.g. for audio files where actual processing is handled by the ASR pipeline.
    """

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        _log.debug(f"NoOpBackend initialized for: {path_or_stream}")

        # Validate input
        try:
            if isinstance(self.path_or_stream, BytesIO):
                # Check if stream has content
                self.valid = len(self.path_or_stream.getvalue()) > 0
                _log.debug(
                    f"BytesIO stream length: {len(self.path_or_stream.getvalue())}"
                )
            elif isinstance(self.path_or_stream, Path):
                # Check if file exists
                self.valid = self.path_or_stream.exists()
                _log.debug(f"File exists: {self.valid}")
            else:
                self.valid = False
        except Exception as e:
            _log.error(f"NoOpBackend validation failed: {e}")
            self.valid = False

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return set(InputFormat)
