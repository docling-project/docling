from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class WavDocumentBackend(AbstractDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

    def is_valid(self) -> bool:
        return True

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()

        self.path_or_stream = None

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.WAV}
