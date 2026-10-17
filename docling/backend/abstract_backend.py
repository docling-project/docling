# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling_core.types.doc import DoclingDocument

from docling.datamodel.backend_options import (
    BackendOptions,
    BaseBackendOptions,
    DeclarativeBackendOptions,
)

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument


class AbstractDocumentBackend(ABC):
    @abstractmethod
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[BaseBackendOptions] = None,
    ):
        self.file = in_doc.file
        self.path_or_stream = path_or_stream
        self.document_hash = in_doc.document_hash
        self.input_format = in_doc.format
        self.options = BaseBackendOptions() if options is None else options

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def supports_pagination(cls) -> bool:
        pass

    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()

        self.path_or_stream = None

    @classmethod
    @abstractmethod
    def supported_formats(cls) -> set["InputFormat"]:
        pass


class PaginatedDocumentBackend(AbstractDocumentBackend):
    """PaginatedDocumentBackend.

    A backend designed for handling multi-page documents (like PDFs or TIFFs)
    that require page-count awareness and page-by-page processing.
    """

    @abstractmethod
    def page_count(self) -> int:
        pass


class DeclarativeDocumentBackend(AbstractDocumentBackend):
    """DeclarativeDocumentBackend.

    A declarative document backend is a backend that can transform to DoclingDocument
    straight without a recognition pipeline.
    """

    @abstractmethod
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[BackendOptions] = None,
    ) -> None:
        if options is None:
            options = DeclarativeBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

    @abstractmethod
    def convert(self) -> DoclingDocument:
        pass

    def load_media(self, location: str) -> Optional[bytes]:
        """Return the bytes of a video or audio file that the document plays.

        A backend that records media on pictures (see
        ``docling.backend.utils.media``) overrides this, so that a pipeline can
        convert the media. The default knows no media and returns None.

        Args:
            location: The value of a ``docling__video`` or ``docling__audio``
                meta field: a path inside the document package, or a link.

        Raises:
            OperationNotAllowed: If the file is linked and the backend options
                do not allow fetching it.
        """
        return None
