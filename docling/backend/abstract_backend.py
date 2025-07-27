from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Set, TypeVar, Union

from docling_core.types.doc import DoclingDocument

from docling import backend
from docling.datamodel.backend_options import BackendOptions

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument


class AbstractDocumentBackend(ABC):
    @abstractmethod
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        self.file = in_doc.file
        self.path_or_stream = path_or_stream
        self.document_hash = in_doc.document_hash
        self.input_format = in_doc.format

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
    def supported_formats(cls) -> Set["InputFormat"]:
        pass

    @classmethod
    def get_default_options(cls) -> BackendOptions:
        return BackendOptions()


class PaginatedDocumentBackend(AbstractDocumentBackend):
    """DeclarativeDocumentBackend.

    A declarative document backend is a backend that can transform to DoclingDocument
    straight without a recognition pipeline.
    """

    @abstractmethod
    def page_count(self) -> int:
        pass


TBackendOptions = TypeVar("TBackendOptions", bound=BackendOptions)


class DeclarativeDocumentBackend(Generic[TBackendOptions], AbstractDocumentBackend):
    """DeclarativeDocumentBackend.

    A declarative document backend is a backend that can transform to DoclingDocument
    straight without a recognition pipeline.
    """

    @abstractmethod
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        backend_options: TBackendOptions,
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        self.backend_options: TBackendOptions = backend_options

    @abstractmethod
    def convert(self) -> DoclingDocument:
        pass
