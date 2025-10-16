from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

from docling_core.types.doc import DoclingDocument
from pydantic import AnyUrl, BaseModel, Field

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument


class BaseBackendOptions(BaseModel):
    """Common options for all declarative document backends.

    This is placeholder to define all common options among declarative backends.
    """


class DeclarativeBackendOptions(BaseBackendOptions):
    """Default backend options for a declarative document backend."""

    kind: Literal["declarative"] = "declarative"


class HTMLBackendOptions(BaseBackendOptions):
    """Options specific to the HTML backend.

    This class can be extended to include options specific to HTML processing.
    """

    kind: Literal["html"] = "html"
    image_fetch: bool = Field(
        False,
        description=(
            "Whether the backend should access remote or local resources to parse "
            "images in an HTML document."
        ),
    )
    source_location: Optional[Union[AnyUrl, PurePath]] = Field(
        None,
        description=(
            "The URL that originates the HTML document. If provided, the backend "
            "will use it to resolve relative paths in the HTML document."
        ),
    )


BackendOptions = Annotated[
    Union[DeclarativeBackendOptions, HTMLBackendOptions], Field(discriminator="kind")
]


class AbstractDocumentBackend(ABC):
    enable_remote_fetch: bool = False
    enable_local_fetch: bool = False

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
    def supported_formats(cls) -> set["InputFormat"]:
        pass


class PaginatedDocumentBackend(AbstractDocumentBackend):
    """DeclarativeDocumentBackend.

    A declarative document backend is a backend that can transform to DoclingDocument
    straight without a recognition pipeline.
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
        options: BackendOptions = DeclarativeBackendOptions(),
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        self.options: BackendOptions = options

    @abstractmethod
    def convert(self) -> DoclingDocument:
        pass

    @classmethod
    def get_default_options(cls) -> BackendOptions:
        return DeclarativeBackendOptions()
