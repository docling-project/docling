import enum
from functools import cache
from typing import Annotated, Generic, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from docling.datamodel.service.callbacks import CallbackSpec
from docling.datamodel.service.chunking import (
    BaseChunkerOptions,
)
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.sources import FileSource, HttpSource, S3Coordinates
from docling.datamodel.service.targets import (
    InBodyTarget,
    PutTarget,
    S3Target,
    ZipTarget,
)

## Sources


class FileSourceRequest(FileSource):
    kind: Literal["file"] = "file"


class HttpSourceRequest(HttpSource):
    kind: Literal["http"] = "http"


class S3SourceRequest(S3Coordinates):
    kind: Literal["s3"] = "s3"


## Multipart targets
class TargetName(str, enum.Enum):
    INBODY = InBodyTarget().kind
    ZIP = ZipTarget().kind


## Aliases
SourceRequestItem = Annotated[
    FileSourceRequest | HttpSourceRequest | S3SourceRequest, Field(discriminator="kind")
]

TargetRequest = Annotated[
    InBodyTarget | ZipTarget | S3Target | PutTarget,
    Field(discriminator="kind"),
]


## Complete Source request
class ConvertDocumentsRequest(BaseModel):
    options: ConvertDocumentsOptions = ConvertDocumentsOptions()
    sources: list[SourceRequestItem]
    target: TargetRequest = InBodyTarget()
    callbacks: list[CallbackSpec] = []


## Source chunking requests


class BaseChunkDocumentsRequest(BaseModel):
    convert_options: Annotated[
        ConvertDocumentsOptions, Field(description="Conversion options.")
    ] = ConvertDocumentsOptions()
    sources: Annotated[
        list[SourceRequestItem],
        Field(description="List of input document sources to process."),
    ]
    include_converted_doc: Annotated[
        bool,
        Field(
            description="If true, the output will include both the chunks and the converted document."
        ),
    ] = False
    target: Annotated[
        TargetRequest, Field(description="Specification for the type of output target.")
    ] = InBodyTarget()
    callbacks: list[CallbackSpec] = []


ChunkingOptT = TypeVar("ChunkingOptT", bound=BaseChunkerOptions)


class GenericChunkDocumentsRequest(BaseChunkDocumentsRequest, Generic[ChunkingOptT]):
    chunking_options: ChunkingOptT


@cache
def make_request_model(
    opt_type: type[ChunkingOptT],
) -> type[GenericChunkDocumentsRequest[ChunkingOptT]]:
    """
    Dynamically create (and cache) a subclass of GenericChunkDocumentsRequest[opt_type]
    with chunking_options having a default factory.
    """
    return type(
        f"{opt_type.__name__}DocumentsRequest",
        (GenericChunkDocumentsRequest[opt_type],),  # type: ignore[valid-type]
        {
            "__annotations__": {"chunking_options": opt_type},
            "chunking_options": Field(
                default_factory=opt_type, description="Options specific to the chunker."
            ),
        },
    )
