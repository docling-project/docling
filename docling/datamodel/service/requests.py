import enum
from functools import cache
from typing import Annotated, Generic, Literal

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator
from typing_extensions import TypeVar

from docling.datamodel.service.callbacks import CallbackSpec
from docling.datamodel.service.chunking import BaseChunkerOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.sources import FileSource, HttpSource, S3Coordinates
from docling.datamodel.service.targets import (
    ChunkTarget,
    InBodyTarget,
    PresignedUrlTarget,
    PutTarget,
    S3Target,
    ZipTarget,
)

## Sources


class FileSourceRequest(FileSource):
    kind: Literal["file"] = "file"


class AnyHttpSourceRequest(HttpSource):
    kind: Literal["http"] = "http"


class HttpSourceRequest(AnyHttpSourceRequest):
    """HTTP source for convert endpoints — rejects ZIP URLs."""

    @field_validator("url")
    @classmethod
    def reject_zip_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        path = str(value).lower().split("?", maxsplit=1)[0]
        if path.endswith(".zip"):
            raise ValueError("ZIP URLs are not accepted on the convert endpoint")
        return value


class S3SourceRequest(S3Coordinates):
    kind: Literal["s3"] = "s3"


## Multipart targets
class TargetName(str, enum.Enum):
    INBODY = InBodyTarget().kind
    PRESIGNED_URL = PresignedUrlTarget().kind
    ZIP = ZipTarget().kind


## Aliases
BatchSourceRequestItem = Annotated[
    AnyHttpSourceRequest | S3SourceRequest,
    Field(discriminator="kind"),
]

SourceRequestItem = Annotated[
    FileSourceRequest | HttpSourceRequest, Field(discriminator="kind")
]

TargetRequest = Annotated[
    InBodyTarget | ZipTarget | S3Target | PutTarget | PresignedUrlTarget,
    Field(discriminator="kind"),
]

BatchTargetRequest = Annotated[
    S3Target | PresignedUrlTarget,
    Field(discriminator="kind"),
]


## Complete Source request
class BatchConvertSourcesRequest(BaseModel):
    options: ConvertDocumentsOptions = ConvertDocumentsOptions()
    sources: list[BatchSourceRequestItem] = Field(min_length=1)
    target: BatchTargetRequest
    chunk_target: ChunkTarget | None = None
    callbacks: list[CallbackSpec] = []


class ConvertSourcesRequest(BaseModel):
    options: ConvertDocumentsOptions = ConvertDocumentsOptions()
    sources: list[SourceRequestItem] = Field(min_length=1)
    target: TargetRequest = PresignedUrlTarget()
    chunk_target: ChunkTarget | None = None
    callbacks: list[CallbackSpec] = []


## Deprecated aliases — will be removed in a future release
ConvertDocumentsRequest = ConvertSourcesRequest

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

    @field_validator("convert_options", mode="after")
    @classmethod
    def sync_convert_chunking_options(cls, value: ConvertDocumentsOptions):
        value.do_chunking = True
        value.chunking_options = None
        value.chunking_preset = None
        return value


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
