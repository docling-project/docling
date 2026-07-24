import enum
from collections.abc import Mapping
from functools import cache
from typing import Annotated, Any, Generic, Literal, TypeAlias, get_args

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)
from typing_extensions import TypeVar

from docling.datamodel.service.callbacks import CallbackSpec
from docling.datamodel.service.chunking import BaseChunkerOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.sources import (
    AzureBlobCoordinates,
    FileSource,
    GoogleCloudStorageCoordinates,
    GoogleDriveCoordinates,
    HttpSource,
    S3Coordinates,
)
from docling.datamodel.service.targets import (
    AzureBlobTarget,
    GoogleCloudStorageTarget,
    GoogleDriveTarget,
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


class AzureBlobSourceRequest(AzureBlobCoordinates):
    kind: Literal["azure_blob"] = "azure_blob"


class GoogleCloudStorageSourceRequest(GoogleCloudStorageCoordinates):
    kind: Literal["google_cloud_storage"] = "google_cloud_storage"


class GoogleDriveSourceRequest(GoogleDriveCoordinates):
    kind: Literal["google_drive"] = "google_drive"


## Multipart targets
class TargetName(str, enum.Enum):
    INBODY = InBodyTarget().kind
    PRESIGNED_URL = PresignedUrlTarget().kind
    ZIP = ZipTarget().kind


## Aliases
KnownBatchSourceRequestItem = Annotated[
    AnyHttpSourceRequest
    | S3SourceRequest
    | AzureBlobSourceRequest
    | GoogleCloudStorageSourceRequest
    | GoogleDriveSourceRequest,
    Field(discriminator="kind"),
]


class GenericSourceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = Field(min_length=1)


_KNOWN_BATCH_SOURCE_MODELS = {
    source_type.model_fields["kind"].default: source_type
    for source_type in (
        *get_args(get_args(KnownBatchSourceRequestItem)[0]),
        FileSourceRequest,
    )
}
_KNOWN_BATCH_SOURCE_TYPES = tuple(_KNOWN_BATCH_SOURCE_MODELS.values())


def _validate_batch_source(value: Any) -> Any:
    if isinstance(value, _KNOWN_BATCH_SOURCE_TYPES):
        return value
    if isinstance(value, BaseModel):
        payload = value.model_dump()
    elif isinstance(value, Mapping):
        payload = value
    else:
        return value

    kind = payload.get("kind")
    source_type = (
        _KNOWN_BATCH_SOURCE_MODELS.get(kind) if isinstance(kind, str) else None
    )
    return source_type.model_validate(payload) if source_type is not None else value


BatchSourceRequestItem = Annotated[
    KnownBatchSourceRequestItem | GenericSourceRequest,
    BeforeValidator(_validate_batch_source),
]
BatchSourceRequestInput: TypeAlias = BatchSourceRequestItem | Mapping[str, Any]

SourceRequestItem = Annotated[
    FileSourceRequest | HttpSourceRequest, Field(discriminator="kind")
]

TargetRequest = Annotated[
    InBodyTarget
    | ZipTarget
    | S3Target
    | AzureBlobTarget
    | GoogleCloudStorageTarget
    | GoogleDriveTarget
    | PutTarget
    | PresignedUrlTarget,
    Field(discriminator="kind"),
]

KnownBatchTargetRequest = Annotated[
    S3Target
    | AzureBlobTarget
    | GoogleCloudStorageTarget
    | GoogleDriveTarget
    | PresignedUrlTarget,
    Field(discriminator="kind"),
]


class GenericTargetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = Field(min_length=1)


_KNOWN_TARGET_MODELS = {
    target_type.model_fields["kind"].default: target_type
    for target_type in get_args(get_args(TargetRequest)[0])
}
_KNOWN_TARGET_TYPES = tuple(_KNOWN_TARGET_MODELS.values())


def _validate_batch_target(value: Any) -> Any:
    if isinstance(value, _KNOWN_TARGET_TYPES):
        return value
    if isinstance(value, BaseModel):
        payload = value.model_dump()
    elif isinstance(value, Mapping):
        payload = value
    else:
        return value

    kind = payload.get("kind")
    target_type = _KNOWN_TARGET_MODELS.get(kind) if isinstance(kind, str) else None
    return target_type.model_validate(payload) if target_type is not None else value


BatchTargetRequest = Annotated[
    KnownBatchTargetRequest | GenericTargetRequest,
    BeforeValidator(_validate_batch_target),
]
BatchTargetRequestInput: TypeAlias = BatchTargetRequest | Mapping[str, Any]


## Complete Source request
class BatchConvertSourcesRequest(BaseModel):
    options: ConvertDocumentsOptions = ConvertDocumentsOptions()
    sources: list[BatchSourceRequestItem] = Field(min_length=1)
    # Singular convenience alias — normalised to targets=[target] downstream.
    # Mutually exclusive with targets; neither is deprecated.
    target: BatchTargetRequest | None = None
    targets: list[BatchTargetRequest] | None = None
    callbacks: list[CallbackSpec] = []


class ConvertSourcesRequest(BaseModel):
    options: ConvertDocumentsOptions = ConvertDocumentsOptions()
    sources: list[SourceRequestItem] = Field(min_length=1)
    target: TargetRequest = PresignedUrlTarget()
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
