from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from docling.datamodel.service.sources import (
    AzureBlobCoordinates,
    GoogleCloudStorageCoordinates,
    GoogleDriveCoordinates,
    S3Coordinates,
)


class InBodyTarget(BaseModel):
    kind: Literal["inbody"] = "inbody"


class ZipTarget(BaseModel):
    kind: Literal["zip"] = "zip"


class S3Target(S3Coordinates):
    kind: Literal["s3"] = "s3"


class AzureBlobTarget(AzureBlobCoordinates):
    kind: Literal["azure_blob"] = "azure_blob"


class GoogleCloudStorageTarget(GoogleCloudStorageCoordinates):
    kind: Literal["google_cloud_storage"] = "google_cloud_storage"


class GoogleDriveTarget(GoogleDriveCoordinates):
    kind: Literal["google_drive"] = "google_drive"


class PutTarget(BaseModel):
    kind: Literal["put"] = "put"
    url: AnyHttpUrl


class PresignedUrlTarget(BaseModel):
    kind: Literal["presigned_url"] = "presigned_url"


Target = Annotated[
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

# ---------------------------------------------------------------------------
# Chunk targets — where chunked output should be written.
# Known concrete targets reuse the same coordinate models as document targets.
# Unknown server-defined chunk targets are preserved as GenericChunkTarget.
# ---------------------------------------------------------------------------

KnownChunkTarget = Annotated[
    PresignedUrlTarget | ZipTarget | S3Target,
    Field(discriminator="kind"),
]


class GenericChunkTarget(BaseModel):
    """Passthrough for chunk-target kinds not known to this client version."""

    model_config = ConfigDict(extra="allow")

    kind: str = Field(min_length=1)


ChunkTarget = Annotated[
    KnownChunkTarget | GenericChunkTarget,
    Field(discriminator="kind"),
]
