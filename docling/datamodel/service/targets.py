from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, Field

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
