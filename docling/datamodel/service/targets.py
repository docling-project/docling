from typing import Annotated, Any, Dict, Literal

from pydantic import AnyHttpUrl, BaseModel, Field

from docling.datamodel.service.sources import S3Coordinates


class InBodyTarget(BaseModel):
    kind: Literal["inbody"] = "inbody"


class ZipTarget(BaseModel):
    kind: Literal["zip"] = "zip"


class S3Target(S3Coordinates):
    kind: Literal["s3"] = "s3"


class PutTarget(BaseModel):
    kind: Literal["put"] = "put"
    url: AnyHttpUrl


class PresignedUrlTarget(BaseModel):
    kind: Literal["presigned_url"] = "presigned_url"


Target = Annotated[
    InBodyTarget | ZipTarget | S3Target | PutTarget | PresignedUrlTarget,
    Field(discriminator="kind"),
]

ChunkTarget = Annotated[
    PresignedUrlTarget | S3Target | ZipTarget | Dict[str, Any],
    Field(),
]
