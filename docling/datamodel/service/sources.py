import base64
from io import BytesIO
from typing import Annotated

from pydantic import BaseModel, Field, StrictStr

# HttpSource lives in the core datamodel so DocumentConverter can accept it as an
# input source; re-exported here to keep the service-layer import path stable.
from docling.datamodel.base_models import DocumentStream, HttpSource

__all__ = ["FileSource", "HttpSource", "S3Coordinates"]


class FileSource(BaseModel):
    base64_string: Annotated[
        str,
        Field(
            description="Content of the file serialized in base64. "
            "For example it can be obtained via "
            "`base64 -w 0 /path/to/file/pdf-to-convert.pdf`."
        ),
    ]
    filename: Annotated[
        str,
        Field(description="Filename of the uploaded document", examples=["file.pdf"]),
    ]

    def to_document_stream(self) -> DocumentStream:
        buf = BytesIO(base64.b64decode(self.base64_string))
        return DocumentStream(stream=buf, name=self.filename)


class S3Coordinates(BaseModel):
    endpoint: Annotated[
        StrictStr,
        Field(
            description=("S3 service endpoint, without protocol. Required."),
            examples=[
                "s3.eu-de.cloud-object-storage.appdomain.cloud",
                "s3.us-east-2.amazonaws.com ",
            ],
        ),
    ]

    verify_ssl: Annotated[
        bool,
        Field(
            description=(
                "If enabled, SSL will be used to connect to s3. "
                "Boolean. Optional, defaults to true"
            ),
        ),
    ] = True

    access_key: Annotated[
        StrictStr,
        Field(
            description=("S3 access key. Required."),
        ),
    ]

    secret_key: Annotated[
        StrictStr,
        Field(
            description=("S3 secret key. Required."),
        ),
    ]

    bucket: Annotated[
        str,
        Field(
            description=("S3 bucket name. Required."),
        ),
    ]

    key_prefix: Annotated[
        str,
        Field(
            description=(
                "Prefix for the object keys on s3. Optional, defaults to empty."
            ),
        ),
    ] = ""

    max_num_elements: Annotated[
        int | None,
        Field(
            description=(
                "Maximum number of S3 objects to iterate for this source. "
                "Optional, defaults to no limit."
            ),
            ge=1,
        ),
    ] = None
