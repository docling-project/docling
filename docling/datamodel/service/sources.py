import base64
from io import BytesIO
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, HttpUrl, SecretStr, StrictStr, model_validator

# HttpSource lives in the core datamodel so DocumentConverter can accept it as an
# input source; re-exported here to keep the service-layer import path stable.
from docling.datamodel.base_models import DocumentStream, HttpSource

__all__ = [
    "AzureBlobCoordinates",
    "FileSource",
    "GoogleCloudStorageCoordinates",
    "GoogleCloudStorageServiceAccountInfo",
    "GoogleDriveCoordinates",
    "GoogleDriveCredentials",
    "HttpSource",
    "S3Coordinates",
]


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


class AzureBlobCoordinates(BaseModel):
    account_name: Annotated[
        str,
        Field(description="Azure Storage account name"),
    ]

    container: Annotated[
        str,
        Field(description="Azure Blob container name"),
    ]

    connection_string: Annotated[
        str,
        Field(description="Azure Storage connection string for authentication"),
    ]

    blob_prefix: Annotated[
        str,
        Field(default="", description="Prefix for blob names"),
    ] = ""

    max_num_elements: Annotated[
        int | None,
        Field(default=None, description="Maximum number of blobs to process"),
    ] = None


class GoogleCloudStorageServiceAccountInfo(BaseModel):
    project_id: Annotated[
        StrictStr,
        Field(
            description="GCP project ID the service account belongs to.",
            examples=["my-gcp-project"],
        ),
    ]

    private_key_id: Annotated[
        SecretStr,
        Field(description="Key ID of the private key.", examples=["2aca9ed8..."]),
    ]

    private_key: Annotated[
        SecretStr,
        Field(
            description="RSA private key in PEM format.",
            examples=[
                "-----BEGIN PRIVATE KEY-----\nMIIEv...\n-----END PRIVATE KEY-----\n"
            ],
        ),
    ]

    client_email: Annotated[
        SecretStr,
        Field(
            description="Service account email address.",
            examples=["my-sa@my-gcp-project.iam.gserviceaccount.com"],
        ),
    ]

    client_id: Annotated[
        SecretStr,
        Field(
            description="Numeric client ID of the service account.",
            examples=["111274397167470984881"],
        ),
    ]

    auth_uri: Annotated[
        StrictStr,
        Field(
            description="OAuth 2.0 authorization endpoint.",
            examples=["https://accounts.google.com/o/oauth2/auth"],
        ),
    ]

    token_uri: Annotated[
        StrictStr,
        Field(
            description="OAuth 2.0 token endpoint.",
            examples=["https://oauth2.googleapis.com/token"],
        ),
    ]

    auth_provider_x509_cert_url: Annotated[
        StrictStr,
        Field(
            description="X.509 certificate URL for the auth provider.",
            examples=["https://www.googleapis.com/oauth2/v1/certs"],
        ),
    ]

    client_x509_cert_url: Annotated[
        SecretStr,
        Field(
            description="X.509 certificate URL for the service account.",
            examples=["https://www.googleapis.com/robot/v1/metadata/x509/my-sa%40..."],
        ),
    ]

    universe_domain: Annotated[
        StrictStr,
        Field(
            description="Google Cloud universe domain.",
            examples=["googleapis.com"],
        ),
    ]


class GoogleCloudStorageCoordinates(BaseModel):
    bucket: Annotated[
        StrictStr,
        Field(
            description="GCS bucket name.",
            examples=["my-docling-bucket"],
        ),
    ]

    key_prefix: Annotated[
        str,
        Field(
            description="Object key prefix for traversal (sources) and output (target); defaults to bucket root.",
            examples=["incoming/docs/", "processed/"],
        ),
    ] = ""

    max_num_elements: Annotated[
        Optional[int],
        Field(
            default=None,
            description=(
                "Maximum number of GCS objects to iterate for this source"
                "Optional, defaults to no limit"
            ),
            ge=1,
        ),
    ]

    project: Annotated[
        Optional[StrictStr],
        Field(
            default=None,
            description="GCP project ID. Optional (billing / ADC project).",
            examples=["my-gcp-project"],
        ),
    ] = None

    service_account_key: Annotated[
        Optional[GoogleCloudStorageServiceAccountInfo],
        Field(
            default=None,
            description=(
                "Service account credentials. Optional; omit to use Application Default "
                "Credentials / Workload Identity (e.g. on GKE or Cloud Run).To create a key: "
                "GCP console -> IAM & Admin -> Service Accounts -> Keys -> Add Key -> JSON "
                "(the 'type' field is omitted as it is always 'service_account')."
            ),
        ),
    ] = None


class GoogleDriveCredentials(BaseModel):
    client_id: Annotated[
        StrictStr,
        Field(
            description="OAuth 2.0 Client ID issued by Google. Required.",
        ),
    ]

    project_id: Annotated[
        StrictStr,
        Field(
            description="Google Cloud project ID associated with the OAuth client. Required.",
            examples=["docling-test-473014"],
        ),
    ]

    auth_uri: Annotated[
        HttpUrl,
        Field(
            description="Authorization endpoint URI. Required.",
            examples=["https://accounts.google.com/o/oauth2/auth"],
        ),
    ]

    token_uri: Annotated[
        HttpUrl,
        Field(
            description="Token endpoint URI. Required.",
            examples=["https://oauth2.googleapis.com/token"],
        ),
    ]

    auth_provider_x509_cert_url: Annotated[
        HttpUrl,
        Field(
            description="Certs URL for Google's OAuth provider. Required.",
            examples=["https://www.googleapis.com/oauth2/v1/certs"],
        ),
    ]

    client_secret: Annotated[
        SecretStr,
        Field(
            description="OAuth 2.0 client secret. Required.",
        ),
    ]

    redirect_uris: Annotated[
        List[HttpUrl],
        Field(
            description="OAuth 2.0 redirect URIs. Required.",
            examples=[["http://localhost"]],
        ),
    ]


class GoogleDriveCoordinates(BaseModel):
    path_id: Annotated[
        StrictStr,
        Field(
            description=(
                "Identifier for a file or folder in Google Drive. It can be obtained from the URL as follows:"
                "Folder: https://drive.google.com/drive/u/0/folders/1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5 -> folder id is 1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5"
                "File: https://docs.google.com/document/d/1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw/edit -> document id is 1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw."
                "Required."
            ),
            examples=[
                "11hgbUnDr-fyX4Hsi3T2q3xvYimvkOrfN",
            ],
        ),
    ]

    token_path: Annotated[
        Optional[StrictStr],
        Field(
            default=None,
            description=(
                "Path to save the OAuth 2.0 access token, which is generated on the fly. One of 'token_path' or 'refresh_token' is required."
            ),
            examples=[
                "./dev/google_drive_token.json",
            ],
        ),
    ]

    refresh_token: Annotated[
        Optional[StrictStr],
        Field(
            default=None,
            description=(
                "Refresh token for the OAuth 2.0 access, if already pre-generated. One of 'token_path' or 'refresh_token' is required."
            ),
        ),
    ]

    credentials_path: Annotated[
        Optional[StrictStr],
        Field(
            default=None,
            description=(
                "Path to the OAuth 2.0 Client ID credentials (available in Google Cloud console). One of 'credentials_path' or 'credentials' is required."
            ),
            examples=[
                "./dev/google_drive_credentials.json",
            ],
        ),
    ]

    credentials: Annotated[
        Optional[GoogleDriveCredentials],
        Field(
            default=None,
            description="OAuth 2.0 Client ID' credentials (available in Google Cloud console). One of 'credentials_path' or 'credentials' is required.",
        ),
    ]

    @model_validator(mode="after")
    def validate_auth_inputs(self) -> "GoogleDriveCoordinates":
        if not (self.token_path or self.refresh_token):
            raise ValueError("One of 'token_path' or 'refresh_token' is required.")
        if not (self.credentials_path or self.credentials):
            raise ValueError("One of 'credentials_path' or 'credentials' is required.")
        return self
