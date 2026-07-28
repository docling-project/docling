"""Client SDK for interacting with docling-serve."""

from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    BatchSourceRequestInput,
    BatchSourceRequestItem,
    BatchTargetRequestInput,
    GenericSourceRequest,
    GenericTargetRequest,
    S3SourceRequest,
)
from docling.datamodel.service.responses import (
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
)
from docling.datamodel.service.targets import PresignedUrlTarget, S3Target
from docling.service_client._async_client import AsyncDoclingServiceClient
from docling.service_client.client import (
    DEFAULT_MAX_CONCURRENCY,
    MAX_CONCURRENCY_LIMIT,
    ChunkerKind,
    ConversionItem,
    DoclingServiceClient,
    RawServiceResult,
    StatusWatcherKind,
)
from docling.service_client.exceptions import (
    ArtifactDownloadError,
    ConversionError,
    DoclingServiceClientError,
    ResponseSchemaMismatchError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskExecutionError,
    TaskNotFoundError,
    TaskTimeoutError,
    UsageLimitExceededError,
)
from docling.service_client.job import AsyncConversionJob, ConversionJob

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONCURRENCY_LIMIT",
    "AnyHttpSourceRequest",
    "ArtifactDownloadError",
    "AsyncConversionJob",
    "AsyncDoclingServiceClient",
    "BatchSourceRequestInput",
    "BatchSourceRequestItem",
    "BatchTargetRequestInput",
    "ChunkerKind",
    "ConversionError",
    "ConversionItem",
    "ConversionJob",
    "DoclingServiceClient",
    "DoclingServiceClientError",
    "GenericSourceRequest",
    "GenericTargetRequest",
    "PresignedUrlConvertDocumentResponse",
    "PresignedUrlConvertResponse",
    "PresignedUrlTarget",
    "RawServiceResult",
    "ResponseSchemaMismatchError",
    "ResultExpiredError",
    "ResultNotReadyError",
    "S3SourceRequest",
    "S3Target",
    "ServiceError",
    "ServiceUnavailableError",
    "StatusWatcherKind",
    "TaskExecutionError",
    "TaskNotFoundError",
    "TaskTimeoutError",
    "UsageLimitExceededError",
]
