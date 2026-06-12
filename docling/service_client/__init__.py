"""Client SDK for interacting with docling-serve."""

from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    BatchSourceRequestItem,
    S3SourceRequest,
)
from docling.datamodel.service.responses import (
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
)
from docling.datamodel.service.targets import PresignedUrlTarget, S3Target
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
from docling.service_client.job import ConversionJob

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONCURRENCY_LIMIT",
    "AnyHttpSourceRequest",
    "BatchSourceRequestItem",
    "ChunkerKind",
    "ConversionError",
    "ConversionItem",
    "ConversionJob",
    "DoclingServiceClient",
    "DoclingServiceClientError",
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
