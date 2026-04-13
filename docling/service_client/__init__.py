"""Client SDK for interacting with docling-serve."""

from docling.service_client.client import (
    ChunkerKind,
    ConversionItem,
    DoclingServiceClient,
    HealthResponse,
    RawServiceResult,
    StatusWatcherKind,
    VersionResponse,
)
from docling.service_client.exceptions import (
    BatchConversionError,
    ConversionError,
    DoclingServiceClientError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskNotFoundError,
    TaskTimeoutError,
)
from docling.service_client.job import ConversionJob

__all__ = [
    "BatchConversionError",
    "ChunkerKind",
    "ConversionError",
    "ConversionItem",
    "ConversionJob",
    "DoclingServiceClient",
    "DoclingServiceClientError",
    "HealthResponse",
    "RawServiceResult",
    "ResultExpiredError",
    "ResultNotReadyError",
    "ServiceError",
    "ServiceUnavailableError",
    "StatusWatcherKind",
    "TaskNotFoundError",
    "TaskTimeoutError",
    "VersionResponse",
]
