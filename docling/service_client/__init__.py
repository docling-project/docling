"""Client SDK for interacting with docling-serve."""

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
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskNotFoundError,
    TaskTimeoutError,
    UsageLimitExceededError,
)
from docling.service_client.job import ConversionJob

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONCURRENCY_LIMIT",
    "ChunkerKind",
    "ConversionError",
    "ConversionItem",
    "ConversionJob",
    "DoclingServiceClient",
    "DoclingServiceClientError",
    "RawServiceResult",
    "ResultExpiredError",
    "ResultNotReadyError",
    "ServiceError",
    "ServiceUnavailableError",
    "StatusWatcherKind",
    "TaskNotFoundError",
    "TaskTimeoutError",
    "UsageLimitExceededError",
]
