# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Client SDK for interacting with docling-serve."""

from docling.datamodel.extraction import (
    DocumentExtractionResult,
    ExtractionTarget,
    ExtractionTemplate,
)
from docling.datamodel.service.options import ExtractDocumentsOptions
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    BatchSourceRequestInput,
    BatchSourceRequestItem,
    BatchTargetRequestInput,
    ExtractSourceRequestInput,
    ExtractSourceRequestItem,
    ExtractSourcesRequest,
    ExtractTargetRequest,
    FileSourceRequest,
    GenericSourceRequest,
    GenericTargetRequest,
    S3SourceRequest,
)
from docling.datamodel.service.responses import (
    ExtractDocumentResponse,
    ExtractionDocumentResult,
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
    ExtractionError,
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
    "DocumentExtractionResult",
    "ExtractDocumentResponse",
    "ExtractDocumentsOptions",
    "ExtractSourceRequestInput",
    "ExtractSourceRequestItem",
    "ExtractSourcesRequest",
    "ExtractTargetRequest",
    "ExtractionDocumentResult",
    "ExtractionError",
    "ExtractionTarget",
    "ExtractionTemplate",
    "FileSourceRequest",
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
