"""Shared config and pure helpers for the docling-serve clients.

Both the synchronous `DoclingServiceClient` and the asynchronous
`AsyncDoclingServiceClient` inherit from `_BaseClient`. Pure helpers that
do not perform HTTP I/O (URL building, source description, options
resolution, retry/error decisions, result assembly) live here so that the
two clients share a single source of truth for their behavior.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, TypeAlias
from urllib.parse import urlencode, urlparse

import httpx
from docling_core.types.doc import DoclingDocument
from docling_core.types.io import DocumentStream
from pydantic import ValidationError

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import AssembledUnit, ConversionResult, InputDocument
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.responses import (
    ConvertDocumentResponse,
    TaskStatusResponse,
    UsageLimitExceededResponse,
)
from docling.datamodel.settings import DocumentLimits, PageRange
from docling.service_client.exceptions import (
    ConversionError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskNotFoundError,
    UsageLimitExceededError,
)
from docling.service_client.watchers import is_terminal_task_status

SourceType: TypeAlias = Path | str | DocumentStream
logger = logging.getLogger(__name__)


class ExperimentalWarning(UserWarning):
    """Warning emitted by experimental features."""


SUCCESS_CONVERSION_STATUSES: set[ConversionStatus] = {
    ConversionStatus.SUCCESS,
    ConversionStatus.PARTIAL_SUCCESS,
}
DEFAULT_MAX_CONCURRENCY = 8
MAX_CONCURRENCY_LIMIT = 512
SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS = 64
HTTP_RETRY_BACKOFF_BASE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class RawServiceResult:
    """Typed wrapper for non-JSON result payloads."""

    content: bytes
    content_type: str
    filename: str | None = None


@dataclass(slots=True)
class ConversionItem:
    source: SourceType
    options: ConvertDocumentsRequestOptions | None = None
    headers: dict[str, str] | None = None
    source_headers: dict[str, str] | None = None
    metadata: Any = None


@dataclass(slots=True)
class _ResolvedOptions:
    options: ConvertDocumentsRequestOptions
    limits: DocumentLimits


@dataclass(slots=True)
class _SourceDescriptor:
    source_name: str
    input_format: InputFormat
    file_size: int | None


class StatusWatcherKind(str, Enum):
    WEBSOCKET = "websocket"
    POLLING = "polling"


class ChunkerKind(str, Enum):
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class _BaseClient:
    """Shared config and pure helpers for docling-serve clients."""

    def __init__(
        self,
        url: str,
        api_key: str = "",
        options: ConvertDocumentsRequestOptions | None = None,
        status_watcher: StatusWatcherKind = StatusWatcherKind.WEBSOCKET,
        ws_fallback_to_poll: bool = True,
        poll_server_wait: float = 5.0,
        poll_client_interval: float | None = None,
        job_timeout: float = 300.0,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        http_retries: int = 3,
        http_connect_timeout: float = 10.0,
        http_read_timeout: float = 60.0,
    ) -> None:
        self._base_url = self._normalize_base_url(url)
        self._api_key = api_key
        self._extension_to_format = self._build_extension_to_format_map()
        self._default_options = (
            options.model_copy(deep=True)
            if options is not None
            else ConvertDocumentsRequestOptions()
        )
        self._status_watcher_kind = status_watcher
        self._ws_fallback_to_poll = ws_fallback_to_poll
        self._poll_server_wait = poll_server_wait
        self._poll_client_interval = (
            poll_server_wait if poll_client_interval is None else poll_client_interval
        )
        self._job_timeout = job_timeout
        self._max_concurrency = self._validate_concurrency(
            max_concurrency, name="max_concurrency"
        )
        self._http_retries = http_retries
        self._http_connect_timeout = http_connect_timeout
        self._http_read_timeout = http_read_timeout

    # --- options resolution --------------------------------------------------

    def _resolve_options(
        self,
        options: ConvertDocumentsRequestOptions | None,
        max_num_pages: int | None,
        max_file_size: int | None,
        page_range: PageRange | None,
    ) -> _ResolvedOptions:
        merged = self._default_options.model_copy(deep=True)
        if options is not None and options.model_fields_set:
            # Only override fields explicitly set by the caller, preserving client defaults
            # for everything else. Using model_fields_set (vs exclude_none) means callers
            # can explicitly set a field to None to clear a client default.
            explicit = {
                field: getattr(options, field) for field in options.model_fields_set
            }
            merged = merged.model_copy(update=explicit, deep=True)

        effective_range = merged.page_range if page_range is None else page_range
        if max_num_pages is not None:
            effective_range = (
                effective_range[0],
                min(effective_range[1], max_num_pages),
            )
        merged.page_range = effective_range

        limits = DocumentLimits(
            max_num_pages=sys.maxsize if max_num_pages is None else max_num_pages,
            max_file_size=sys.maxsize if max_file_size is None else max_file_size,
            page_range=effective_range,
        )
        return _ResolvedOptions(options=merged, limits=limits)

    def _options_for_target_format(
        self,
        options: ConvertDocumentsRequestOptions,
        target_format: OutputFormat | None,
    ) -> tuple[ConvertDocumentsRequestOptions, bool]:
        if target_format is None or target_format == OutputFormat.JSON:
            formats = list(options.to_formats)
            if OutputFormat.JSON not in formats:
                formats.append(OutputFormat.JSON)
            return options.model_copy(update={"to_formats": formats}, deep=True), False
        return options.model_copy(
            update={"to_formats": [target_format]}, deep=True
        ), True

    @staticmethod
    def _validate_concurrency(value: int, *, name: str) -> int:
        if value < 1 or value > MAX_CONCURRENCY_LIMIT:
            raise ValueError(
                f"{name} must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}."
            )
        return value

    def _effective_concurrency(self, override: int | None) -> int:
        if override is None:
            return self._max_concurrency
        return self._validate_concurrency(override, name="max_concurrency")

    # --- source description --------------------------------------------------

    def _describe_source(self, source: SourceType) -> _SourceDescriptor:
        if isinstance(source, Path):
            return _SourceDescriptor(
                source_name=source.name,
                input_format=self._guess_input_format(source.name),
                file_size=source.stat().st_size,
            )
        if isinstance(source, DocumentStream):
            return _SourceDescriptor(
                source_name=source.name,
                input_format=self._guess_input_format(source.name),
                file_size=len(source.stream.getbuffer()),
            )

        parsed = urlparse(source)
        filename = Path(parsed.path).name if parsed.path else "document"
        return _SourceDescriptor(
            source_name=filename,
            input_format=self._guess_input_format(filename),
            file_size=None,
        )

    def _source_name(self, source: SourceType) -> str:
        return self._describe_source(source).source_name

    def _guess_input_format(self, name: str) -> InputFormat:
        lowered = name.lower()
        extension = (
            "tar.gz" if lowered.endswith(".tar.gz") else Path(name).suffix[1:].lower()
        )
        if extension in self._extension_to_format:
            return self._extension_to_format[extension]
        return InputFormat.PDF

    def _validate_http_source(self, source: str) -> None:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("String sources must be HTTP or HTTPS URLs.")

    def _build_extension_to_format_map(self) -> dict[str, InputFormat]:
        extension_to_format: dict[str, InputFormat] = {}
        for input_format, extensions in FormatToExtensions.items():
            for extension in extensions:
                extension_to_format.setdefault(extension.lower(), input_format)
        return extension_to_format

    # --- result assembly -----------------------------------------------------

    def _preflight_limits(
        self,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
    ) -> ConversionResult | None:
        if limits.max_file_size >= sys.maxsize:
            return None

        size = descriptor.file_size
        if size is None or size <= limits.max_file_size:
            return None

        message = f"Input size {size} exceeds max_file_size limit {limits.max_file_size} bytes."
        return self._build_failed_conversion_result(
            descriptor=descriptor,
            limits=limits,
            error_message=message,
            status=ConversionStatus.SKIPPED,
        )

    def _build_conversion_result(
        self,
        payload: ConvertDocumentResponse,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
    ) -> ConversionResult:
        source_name = payload.document.filename or descriptor.source_name
        input_doc = self._build_input_document(
            source_name=source_name,
            input_format=descriptor.input_format,
            file_size=descriptor.file_size,
            limits=limits,
        )
        document = payload.document.json_content
        if document is None:
            document = DoclingDocument(name=Path(source_name).stem)

        return ConversionResult(
            input=input_doc,
            assembled=AssembledUnit(),
            status=payload.status,
            errors=payload.errors,
            timings=payload.timings,
            document=document,
        )

    def _build_failed_conversion_result(
        self,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        error_message: str,
        status: ConversionStatus,
    ) -> ConversionResult:
        input_doc = self._build_input_document(
            source_name=descriptor.source_name,
            input_format=descriptor.input_format,
            file_size=descriptor.file_size,
            limits=limits,
        )
        error = ErrorItem(
            component_type=DoclingComponentType.USER_INPUT,
            module_name="docling.service_client",
            error_message=error_message,
        )
        return ConversionResult(
            input=input_doc,
            assembled=AssembledUnit(),
            status=status,
            errors=[error],
            document=DoclingDocument(name=Path(descriptor.source_name).stem),
        )

    def _build_input_document(
        self,
        source_name: str,
        input_format: InputFormat,
        file_size: int | None,
        limits: DocumentLimits,
    ) -> InputDocument:
        # Build a lightweight InputDocument for compatibility results without
        # invoking expensive parsing backends.
        input_doc = InputDocument(
            path_or_stream=BytesIO(b"x"),
            format=input_format,
            backend=NoOpBackend,
            filename=source_name,
            limits=limits,
        )
        input_doc.file = PurePath(source_name)
        input_doc.document_hash = source_name
        input_doc.filesize = file_size
        input_doc.page_count = 0
        return input_doc

    def _decode_raw_result(self, response: httpx.Response) -> RawServiceResult:
        content_type = response.headers.get("content-type", "application/octet-stream")
        filename = self._filename_from_headers(response.headers)
        return RawServiceResult(
            content=response.content,
            content_type=content_type,
            filename=filename,
        )

    def _filename_from_headers(self, headers: httpx.Headers) -> str | None:
        disposition = headers.get("content-disposition")
        if disposition is None:
            return None
        match = re.search(r'filename="?(?P<name>[^";]+)"?', disposition)
        if match is None:
            return None
        return match.group("name")

    def _failure_message(self, result: ConversionResult) -> str:
        if result.errors:
            messages = "; ".join(item.error_message for item in result.errors)
            return (
                f"Conversion failed for {result.input.file} with status "
                f"{result.status.value}. Errors: {messages}"
            )
        return f"Conversion failed for {result.input.file} with status {result.status.value}."

    # --- retry policy --------------------------------------------------------

    def _check_retry(
        self,
        response: httpx.Response,
        attempt: int,
        max_retries: int,
    ) -> tuple[httpx.Response | None, float]:
        """Return (response, 0.0) to yield, (None, delay) to retry after delay, or raise."""
        if response.status_code == 500:
            return self._retry_with_exponential_backoff(
                response=response,
                attempt=attempt,
                max_retries=max_retries,
                error_message="Service returned HTTP 500 after retries.",
            )
        if response.status_code in {429, 503}:
            return self._retry_with_retry_after_header(
                response=response,
                attempt=attempt,
                max_retries=max_retries,
            )
        return response, 0.0

    def _retry_with_exponential_backoff(
        self,
        response: httpx.Response,
        attempt: int,
        max_retries: int,
        error_message: str,
    ) -> tuple[httpx.Response | None, float]:
        if attempt < max_retries:
            return None, self._exponential_backoff_delay(attempt)
        raise ServiceUnavailableError(
            error_message,
            status_code=response.status_code,
            detail=self._http_error_detail(response),
        )

    def _retry_with_retry_after_header(
        self,
        response: httpx.Response,
        attempt: int,
        max_retries: int,
    ) -> tuple[httpx.Response | None, float]:
        retry_after_delay = self._retry_after_delay_seconds(response)
        if retry_after_delay is None:
            return response, 0.0
        if attempt < max_retries:
            return None, retry_after_delay
        raise ServiceUnavailableError(
            f"Service returned HTTP {response.status_code} after retries.",
            status_code=response.status_code,
            detail=self._http_error_detail(response),
        )

    def _exponential_backoff_delay(self, attempt: int) -> float:
        return HTTP_RETRY_BACKOFF_BASE_SECONDS * (2**attempt)

    def _retry_after_delay_seconds(self, response: httpx.Response) -> float | None:
        retry_after_header = response.headers.get("Retry-After")
        if retry_after_header is None:
            return None

        try:
            return max(0.0, float(retry_after_header))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after_header)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None

        now = datetime.now(tz=retry_at.tzinfo or timezone.utc)
        return max(0.0, (retry_at - now).total_seconds())

    # --- error mapping -------------------------------------------------------

    def _raise_for_result_404(
        self,
        task_id: str,
        response: httpx.Response,
        last_status: TaskStatusResponse | None,
    ) -> None:
        detail = self._http_error_detail(response)
        if detail == "Task not found.":
            raise TaskNotFoundError(f"Task {task_id} was not found.")
        if detail is not None and detail.startswith("Task result not found"):
            if last_status is not None and is_terminal_task_status(last_status):
                if last_status.task_status == "failure":
                    message = last_status.error_message or f"Task {task_id} failed."
                    raise ConversionError(message)
                raise ResultExpiredError(f"Result for task {task_id} has expired.")
            raise ResultNotReadyError(f"Result for task {task_id} is not ready.")
        raise ServiceError(
            "Unexpected result lookup error.",
            status_code=response.status_code,
            detail=detail,
        )

    def _raise_for_generic_http_error(
        self,
        response: httpx.Response,
        message: str,
    ) -> None:
        if response.status_code == 402:
            usage_limit = self._parse_usage_limit_exceeded_response(response)
            raise UsageLimitExceededError(
                message,
                status_code=response.status_code,
                detail=None if usage_limit is None else usage_limit.message,
                current_usage=(
                    None if usage_limit is None else usage_limit.details.currentUsage
                ),
                limit=None if usage_limit is None else usage_limit.details.limit,
            )

        detail = self._http_error_detail(response)
        if 400 <= response.status_code < 500:
            raise ServiceError(message, status_code=response.status_code, detail=detail)
        raise ServiceUnavailableError(
            message,
            status_code=response.status_code,
            detail=detail,
        )

    def _parse_usage_limit_exceeded_response(
        self,
        response: httpx.Response,
    ) -> UsageLimitExceededResponse | None:
        try:
            return UsageLimitExceededResponse.model_validate_json(response.text)
        except (ValidationError, ValueError):
            return None

    def _http_error_detail(self, response: httpx.Response) -> str | None:
        try:
            detail = response.json().get("detail")
        except Exception:
            return None
        return detail if isinstance(detail, str) else None

    # --- URL building --------------------------------------------------------

    def _url(self, path: str) -> str:
        if path.startswith("/"):
            return f"{self._base_url}{path}"
        return f"{self._base_url}/{path}"

    def _build_ws_status_url(self, task_id: str) -> str:
        parsed = urlparse(self._base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{parsed.netloc}{parsed.path}/v1/status/ws/{task_id}"
        if not self._api_key:
            return ws_url
        return f"{ws_url}?{urlencode({'api_key': self._api_key})}"

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Client URL must be an absolute http(s) base URL.")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "Client URL must not include query or fragment components."
            )
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            raise ValueError(
                "Client URL must be the service base URL, not include /v1."
            )
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    # --- misc ----------------------------------------------------------------

    @staticmethod
    def _normalize_exception(exc: BaseException) -> Exception:
        if isinstance(exc, Exception):
            return exc
        return RuntimeError(str(exc))
