"""Synchronous client SDK for docling-serve."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import sys
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePath
from typing import IO, Any, TypeAlias, TypeVar
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
from docling.datamodel.service.chunking import (
    HierarchicalChunkerOptions,
    HybridChunkerOptions,
)
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import (
    BatchConvertSourcesRequest,
    BatchSourceRequestItem,
    ConvertDocumentsRequest,
    HttpSourceRequest,
)
from docling.datamodel.service.responses import (
    ChunkDocumentResponse,
    ConvertDocumentResponse,
    HealthCheckResponse,
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
    TaskStatusResponse,
    UsageLimitExceededResponse,
)
from docling.datamodel.service.targets import (
    InBodyTarget,
    PresignedUrlTarget,
    S3Target,
    ZipTarget,
)
from docling.datamodel.settings import DocumentLimits, PageRange
from docling.service_client._scheduler import _run_bounded
from docling.service_client.exceptions import (
    ConversionError,
    ResponseSchemaMismatchError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskNotFoundError,
    TaskTimeoutError,
    UsageLimitExceededError,
)
from docling.service_client.job import ConversionJob, _JobHandlers
from docling.service_client.watchers import (
    PollingWatcher,
    StatusWatcher,
    WebSocketWatcher,
    _poll_sleep_duration,
    is_terminal_task_status,
)

SourceType: TypeAlias = Path | str | DocumentStream | HttpSourceRequest
SubmitTarget: TypeAlias = InBodyTarget | ZipTarget | PresignedUrlTarget
BatchSubmitTarget: TypeAlias = S3Target | PresignedUrlTarget
logger = logging.getLogger(__name__)
_T = TypeVar("_T")


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
TRANSPORT_RETRYABLE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


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
    metadata: Any = None


@dataclass(slots=True)
class _ResolvedOptions:
    options: ConvertDocumentsRequestOptions
    limits: DocumentLimits


@dataclass(slots=True)
class _ConvertAllItemMetadata:
    source_index: int
    descriptor: _SourceDescriptor


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


class DoclingServiceClient:
    """Client for docling-serve compatibility and task APIs."""

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

        warnings.warn(
            "DoclingServiceClient is experimental and may change in future releases.",
            ExperimentalWarning,
            stacklevel=2,
        )

        timeout = httpx.Timeout(
            connect=http_connect_timeout,
            read=http_read_timeout,
            write=http_read_timeout,
            pool=http_read_timeout,
        )
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._http_client = httpx.Client(timeout=timeout, headers=headers)

        ws_headers = {"X-Api-Key": api_key} if api_key else {}
        self._polling_watcher = PollingWatcher(
            poll_status=self._poll_task_status,
            poll_server_wait=self._poll_server_wait,
            poll_client_interval=self._poll_client_interval,
            default_timeout=self._job_timeout,
        )
        self._ws_watcher = WebSocketWatcher(
            ws_url_for_task=self._build_ws_status_url,
            poll_fallback=self._polling_watcher,
            fallback_to_poll=self._ws_fallback_to_poll,
            connect_timeout=self._http_connect_timeout,
            default_timeout=self._job_timeout,
            additional_headers=ws_headers,
        )

    def __enter__(self) -> DoclingServiceClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Release HTTP resources owned by this client."""
        self._http_client.close()

    def convert(
        self,
        source: SourceType,
        headers: dict[str, str] | None = None,
        max_num_pages: int | None = None,
        max_file_size: int | None = None,
        page_range: PageRange | None = None,
        options: ConvertDocumentsRequestOptions | None = None,
        raises_on_error: bool = True,
    ) -> ConversionResult:
        resolved = self._resolve_options(
            options=options,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
        )
        result = self._convert_single(
            source=source,
            headers=headers,
            resolved=resolved,
        )
        if raises_on_error and result.status not in SUCCESS_CONVERSION_STATUSES:
            raise ConversionError(self._failure_message(result))
        return result

    def convert_all(
        self,
        sources: Iterable[SourceType],
        headers: dict[str, str] | None = None,
        max_num_pages: int | None = None,
        max_file_size: int | None = None,
        page_range: PageRange | None = None,
        options: ConvertDocumentsRequestOptions | None = None,
        max_concurrency: int | None = None,
    ) -> Iterator[ConversionResult]:
        resolved = self._resolve_options(
            options=options,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
        )
        effective_cap = self._effective_concurrency(max_concurrency)
        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=None,
            target=InBodyTarget(),
        )
        return self._iterate_convert_all_sync(
            sources=sources,
            headers=headers,
            resolved=resolved,
            submit_options=submit_options,
            in_flight=effective_cap,
        )

    def submit_and_retrieve_each(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> Iterator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ]
    ]:
        """Yield one outcome per submitted item.

        `target=None` uses the same auto-target behavior as `submit()`: the client
        first tries `PresignedUrlTarget()` and falls back to `InBodyTarget()` when the
        server rejects presigned output because artifact storage is not configured.
        """
        return self._run_submit_and_retrieve_many_async(
            item_list=items,
            max_in_flight=self._validate_concurrency(
                max_in_flight, name="max_in_flight"
            ),
            ordered=ordered,
            target=target,
        )

    def submit_and_retrieve_many(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> Iterator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ]
    ]:
        warnings.warn(
            "submit_and_retrieve_many() is deprecated; use submit_and_retrieve_each().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.submit_and_retrieve_each(
            items=items,
            max_in_flight=max_in_flight,
            ordered=ordered,
            target=target,
        )

    def chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions | None = None,
    ) -> ChunkDocumentResponse:
        job = self.submit_chunk(source=source, chunker=chunker, options=options)
        return job.result(timeout=self._job_timeout)

    def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: SubmitTarget | None = None,
    ) -> (
        ConversionJob[ConversionResult]
        | ConversionJob[RawServiceResult]
        | ConversionJob[PresignedUrlConvertResponse]
    ):
        descriptor = self._describe_source(source)
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        resolved_target = PresignedUrlTarget() if target is None else target
        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=output_formats,
            target=resolved_target,
        )
        if target is None:
            return self._submit_conversion_job_with_auto_target(
                descriptor=descriptor,
                source=source,
                options=submit_options,
                limits=resolved.limits,
                request_headers=headers,
            )
        return self._submit_conversion_job(
            source=source,
            options=submit_options,
            limits=resolved.limits,
            target=target,
            request_headers=headers,
        )

    def submit_batch(
        self,
        sources: list[BatchSourceRequestItem],
        target: BatchSubmitTarget,
        output_formats: list[OutputFormat] | None = None,
        options: ConvertDocumentsRequestOptions | None = None,
        headers: dict[str, str] | None = None,
    ) -> (
        ConversionJob[PresignedUrlConvertDocumentResponse]
        | ConversionJob[PresignedUrlConvertResponse]
    ):
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=output_formats,
            target=target,
        )
        return self._submit_batch_conversion_job(
            sources=sources,
            options=submit_options,
            target=target,
            request_headers=headers,
        )

    def submit_chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions | None = None,
    ) -> ConversionJob[ChunkDocumentResponse]:
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        initial_status = self._submit_chunk_task(
            source=source,
            chunker=chunker,
            options=resolved.options,
        )
        handlers = _JobHandlers[ChunkDocumentResponse](
            poll=self._poll_task_status,
            watch=self._watch_task_updates,
            wait=self._wait_for_terminal_status,
            fetch_result=lambda task_id, last_status: self._fetch_chunk_result(
                task_id=task_id,
                last_status=last_status,
            ),
        )
        return ConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    def health(self) -> HealthCheckResponse:
        response = self._request_with_retry("GET", "/health", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Health check request failed.")
        return HealthCheckResponse.model_validate_json(response.text)

    def version(self) -> dict[str, Any]:
        response = self._request_with_retry("GET", "/version", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Version request failed.")
        return response.json()

    def _convert_single(
        self,
        source: SourceType,
        headers: dict[str, str] | None,
        resolved: _ResolvedOptions,
    ) -> ConversionResult:
        descriptor = self._describe_source(source)
        preflight = self._preflight_limits(
            descriptor=descriptor, limits=resolved.limits
        )
        if preflight is not None:
            return preflight

        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=None,
            target=InBodyTarget(),
        )
        job = self._submit_conversion_job(
            source=source,
            options=submit_options,
            limits=resolved.limits,
            target=InBodyTarget(),
            descriptor=descriptor,
            request_headers=headers,
        )
        result = job.result(timeout=self._job_timeout)
        if not isinstance(result, ConversionResult):
            raise TypeError(
                "InBodyTarget submission returned an unexpected result type."
            )
        return result

    def _submit_conversion_job(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions,
        limits: DocumentLimits,
        target: SubmitTarget,
        descriptor: _SourceDescriptor | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> (
        ConversionJob[ConversionResult]
        | ConversionJob[RawServiceResult]
        | ConversionJob[PresignedUrlConvertResponse]
    ):
        descriptor = descriptor or self._describe_source(source)
        initial_status = self._submit_convert_task(
            source=source,
            options=options,
            target=target,
            request_headers=request_headers,
        )
        fetch_result = self._make_convert_fetch_result_handler(
            descriptor=descriptor,
            limits=limits,
            target=target,
        )
        handlers = _JobHandlers[Any](
            poll=self._poll_task_status,
            watch=self._watch_task_updates,
            wait=self._wait_for_terminal_status,
            fetch_result=fetch_result,
        )
        return ConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    def _submit_convert_task(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions,
        target: SubmitTarget,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        if isinstance(source, (str, HttpSourceRequest)):
            request_source = self._normalize_http_source(source)
            request = ConvertDocumentsRequest(
                options=options,
                sources=[request_source],
                target=target,
            )
            response = self._request_with_retry(
                method="POST",
                path="/v1/convert/source/async",
                json=request.model_dump(mode="json"),
                headers=request_headers,
            )
        else:
            files = self._source_to_upload_files(source)
            data = options.model_dump(mode="json", exclude_none=True)
            data["target_type"] = target.kind
            response = self._request_with_retry(
                method="POST",
                path="/v1/convert/file/async",
                data=data,
                files=files,
                headers=request_headers,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Task submission failed.")
        status = TaskStatusResponse.model_validate_json(response.text)
        logger.info(
            "Submitted convert task for source=%s task_id=%s status=%s position=%s",
            source_name,
            status.task_id,
            status.task_status,
            status.task_position,
        )
        return status

    def _submit_chunk_task(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions,
    ) -> TaskStatusResponse:
        if isinstance(source, (str, HttpSourceRequest)):
            request_source = self._normalize_http_source(source)
            chunking_options: HybridChunkerOptions | HierarchicalChunkerOptions
            if chunker == ChunkerKind.HYBRID:
                chunking_options = HybridChunkerOptions()
            else:
                chunking_options = HierarchicalChunkerOptions()

            payload = {
                "convert_options": options.model_dump(mode="json", exclude_none=True),
                "chunking_options": chunking_options.model_dump(
                    mode="json", exclude_none=True
                ),
                "sources": [request_source.model_dump(mode="json")],
                "include_converted_doc": False,
                "target": InBodyTarget().model_dump(mode="json"),
                "callbacks": [],
            }
            response = self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/source/async",
                json=payload,
            )
        else:
            files = self._source_to_upload_files(source)
            data: dict[str, Any] = {
                f"convert_{key}": value
                for key, value in options.model_dump(
                    mode="json", exclude_none=True
                ).items()
            }
            chunk_model: HybridChunkerOptions | HierarchicalChunkerOptions
            if chunker == ChunkerKind.HYBRID:
                chunk_model = HybridChunkerOptions()
            else:
                chunk_model = HierarchicalChunkerOptions()
            chunk_payload = chunk_model.model_dump(mode="json", exclude_none=True)
            chunk_payload.pop("chunker", None)
            data.update(
                {f"chunking_{key}": value for key, value in chunk_payload.items()}
            )
            data["include_converted_doc"] = False
            data["target_type"] = "inbody"
            response = self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/file/async",
                data=data,
                files=files,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, "Chunk task submission failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    def _poll_task_status(self, task_id: str, wait: float) -> TaskStatusResponse:
        response = self._request_with_retry(
            method="GET",
            path=f"/v1/status/poll/{task_id}",
            params={"wait": wait},
        )
        if response.status_code == 404:
            raise TaskNotFoundError(f"Task {task_id} was not found.")
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, f"Polling task {task_id} failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    def _watch_task_updates(
        self,
        task_id: str,
        timeout: float | None,
    ) -> Iterator[TaskStatusResponse]:
        watcher = self._status_watcher()
        return watcher.iter_updates(task_id=task_id, timeout=timeout)

    def _wait_for_terminal_status(
        self,
        task_id: str,
        timeout: float | None,
    ) -> TaskStatusResponse:
        watcher = self._status_watcher()
        return watcher.wait_for_terminal(task_id=task_id, timeout=timeout)

    def _status_watcher(self) -> StatusWatcher:
        if self._status_watcher_kind == "polling":
            return self._polling_watcher
        return self._ws_watcher

    def _submit_conversion_job_with_auto_target(
        self,
        descriptor: _SourceDescriptor,
        source: SourceType,
        options: ConvertDocumentsRequestOptions,
        limits: DocumentLimits,
        request_headers: dict[str, str] | None = None,
    ) -> ConversionJob[ConversionResult] | ConversionJob[PresignedUrlConvertResponse]:
        try:
            return self._submit_conversion_job(
                source=source,
                options=options,
                limits=limits,
                target=PresignedUrlTarget(),
                descriptor=descriptor,
                request_headers=request_headers,
            )
        except ServiceError as exc:
            if not self._should_fallback_from_presigned_target(exc):
                raise
        return self._submit_conversion_job(
            source=source,
            options=self._options_for_output_formats(
                options,
                output_formats=None,
                target=InBodyTarget(),
            ),
            limits=limits,
            target=InBodyTarget(),
            descriptor=descriptor,
            request_headers=request_headers,
        )

    def _make_convert_fetch_result_handler(
        self,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        target: SubmitTarget,
    ) -> Any:
        if isinstance(target, ZipTarget):
            return lambda task_id, last_status: self._fetch_raw_result(
                task_id=task_id,
                last_status=last_status,
            )
        if isinstance(target, PresignedUrlTarget):
            return lambda task_id, last_status: self._fetch_presigned_result(
                task_id=task_id,
                last_status=last_status,
            )
        return lambda task_id, last_status: self._build_conversion_result(
            payload=self._fetch_convert_result_payload(
                task_id=task_id,
                last_status=last_status,
            ),
            descriptor=descriptor,
            limits=limits,
        )

    def _submit_batch_conversion_job(
        self,
        sources: list[BatchSourceRequestItem],
        options: ConvertDocumentsRequestOptions,
        target: BatchSubmitTarget,
        request_headers: dict[str, str] | None = None,
    ) -> (
        ConversionJob[PresignedUrlConvertDocumentResponse]
        | ConversionJob[PresignedUrlConvertResponse]
    ):
        initial_status = self._submit_batch_task(
            sources=sources,
            options=options,
            target=target,
            request_headers=request_headers,
        )
        if isinstance(target, S3Target):

            def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertDocumentResponse:
                return self._fetch_presigned_document_result(
                    task_id=task_id,
                    last_status=last_status,
                )
        else:

            def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertResponse:
                return self._fetch_presigned_result(
                    task_id=task_id,
                    last_status=last_status,
                )

        handlers = _JobHandlers[Any](
            poll=self._poll_task_status,
            watch=self._watch_task_updates,
            wait=self._wait_for_terminal_status,
            fetch_result=fetch_result,
        )
        return ConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    def _submit_batch_task(
        self,
        sources: list[BatchSourceRequestItem],
        options: ConvertDocumentsRequestOptions,
        target: BatchSubmitTarget,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        request = BatchConvertSourcesRequest(
            options=options,
            sources=sources,
            target=target,
        )
        response = self._request_with_retry(
            method="POST",
            path="/v1/convert/source/batch",
            json=request.model_dump(mode="json"),
            headers=request_headers,
        )
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, "Batch task submission failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    def _fetch_result_response(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        *,
        error_message: str,
    ) -> httpx.Response:
        response = self._request_with_retry(
            method="GET",
            path=f"/v1/result/{task_id}",
        )
        if response.status_code == 404:
            self._raise_for_result_404(
                task_id=task_id, response=response, last_status=last_status
            )
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, error_message)
        return response

    def _fetch_convert_result_payload(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ConvertDocumentResponse:
        response = self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ConvertDocumentResponse)

    def _fetch_raw_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> RawServiceResult:
        response = self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._decode_raw_result(response)

    def _fetch_presigned_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> PresignedUrlConvertResponse:
        response = self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, PresignedUrlConvertResponse)

    def _fetch_presigned_document_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> PresignedUrlConvertDocumentResponse:
        response = self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(
            response, PresignedUrlConvertDocumentResponse
        )

    def _fetch_chunk_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ChunkDocumentResponse:
        response = self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching chunk result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ChunkDocumentResponse)

    def _parse_result_model_response(
        self,
        response: httpx.Response,
        model_cls: type[_T],
    ) -> _T:
        try:
            return model_cls.model_validate_json(response.text)
        except (ValidationError, ValueError) as exc:
            raise ResponseSchemaMismatchError(
                "Response schema mismatch — client and server versions may differ.",
                status_code=response.status_code,
                detail=str(exc),
            ) from exc

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

    def _options_for_output_formats(
        self,
        options: ConvertDocumentsRequestOptions,
        output_formats: list[OutputFormat] | None,
        target: InBodyTarget | ZipTarget | S3Target | PresignedUrlTarget,
    ) -> ConvertDocumentsRequestOptions:
        effective = options
        if output_formats is not None:
            effective = options.model_copy(
                update={"to_formats": list(output_formats)},
                deep=True,
            )
        if isinstance(target, InBodyTarget):
            formats = list(effective.to_formats)
            if OutputFormat.JSON not in formats:
                formats.append(OutputFormat.JSON)
            effective = effective.model_copy(update={"to_formats": formats}, deep=True)
        return effective

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

        request_source = self._normalize_http_source(source)
        parsed = urlparse(str(request_source.url))
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

    def _normalize_http_source(
        self, source: str | HttpSourceRequest
    ) -> HttpSourceRequest:
        if isinstance(source, HttpSourceRequest):
            return source
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("String sources must be HTTP or HTTPS URLs.")
        return HttpSourceRequest(url=source, headers={})

    def _source_to_upload_files(
        self,
        source: Path | DocumentStream,
    ) -> dict[str, tuple[str, IO[bytes], str]]:
        """Build multipart files dict for a sync upload. Passes file handles — no full read."""
        if isinstance(source, Path):
            filename = source.name
            content: IO[bytes] = source.open("rb")
        else:
            filename = source.name
            source.stream.seek(0)
            content = source.stream
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {"files": (filename, content, mime)}

    async def _source_to_upload_files_async(
        self,
        source: Path | DocumentStream,
    ) -> dict[str, tuple[str, IO[bytes] | bytes, str]]:
        """Build multipart files dict for an async upload.
        Path sources are read in a thread to avoid blocking the event loop.
        DocumentStream data is already in memory — passed directly.
        """
        if isinstance(source, Path):
            filename = source.name
            content: IO[bytes] | bytes = await asyncio.to_thread(source.read_bytes)
        else:
            filename = source.name
            source.stream.seek(0)
            content = source.stream
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {"files": (filename, content, mime)}

    def _iterate_convert_all_sync(
        self,
        sources: Iterable[SourceType],
        headers: dict[str, str] | None,
        resolved: _ResolvedOptions,
        submit_options: ConvertDocumentsRequestOptions,
        in_flight: int,
    ) -> Iterator[ConversionResult]:
        self._ensure_sync_bridge_allowed()
        return self._iterate_async_generator_sync(
            self._convert_all_async(
                sources=sources,
                headers=headers,
                resolved=resolved,
                submit_options=submit_options,
                in_flight=in_flight,
            )
        )

    def _run_submit_and_retrieve_many_async(
        self,
        item_list: Iterable[ConversionItem],
        max_in_flight: int,
        ordered: bool,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> Iterator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ]
    ]:
        self._ensure_sync_bridge_allowed()
        return self._iterate_submit_and_retrieve_many_sync(
            item_list=item_list,
            max_in_flight=max_in_flight,
            ordered=ordered,
            target=target,
        )

    def _iterate_submit_and_retrieve_many_sync(
        self,
        item_list: Iterable[ConversionItem],
        max_in_flight: int,
        ordered: bool,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> Iterator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ]
    ]:
        return self._iterate_async_generator_sync(
            self._submit_and_retrieve_many_async(
                item_list=item_list,
                max_in_flight=max_in_flight,
                ordered=ordered,
                target=target,
            )
        )

    def _ensure_sync_bridge_allowed(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "This method cannot run inside an active asyncio loop. "
            "Call it from synchronous code."
        )

    def _iterate_async_generator_sync(
        self, async_iterator: AsyncGenerator[_T, None]
    ) -> Iterator[_T]:
        loop = asyncio.new_event_loop()

        def iterator() -> Iterator[_T]:
            try:
                asyncio.set_event_loop(loop)
                while True:
                    try:
                        yield loop.run_until_complete(anext(async_iterator))
                    except StopAsyncIteration:
                        break
            finally:
                try:
                    loop.run_until_complete(async_iterator.aclose())
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

        return iterator()

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

    @staticmethod
    def _normalize_exception(exc: BaseException) -> Exception:
        if isinstance(exc, Exception):
            return exc
        return RuntimeError(str(exc))

    async def _convert_all_async(
        self,
        sources: Iterable[SourceType],
        headers: dict[str, str] | None,
        resolved: _ResolvedOptions,
        submit_options: ConvertDocumentsRequestOptions,
        in_flight: int,
    ) -> AsyncGenerator[ConversionResult, None]:
        results: dict[int, ConversionResult] = {}
        descriptors: dict[int, _SourceDescriptor] = {}
        errors: dict[int, Exception] = {}
        total = 0

        def result_for_index(idx: int) -> ConversionResult:
            if idx in results:
                return results[idx]

            exc = errors.get(idx)
            error_message = str(exc) if exc is not None else "Unknown conversion error."
            result = self._build_failed_conversion_result(
                descriptor=descriptors[idx],
                limits=resolved.limits,
                error_message=error_message,
                status=ConversionStatus.FAILURE,
            )
            results[idx] = result
            return result

        def make_items() -> Iterator[ConversionItem]:
            nonlocal total
            for idx, source in enumerate(sources):
                total = idx + 1
                try:
                    descriptor = self._describe_source(source)
                except Exception as exc:
                    errors[idx] = self._normalize_exception(exc)
                    if isinstance(source, (Path, DocumentStream)):
                        name = source.name
                    else:
                        name = str(source)
                    descriptors[idx] = _SourceDescriptor(
                        source_name=name,
                        input_format=self._guess_input_format(name),
                        file_size=None,
                    )
                    continue
                descriptors[idx] = descriptor
                preflight = self._preflight_limits(
                    descriptor=descriptor, limits=resolved.limits
                )
                if preflight is not None:
                    results[idx] = preflight
                    continue
                yield ConversionItem(
                    source=source,
                    options=submit_options,
                    headers=headers,
                    metadata=_ConvertAllItemMetadata(
                        source_index=idx,
                        descriptor=descriptor,
                    ),
                )

        next_output_index = 0
        async for item, outcome in self._submit_and_retrieve_many_async(
            item_list=make_items(),
            max_in_flight=in_flight,
            ordered=True,
            target=InBodyTarget(),
        ):
            metadata = item.metadata
            if not isinstance(metadata, _ConvertAllItemMetadata):
                raise TypeError(
                    "ConversionItem metadata must be _ConvertAllItemMetadata."
                )
            while next_output_index < metadata.source_index:
                yield result_for_index(next_output_index)
                next_output_index += 1

            if isinstance(outcome, BaseException):
                errors[metadata.source_index] = self._normalize_exception(outcome)
                yield result_for_index(metadata.source_index)
            else:
                result = self._build_conversion_result(
                    payload=outcome,
                    descriptor=metadata.descriptor,
                    limits=resolved.limits,
                )
                results[metadata.source_index] = result
                yield result
            next_output_index += 1

        for idx in range(next_output_index, total):
            yield result_for_index(idx)

    async def _submit_and_retrieve_many_async(
        self,
        item_list: Iterable[ConversionItem],
        max_in_flight: int,
        ordered: bool,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> AsyncGenerator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ],
        None,
    ]:
        async with self._build_async_http_client() as async_client:

            async def process_one(
                _idx: int,
                item: ConversionItem,
                async_client: httpx.AsyncClient,
            ) -> ConvertDocumentResponse | PresignedUrlConvertResponse:
                resolved = self._resolve_options(
                    options=item.options,
                    max_num_pages=None,
                    max_file_size=None,
                    page_range=None,
                )
                effective_target = PresignedUrlTarget() if target is None else target
                submit_options = self._options_for_output_formats(
                    resolved.options,
                    output_formats=None,
                    target=effective_target,
                )
                try:
                    initial_status = await self._submit_convert_task_async(
                        source=item.source,
                        options=submit_options,
                        target=effective_target,
                        async_client=async_client,
                        request_headers=item.headers,
                    )
                except ServiceError as exc:
                    if (
                        target is not None
                        or not self._should_fallback_from_presigned_target(exc)
                    ):
                        raise
                    effective_target = InBodyTarget()
                    submit_options = self._options_for_output_formats(
                        resolved.options,
                        output_formats=None,
                        target=effective_target,
                    )
                    initial_status = await self._submit_convert_task_async(
                        source=item.source,
                        options=submit_options,
                        target=effective_target,
                        async_client=async_client,
                        request_headers=item.headers,
                    )
                terminal_status = await self._wait_for_terminal_status_for_submit_and_retrieve_many_async(
                    task_id=initial_status.task_id,
                    timeout=self._job_timeout,
                    async_client=async_client,
                    max_in_flight=max_in_flight,
                )
                if isinstance(effective_target, PresignedUrlTarget):
                    return await self._fetch_presigned_result_async(
                        task_id=initial_status.task_id,
                        last_status=terminal_status,
                        async_client=async_client,
                    )
                return await self._fetch_convert_result_payload_async(
                    task_id=initial_status.task_id,
                    last_status=terminal_status,
                    async_client=async_client,
                )

            buffered_results: dict[
                int,
                tuple[
                    ConversionItem,
                    ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
                ],
            ] = {}
            next_ordered_index = 0
            async for idx, item, outcome in _run_bounded(
                items=item_list,
                process_one=process_one,
                async_client=async_client,
                max_in_flight=max_in_flight,
            ):
                normalized: (
                    ConvertDocumentResponse | PresignedUrlConvertResponse | Exception
                )
                if isinstance(outcome, BaseException):
                    normalized = self._normalize_exception(outcome)
                else:
                    normalized = outcome

                if ordered:
                    buffered_results[idx] = (item, normalized)
                    while next_ordered_index in buffered_results:
                        yield buffered_results.pop(next_ordered_index)
                        next_ordered_index += 1
                    continue

                yield item, normalized

    def _build_async_http_client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=self._http_connect_timeout,
            read=self._http_read_timeout,
            write=self._http_read_timeout,
            pool=self._http_read_timeout,
        )
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return httpx.AsyncClient(timeout=timeout, headers=headers)

    async def _submit_convert_task_async(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions,
        target: SubmitTarget,
        async_client: httpx.AsyncClient,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        if isinstance(source, (str, HttpSourceRequest)):
            request_source = self._normalize_http_source(source)
            request = ConvertDocumentsRequest(
                options=options,
                sources=[request_source],
                target=target,
            )
            response = await self._request_with_retry_async(
                async_client=async_client,
                method="POST",
                path="/v1/convert/source/async",
                json=request.model_dump(mode="json"),
                headers=request_headers,
            )
        else:
            files = await self._source_to_upload_files_async(source)
            data = options.model_dump(mode="json", exclude_none=True)
            data["target_type"] = target.kind
            response = await self._request_with_retry_async(
                async_client=async_client,
                method="POST",
                path="/v1/convert/file/async",
                data=data,
                files=files,
                headers=request_headers,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Task submission failed.")
        status = TaskStatusResponse.model_validate_json(response.text)
        logger.info(
            "Submitted convert task for source=%s task_id=%s status=%s position=%s",
            source_name,
            status.task_id,
            status.task_status,
            status.task_position,
        )
        return status

    async def _poll_task_status_async(
        self,
        task_id: str,
        wait: float,
        async_client: httpx.AsyncClient,
    ) -> TaskStatusResponse:
        response = await self._request_with_retry_async(
            async_client=async_client,
            method="GET",
            path=f"/v1/status/poll/{task_id}",
            params={"wait": wait},
        )
        if response.status_code == 404:
            raise TaskNotFoundError(f"Task {task_id} was not found.")
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, f"Polling task {task_id} failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _wait_for_terminal_status_async(
        self,
        task_id: str,
        timeout: float,
        async_client: httpx.AsyncClient,
    ) -> TaskStatusResponse:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(
                    f"Timed out waiting for task {task_id} after {timeout:.2f}s."
                )
            wait = min(self._poll_server_wait, remaining)
            logger.info("Polling status for task_id=%s wait=%.2fs", task_id, wait)
            poll_started = time.monotonic()
            update = await self._poll_task_status_async(
                task_id=task_id,
                wait=wait,
                async_client=async_client,
            )
            logger.info(
                "Received status for task_id=%s status=%s position=%s",
                task_id,
                update.task_status,
                update.task_position,
            )
            if is_terminal_task_status(update):
                return update

            # Keep a minimum client-side poll cadence when server-side wait is ignored.
            sleep_for = _poll_sleep_duration(
                poll_started=poll_started,
                poll_interval=self._poll_client_interval,
                deadline=deadline,
            )
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    def _submit_and_retrieve_many_uses_websocket_wait(
        self,
        max_in_flight: int,
    ) -> bool:
        return (
            self._status_watcher_kind == StatusWatcherKind.WEBSOCKET
            and max_in_flight <= SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS
        )

    async def _wait_for_terminal_status_for_submit_and_retrieve_many_async(
        self,
        task_id: str,
        timeout: float,
        async_client: httpx.AsyncClient,
        max_in_flight: int,
    ) -> TaskStatusResponse:
        if self._submit_and_retrieve_many_uses_websocket_wait(
            max_in_flight=max_in_flight
        ):
            return await asyncio.to_thread(
                self._ws_watcher.wait_for_terminal,
                task_id,
                timeout,
            )
        return await self._wait_for_terminal_status_async(
            task_id=task_id,
            timeout=timeout,
            async_client=async_client,
        )

    async def _fetch_convert_result_async(
        self,
        task_id: str,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConversionResult:
        payload = await self._fetch_convert_result_payload_async(
            task_id=task_id,
            last_status=last_status,
            async_client=async_client,
        )
        return self._build_conversion_result(
            payload=payload,
            descriptor=descriptor,
            limits=limits,
        )

    async def _fetch_convert_result_payload_async(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConvertDocumentResponse:
        response = await self._fetch_result_response_async(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ConvertDocumentResponse)

    async def _fetch_presigned_result_async(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> PresignedUrlConvertResponse:
        response = await self._fetch_result_response_async(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, PresignedUrlConvertResponse)

    async def _fetch_result_response_async(
        self,
        async_client: httpx.AsyncClient,
        task_id: str,
        last_status: TaskStatusResponse | None,
        *,
        error_message: str,
    ) -> httpx.Response:
        response = await self._request_with_retry_async(
            async_client=async_client,
            method="GET",
            path=f"/v1/result/{task_id}",
        )
        if response.status_code == 404:
            self._raise_for_result_404(
                task_id=task_id,
                response=response,
                last_status=last_status,
            )
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, error_message)
        return response

    def _check_retry(
        self,
        response: httpx.Response,
        attempt: int,
        max_retries: int,
    ) -> tuple[httpx.Response | None, float]:
        """Return (response, 0.0) to yield, (None, delay) to retry after delay, or raise."""
        if response.status_code in {500, 502}:
            return self._retry_with_exponential_backoff(
                response=response,
                attempt=attempt,
                max_retries=max_retries,
                error_message=(
                    f"Service returned HTTP {response.status_code} after retries."
                ),
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

    def _transport_retry_delay(
        self,
        *,
        method: str,
        exc: httpx.HTTPError,
        attempt: int,
        max_retries: int,
    ) -> float | None:
        method_name = method.upper()
        if (
            not isinstance(exc, httpx.TransportError)
            or method_name not in TRANSPORT_RETRYABLE_HTTP_METHODS
        ):
            return None
        if attempt < max_retries:
            return self._exponential_backoff_delay(attempt)
        raise ServiceUnavailableError(
            "Service transport request failed after retries.",
            detail=str(exc),
        ) from exc

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

    def _request_with_retry(
        self,
        method: str,
        path: str,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        url = self._url(path)
        method_name = method.upper()
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = self._http_client.request(
                    method=method_name,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                delay = self._transport_retry_delay(
                    method=method_name,
                    exc=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                )
                if delay is not None:
                    time.sleep(delay)
                    continue
                raise ServiceUnavailableError(
                    "Service transport request failed.",
                    detail=str(exc),
                ) from exc
            result, delay = self._check_retry(response, attempt, max_retries)
            if result is not None:
                return result
            if delay > 0:
                time.sleep(delay)

        raise ServiceUnavailableError("Service request failed after retry loop.")

    async def _request_with_retry_async(
        self,
        async_client: httpx.AsyncClient,
        method: str,
        path: str,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        url = self._url(path)
        method_name = method.upper()
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = await async_client.request(
                    method=method_name,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                delay = self._transport_retry_delay(
                    method=method_name,
                    exc=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                )
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
                raise ServiceUnavailableError(
                    "Service transport request failed.",
                    detail=str(exc),
                ) from exc
            result, delay = self._check_retry(response, attempt, max_retries)
            if result is not None:
                return result
            if delay > 0:
                await asyncio.sleep(delay)

        raise ServiceUnavailableError("Service request failed after retry loop.")

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

    def _should_fallback_from_presigned_target(self, exc: ServiceError) -> bool:
        if exc.status_code not in {400, 422} or exc.detail is None:
            return False

        detail = exc.detail.lower()
        if "artifact storage to be configured" in detail:
            return True
        if "presigned_url" not in detail and "presigned url" not in detail:
            return False
        return any(
            phrase in detail
            for phrase in (
                "input should be",
                "unexpected value",
                "validation error",
                "literal_error",
                "enum",
            )
        )

    def _failure_message(self, result: ConversionResult) -> str:
        if result.errors:
            messages = "; ".join(item.error_message for item in result.errors)
            return (
                f"Conversion failed for {result.input.file} with status "
                f"{result.status.value}. Errors: {messages}"
            )
        return f"Conversion failed for {result.input.file} with status {result.status.value}."

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

    def _normalize_base_url(self, url: str) -> str:
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

    def _build_extension_to_format_map(self) -> dict[str, InputFormat]:
        extension_to_format: dict[str, InputFormat] = {}
        for input_format, extensions in FormatToExtensions.items():
            for extension in extensions:
                extension_to_format.setdefault(extension.lower(), input_format)
        return extension_to_format
