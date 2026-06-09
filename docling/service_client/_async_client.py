"""Async client SDK for docling-serve."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import sys
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path, PurePath
from typing import IO, Any, Literal, TypeVar, cast, overload
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
    TaskFailureResult,
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
from docling.service_client.client import (
    DEFAULT_MAX_CONCURRENCY,
    HTTP_RETRY_BACKOFF_BASE_SECONDS,
    MAX_CONCURRENCY_LIMIT,
    SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS,
    TRANSPORT_RETRYABLE_HTTP_METHODS,
    BatchSubmitTarget,
    ChunkerKind,
    ConversionItem,
    RawServiceResult,
    SourceType,
    StatusWatcherKind,
    SubmitTarget,
)
from docling.service_client.exceptions import (
    ConversionError,
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
from docling.service_client.job import AsyncConversionJob, _AsyncJobHandlers
from docling.service_client.watchers import (
    AsyncPollingWatcher,
    AsyncWebSocketWatcher,
    _poll_sleep_duration,
    is_terminal_task_status,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class AsyncDoclingServiceClient:
    """Native async client for docling-serve."""

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

        self._async_client: httpx.AsyncClient | None = None
        self._polling_watcher: AsyncPollingWatcher | None = None
        self._ws_watcher: AsyncWebSocketWatcher | None = None

    async def __aenter__(self) -> AsyncDoclingServiceClient:
        timeout = httpx.Timeout(
            connect=self._http_connect_timeout,
            read=self._http_read_timeout,
            write=self._http_read_timeout,
            pool=self._http_read_timeout,
        )
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        self._async_client = httpx.AsyncClient(timeout=timeout, headers=headers)

        self._polling_watcher = AsyncPollingWatcher(
            poll_status=self._poll_task_status,
            poll_server_wait=self._poll_server_wait,
            poll_client_interval=self._poll_client_interval,
            default_timeout=self._job_timeout,
        )

        ws_headers = {"X-Api-Key": self._api_key} if self._api_key else {}
        self._ws_watcher = AsyncWebSocketWatcher(
            ws_url_for_task=self._build_ws_status_url,
            poll_fallback=self._polling_watcher,
            fallback_to_poll=self._ws_fallback_to_poll,
            connect_timeout=self._http_connect_timeout,
            default_timeout=self._job_timeout,
            additional_headers=ws_headers,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: InBodyTarget = ...,
    ) -> AsyncConversionJob[ConversionResult]: ...

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: ZipTarget,
    ) -> AsyncConversionJob[RawServiceResult]: ...

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: PresignedUrlTarget | None = None,
    ) -> AsyncConversionJob[PresignedUrlConvertResponse | ConversionResult]: ...

    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: SubmitTarget | None = None,
    ) -> (
        AsyncConversionJob[ConversionResult]
        | AsyncConversionJob[RawServiceResult]
        | AsyncConversionJob[PresignedUrlConvertResponse]
    ):
        assert self._async_client is not None, "client not open — use async with"
        descriptor = self._describe_source(source)
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )

        effective_target: SubmitTarget
        if target is None:
            effective_target = PresignedUrlTarget()
            submit_options = self._options_for_output_formats(
                resolved.options,
                output_formats=output_formats,
                target=effective_target,
            )
            try:
                initial_status = await self._submit_convert_task(
                    source=source,
                    options=submit_options,
                    target=effective_target,
                    async_client=self._async_client,
                    request_headers=headers,
                )
            except ServiceError as exc:
                if not self._should_fallback_from_presigned_target(exc):
                    raise
                effective_target = InBodyTarget()
                submit_options = self._options_for_output_formats(
                    resolved.options,
                    output_formats=output_formats,
                    target=effective_target,
                )
                initial_status = await self._submit_convert_task(
                    source=source,
                    options=submit_options,
                    target=effective_target,
                    async_client=self._async_client,
                    request_headers=headers,
                )
        else:
            effective_target = target
            submit_options = self._options_for_output_formats(
                resolved.options,
                output_formats=output_formats,
                target=effective_target,
            )
            initial_status = await self._submit_convert_task(
                source=source,
                options=submit_options,
                target=effective_target,
                async_client=self._async_client,
                request_headers=headers,
            )

        handlers: _AsyncJobHandlers[Any] = _AsyncJobHandlers(
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=self._make_convert_fetch_result_handler(
                descriptor=descriptor,
                limits=resolved.limits,
                target=effective_target,
                async_client=self._async_client,
            ),
        )
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_batch(
        self,
        sources: list[BatchSourceRequestItem],
        target: BatchSubmitTarget,
        output_formats: list[OutputFormat] | None = None,
        options: ConvertDocumentsRequestOptions | None = None,
        headers: dict[str, str] | None = None,
    ) -> (
        AsyncConversionJob[PresignedUrlConvertDocumentResponse]
        | AsyncConversionJob[PresignedUrlConvertResponse]
    ):
        assert self._async_client is not None, "client not open — use async with"
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
        initial_status = await self._submit_batch_task(
            sources=sources,
            options=submit_options,
            target=target,
            async_client=self._async_client,
            request_headers=headers,
        )

        if isinstance(target, S3Target):

            async def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertDocumentResponse:
                return await self._fetch_presigned_document_result(
                    task_id=task_id,
                    last_status=last_status,
                    async_client=self._async_client,
                )

        else:

            async def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertResponse:
                return await self._fetch_presigned_result(
                    task_id=task_id,
                    last_status=last_status,
                    async_client=self._async_client,
                )

        handlers = _AsyncJobHandlers[Any](
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=fetch_result,
        )
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions | None = None,
    ) -> AsyncConversionJob[ChunkDocumentResponse]:
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        initial_status = await self._submit_chunk_task(
            source=source,
            chunker=chunker,
            options=resolved.options,
        )
        handlers: _AsyncJobHandlers[ChunkDocumentResponse] = _AsyncJobHandlers(
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=lambda tid, last: self._fetch_chunk_result(
                task_id=tid,
                last_status=last,
            ),
        )
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_and_retrieve_each(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> AsyncGenerator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ],
        None,
    ]:
        assert self._async_client is not None, "client not open — use async with"
        max_in_flight = self._validate_concurrency(
            max_in_flight,
            name="max_in_flight",
        )

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
                initial_status = await self._submit_convert_task(
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
                initial_status = await self._submit_convert_task(
                    source=item.source,
                    options=submit_options,
                    target=effective_target,
                    async_client=async_client,
                    request_headers=item.headers,
                )

            terminal_status = (
                await self._wait_for_terminal_status_for_submit_and_retrieve_many(
                    task_id=initial_status.task_id,
                    timeout=self._job_timeout,
                    async_client=async_client,
                    max_in_flight=max_in_flight,
                )
            )
            if isinstance(effective_target, PresignedUrlTarget):
                return await self._fetch_presigned_result(
                    task_id=initial_status.task_id,
                    last_status=terminal_status,
                    async_client=async_client,
                )
            return await self._fetch_convert_result_payload(
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
            items=items,
            process_one=process_one,
            async_client=self._async_client,
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

    async def submit_and_retrieve_many(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: InBodyTarget | PresignedUrlTarget | None = None,
    ) -> AsyncGenerator[
        tuple[
            ConversionItem,
            ConvertDocumentResponse | PresignedUrlConvertResponse | Exception,
        ],
        None,
    ]:
        warnings.warn(
            "submit_and_retrieve_many() is deprecated; use submit_and_retrieve_each().",
            DeprecationWarning,
            stacklevel=2,
        )
        async for item, outcome in self.submit_and_retrieve_each(
            items=items,
            max_in_flight=max_in_flight,
            ordered=ordered,
            target=target,
        ):
            yield item, outcome

    async def health(self) -> HealthCheckResponse:
        response = await self._request_with_retry("GET", "/health", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Health check request failed.")
        return HealthCheckResponse.model_validate_json(response.text)

    async def version(self) -> dict[str, Any]:
        response = await self._request_with_retry("GET", "/version", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Version request failed.")
        return response.json()

    def _status_watcher(self) -> AsyncPollingWatcher | AsyncWebSocketWatcher:
        assert self._polling_watcher is not None and self._ws_watcher is not None
        if self._status_watcher_kind == StatusWatcherKind.POLLING:
            return self._polling_watcher
        return self._ws_watcher

    def _make_convert_fetch_result_handler(
        self,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        target: SubmitTarget,
        async_client: httpx.AsyncClient,
    ) -> Any:
        if isinstance(target, ZipTarget):
            return lambda task_id, last_status: self._fetch_raw_result(
                task_id=task_id,
                last_status=last_status,
                async_client=async_client,
            )
        if isinstance(target, PresignedUrlTarget):
            return lambda task_id, last_status: self._fetch_presigned_result(
                task_id=task_id,
                last_status=last_status,
                async_client=async_client,
            )
        return lambda task_id, last_status: self._fetch_convert_result(
            task_id=task_id,
            descriptor=descriptor,
            limits=limits,
            last_status=last_status,
            async_client=async_client,
        )

    async def _request_with_retry(
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
        assert self._async_client is not None, "client not open — use async with"
        return await self._request_with_retry_using_client(
            async_client=self._async_client,
            method=method,
            path=path,
            json=json,
            data=data,
            files=files,
            params=params,
            headers=headers,
            retries=retries,
        )

    async def _request_with_retry_using_client(
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

    async def _submit_convert_task(
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
            response = await self._request_with_retry_using_client(
                async_client=async_client,
                method="POST",
                path="/v1/convert/source/async",
                json=self._serialize_convert_request(request),
                headers=request_headers,
            )
        else:
            files = await self._source_to_upload_files(source)
            data = self._serialize_convert_options(options)
            data["target_type"] = target.kind
            response = await self._request_with_retry_using_client(
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

    async def _submit_batch_task(
        self,
        sources: list[BatchSourceRequestItem],
        options: ConvertDocumentsRequestOptions,
        target: BatchSubmitTarget,
        async_client: httpx.AsyncClient,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        request = BatchConvertSourcesRequest(
            options=options,
            sources=sources,
            target=target,
        )
        response = await self._request_with_retry_using_client(
            async_client=async_client,
            method="POST",
            path="/v1/convert/source/batch",
            json=self._serialize_convert_request(request),
            headers=request_headers,
        )
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response,
                "Batch task submission failed.",
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _submit_chunk_task(
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
                "convert_options": self._serialize_convert_options(options),
                "chunking_options": chunking_options.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "sources": [request_source.model_dump(mode="json", exclude_none=True)],
                "include_converted_doc": False,
                "target": InBodyTarget().model_dump(mode="json"),
                "callbacks": [],
            }
            response = await self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/source/async",
                json=payload,
            )
        else:
            files = await self._source_to_upload_files(source)
            data: dict[str, Any] = {
                f"convert_{key}": value
                for key, value in self._serialize_convert_options(options).items()
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
            data["target_type"] = InBodyTarget().kind
            response = await self._request_with_retry(
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

    async def _poll_task_status(self, task_id: str, wait: float) -> TaskStatusResponse:
        response = await self._request_with_retry(
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

    async def _poll_task_status_using_client(
        self,
        task_id: str,
        wait: float,
        async_client: httpx.AsyncClient,
    ) -> TaskStatusResponse:
        response = await self._request_with_retry_using_client(
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

    async def _wait_for_terminal_status(
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
            update = await self._poll_task_status_using_client(
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

    async def _wait_for_terminal_status_for_submit_and_retrieve_many(
        self,
        task_id: str,
        timeout: float,
        async_client: httpx.AsyncClient,
        max_in_flight: int,
    ) -> TaskStatusResponse:
        if self._submit_and_retrieve_many_uses_websocket_wait(
            max_in_flight=max_in_flight
        ):
            assert self._ws_watcher is not None
            return await self._ws_watcher.wait_for_terminal(task_id, timeout)
        return await self._wait_for_terminal_status(
            task_id=task_id,
            timeout=timeout,
            async_client=async_client,
        )

    async def _fetch_convert_result(
        self,
        task_id: str,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConversionResult:
        payload = await self._fetch_convert_result_payload(
            task_id=task_id,
            last_status=last_status,
            async_client=async_client,
        )
        return self._build_conversion_result(
            payload=payload,
            descriptor=descriptor,
            limits=limits,
        )

    async def _fetch_convert_result_payload(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConvertDocumentResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ConvertDocumentResponse)

    async def _fetch_raw_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> RawServiceResult:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._decode_raw_result(response)

    async def _fetch_presigned_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> PresignedUrlConvertResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, PresignedUrlConvertResponse)

    async def _fetch_presigned_document_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> PresignedUrlConvertDocumentResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(
            response,
            PresignedUrlConvertDocumentResponse,
        )

    async def _fetch_chunk_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ChunkDocumentResponse:
        response = await self._fetch_result_response(
            async_client=self._async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching chunk result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ChunkDocumentResponse)

    async def _fetch_result_response(
        self,
        async_client: httpx.AsyncClient | None,
        task_id: str,
        last_status: TaskStatusResponse | None,
        *,
        error_message: str,
    ) -> httpx.Response:
        assert async_client is not None, "client not open — use async with"
        response = await self._request_with_retry_using_client(
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
        self._raise_if_task_failure_result(response)
        return response

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

    def _serialize_convert_options(
        self,
        options: ConvertDocumentsRequestOptions,
    ) -> dict[str, Any]:
        return options.model_dump(
            mode="json",
            exclude_defaults=True,
            exclude_none=True,
        )

    def _serialize_convert_request(
        self,
        request: ConvertDocumentsRequest | BatchConvertSourcesRequest,
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["options"] = self._serialize_convert_options(request.options)
        return payload

    def _resolve_options(
        self,
        options: ConvertDocumentsRequestOptions | None,
        max_num_pages: int | None,
        max_file_size: int | None,
        page_range: PageRange | None,
    ) -> _ResolvedOptions:
        merged = self._default_options.model_copy(deep=True)
        if options is not None and options.model_fields_set:
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

    def _build_input_document(
        self,
        source_name: str,
        input_format: InputFormat,
        file_size: int | None,
        limits: DocumentLimits,
    ) -> InputDocument:
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
        self,
        source: str | HttpSourceRequest,
    ) -> HttpSourceRequest:
        if isinstance(source, HttpSourceRequest):
            return source
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("String sources must be HTTP or HTTPS URLs.")
        return HttpSourceRequest(url=source, headers={})

    async def _source_to_upload_files(
        self,
        source: Path | DocumentStream,
    ) -> dict[str, tuple[str, IO[bytes] | bytes, str]]:
        if isinstance(source, Path):
            filename = source.name
            content: IO[bytes] | bytes = await asyncio.to_thread(source.read_bytes)
        else:
            filename = source.name
            source.stream.seek(0)
            content = source.stream
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {"files": (filename, content, mime)}

    @staticmethod
    def _validate_concurrency(value: int, *, name: str) -> int:
        if value < 1 or value > MAX_CONCURRENCY_LIMIT:
            raise ValueError(
                f"{name} must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}."
            )
        return value

    @staticmethod
    def _normalize_exception(exc: BaseException) -> Exception:
        if isinstance(exc, Exception):
            return exc
        return RuntimeError(str(exc))

    def _check_retry(
        self,
        response: httpx.Response,
        attempt: int,
        max_retries: int,
    ) -> tuple[httpx.Response | None, float]:
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
            return None, HTTP_RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
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
            return HTTP_RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
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
                    raise TaskExecutionError(message, failure=last_status.failure)
                raise ResultExpiredError(f"Result for task {task_id} has expired.")
            raise ResultNotReadyError(f"Result for task {task_id} is not ready.")
        raise ServiceError(
            "Unexpected result lookup error.",
            status_code=response.status_code,
            detail=detail,
        )

    def _raise_if_task_failure_result(self, response: httpx.Response) -> None:
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type.lower():
            return
        try:
            payload = response.json()
        except ValueError:
            return
        if not isinstance(payload, dict) or payload.get("kind") != "TaskFailureResult":
            return
        task_failure = TaskFailureResult.model_validate(payload)
        raise TaskExecutionError(
            task_failure.failure.message,
            failure=task_failure.failure,
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


class _ResolvedOptions:
    def __init__(
        self,
        options: ConvertDocumentsRequestOptions,
        limits: DocumentLimits,
    ) -> None:
        self.options = options
        self.limits = limits


class _SourceDescriptor:
    def __init__(
        self,
        source_name: str,
        input_format: InputFormat,
        file_size: int | None,
    ) -> None:
        self.source_name = source_name
        self.input_format = input_format
        self.file_size = file_size
