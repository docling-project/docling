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
    BatchSubmitTarget,
    ChunkerKind,
    ConversionItem,
    RawServiceResult,
    SourceType,
    StatusWatcherKind,
    SubmitTarget,
    _BaseDoclingServiceClient,
    _ResolvedOptions,
    _SourceDescriptor,
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


class AsyncDoclingServiceClient(_BaseDoclingServiceClient):
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
        super().__init__(
            url=url,
            api_key=api_key,
            options=options,
            status_watcher=status_watcher,
            ws_fallback_to_poll=ws_fallback_to_poll,
            poll_server_wait=poll_server_wait,
            poll_client_interval=poll_client_interval,
            job_timeout=job_timeout,
            max_concurrency=max_concurrency,
            http_retries=http_retries,
            http_connect_timeout=http_connect_timeout,
            http_read_timeout=http_read_timeout,
        )
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
        source = self._normalize_source(source)
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        if isinstance(source, HttpSourceRequest):
            request = ConvertDocumentsRequest(
                options=options,
                sources=[source],
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
                data=self._form_encode_options(data),
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
        source = self._normalize_source(source)
        if isinstance(source, HttpSourceRequest):
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
                "sources": [source.model_dump(mode="json", exclude_none=True)],
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
                data=self._form_encode_options(data),
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
