"""Async client SDK for docling-serve."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Literal, cast, overload

import httpx
from docling_core.types.io import DocumentStream

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.service.chunking import (
    HierarchicalChunkerOptions,
    HybridChunkerOptions,
)
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import (
    ConvertDocumentsRequest,
    HttpSourceRequest,
)
from docling.datamodel.service.responses import (
    ChunkDocumentResponse,
    ConvertDocumentResponse,
    HealthCheckResponse,
    TaskStatusResponse,
)
from docling.datamodel.service.targets import InBodyTarget, ZipTarget
from docling.datamodel.settings import DocumentLimits
from docling.service_client._scheduler import _run_bounded
from docling.service_client.client import (
    DEFAULT_MAX_CONCURRENCY,
    ChunkerKind,
    ConversionItem,
    RawServiceResult,
    SourceType,
    StatusWatcherKind,
    _build_conversion_result,
    _build_ws_status_url,
    _check_retry,
    _decode_raw_result,
    _describe_source,
    _normalize_base_url,
    _options_for_target_format,
    _raise_for_generic_http_error,
    _raise_for_result_404,
    _resolve_options,
    _source_name,
    _SourceDescriptor,
    _transport_retry_delay,
    _validate_concurrency,
    _validate_http_source,
)
from docling.service_client.exceptions import ServiceUnavailableError, TaskNotFoundError
from docling.service_client.job import AsyncConversionJob, _AsyncJobHandlers
from docling.service_client.watchers import (
    AsyncPollingWatcher,
    AsyncWebSocketWatcher,
)

logger = logging.getLogger(__name__)


class AsyncDoclingServiceClient:
    """Native async client for docling-serve.

    Must be used as an async context manager or closed explicitly via ``aclose()``:

        async with AsyncDoclingServiceClient(url="...") as client:
            job = await client.submit(source)
            result = await job.result()
    """

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
        self._base_url = _normalize_base_url(url)
        self._api_key = api_key
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
        self._max_concurrency = _validate_concurrency(
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
            ws_url_for_task=lambda task_id: _build_ws_status_url(
                self._base_url, self._api_key, task_id
            ),
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
        """Release HTTP resources owned by this client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # ------------------------------------------------------------------ public

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: None | Literal["json"] = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncConversionJob[ConversionResult]: ...

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: OutputFormat = ...,
        headers: dict[str, str] | None = None,
    ) -> AsyncConversionJob[RawServiceResult]: ...

    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: OutputFormat | Literal["json"] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncConversionJob[ConversionResult] | AsyncConversionJob[RawServiceResult]:
        """Submit a single conversion and return an async job handle."""
        normalized_target_format: OutputFormat | None = (
            OutputFormat.JSON
            if target_format == "json"
            else cast(OutputFormat | None, target_format)
        )
        resolved = _resolve_options(
            self._default_options,
            options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        submit_options, raw_result = _options_for_target_format(
            resolved.options, normalized_target_format
        )
        descriptor = _describe_source(source)
        initial_status = await self._submit_convert_task(
            source=source,
            source_headers=None,
            options=submit_options,
            raw_result=raw_result,
            request_headers=headers,
        )
        handlers: _AsyncJobHandlers[Any] = _AsyncJobHandlers(
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=lambda tid, last: self._fetch_convert_result(
                task_id=tid,
                descriptor=descriptor,
                limits=resolved.limits,
                raw_result=raw_result,
                last_status=last,
            ),
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
        """Submit a chunking task and return an async job handle."""
        resolved = _resolve_options(
            self._default_options,
            options,
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
                task_id=tid, last_status=last
            ),
        )
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_and_retrieve_many(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
    ) -> AsyncGenerator[
        tuple[ConversionItem, ConvertDocumentResponse | Exception], None
    ]:
        """Submit many items and yield (item, outcome) pairs as they complete."""
        assert self._async_client is not None, "client not open — use async with"
        max_in_flight = _validate_concurrency(max_in_flight, name="max_in_flight")

        async def process_one(
            _idx: int,
            item: ConversionItem,
            _client: httpx.AsyncClient,  # ignored — self._async_client is used
        ) -> ConvertDocumentResponse:
            resolved = _resolve_options(
                self._default_options,
                item.options,
                max_num_pages=None,
                max_file_size=None,
                page_range=None,
            )
            submit_options, _ = _options_for_target_format(
                resolved.options, OutputFormat.JSON
            )
            initial_status = await self._submit_convert_task(
                source=item.source,
                source_headers=item.source_headers,
                options=submit_options,
                raw_result=False,
                request_headers=item.headers,
            )
            assert self._polling_watcher is not None
            terminal_status = await self._polling_watcher.wait_for_terminal(
                task_id=initial_status.task_id,
                timeout=self._job_timeout,
            )
            return await self._fetch_convert_result_payload(
                task_id=initial_status.task_id,
                last_status=terminal_status,
            )

        buffered: dict[
            int, tuple[ConversionItem, ConvertDocumentResponse | Exception]
        ] = {}
        next_ordered_index = 0

        async for idx, item, outcome in _run_bounded(
            items=items,
            process_one=process_one,
            async_client=self._async_client,
            max_in_flight=max_in_flight,
        ):
            normalized: ConvertDocumentResponse | Exception
            if isinstance(outcome, BaseException):
                normalized = (
                    outcome
                    if isinstance(outcome, Exception)
                    else RuntimeError(str(outcome))
                )
            else:
                normalized = outcome

            if ordered:
                buffered[idx] = (item, normalized)
                while next_ordered_index in buffered:
                    yield buffered.pop(next_ordered_index)
                    next_ordered_index += 1
                continue

            yield item, normalized

    async def health(self) -> HealthCheckResponse:
        response = await self._request_with_retry("GET", "/health", retries=0)
        if response.status_code != 200:
            _raise_for_generic_http_error(response, "Health check request failed.")
        return HealthCheckResponse.model_validate_json(response.text)

    async def version(self) -> dict[str, Any]:
        response = await self._request_with_retry("GET", "/version", retries=0)
        if response.status_code != 200:
            _raise_for_generic_http_error(response, "Version request failed.")
        return response.json()

    # ----------------------------------------------------------------- private

    def _status_watcher(self) -> AsyncPollingWatcher | AsyncWebSocketWatcher:
        assert self._polling_watcher is not None and self._ws_watcher is not None
        if self._status_watcher_kind == StatusWatcherKind.POLLING:
            return self._polling_watcher
        return self._ws_watcher

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
        url = self._url(path)
        method_name = method.upper()
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = await self._async_client.request(
                    method=method_name,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                delay = _transport_retry_delay(
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
            result, delay = _check_retry(response, attempt, max_retries)
            if result is not None:
                return result
            if delay > 0:
                await asyncio.sleep(delay)

        raise ServiceUnavailableError("Service request failed after retry loop.")

    async def _submit_convert_task(
        self,
        source: SourceType,
        source_headers: dict[str, str] | None,
        options: ConvertDocumentsRequestOptions,
        raw_result: bool,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source_name = _source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        target = ZipTarget() if raw_result else InBodyTarget()
        if isinstance(source, str):
            _validate_http_source(source)
            request = ConvertDocumentsRequest(
                options=options,
                sources=[
                    HttpSourceRequest(
                        url=source,
                        headers={} if source_headers is None else source_headers,
                    )
                ],
                target=target,
            )
            response = await self._request_with_retry(
                method="POST",
                path="/v1/convert/source/async",
                json=request.model_dump(mode="json"),
                headers=request_headers,
            )
        else:
            files = await self._source_to_upload_files(source)
            data = options.model_dump(mode="json", exclude_none=True)
            data["target_type"] = "zip" if raw_result else "inbody"
            response = await self._request_with_retry(
                method="POST",
                path="/v1/convert/file/async",
                data=data,
                files=files,
                headers=request_headers,
            )

        if response.status_code != 200:
            _raise_for_generic_http_error(response, "Task submission failed.")
        status = TaskStatusResponse.model_validate_json(response.text)
        logger.info(
            "Submitted convert task for source=%s task_id=%s status=%s position=%s",
            source_name,
            status.task_id,
            status.task_status,
            status.task_position,
        )
        return status

    async def _submit_chunk_task(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions,
    ) -> TaskStatusResponse:
        if isinstance(source, str):
            _validate_http_source(source)
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
                "sources": [
                    HttpSourceRequest(url=source, headers={}).model_dump(mode="json")
                ],
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
            response = await self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/file/async",
                data=data,
                files=files,
            )

        if response.status_code != 200:
            _raise_for_generic_http_error(response, "Chunk task submission failed.")
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
            _raise_for_generic_http_error(response, f"Polling task {task_id} failed.")
        return TaskStatusResponse.model_validate_json(response.text)

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

    async def _fetch_convert_result(
        self,
        task_id: str,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        raw_result: bool,
        last_status: TaskStatusResponse | None,
    ) -> ConversionResult | RawServiceResult:
        if raw_result:
            response = await self._request_with_retry(
                method="GET",
                path=f"/v1/result/{task_id}",
            )
            if response.status_code == 404:
                _raise_for_result_404(task_id, response, last_status)
            if response.status_code != 200:
                _raise_for_generic_http_error(
                    response, f"Fetching result for task {task_id} failed."
                )
            return _decode_raw_result(response)

        payload = await self._fetch_convert_result_payload(
            task_id=task_id, last_status=last_status
        )
        return _build_conversion_result(
            payload=payload, descriptor=descriptor, limits=limits
        )

    async def _fetch_result_response(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        *,
        error_message: str,
    ) -> httpx.Response:
        response = await self._request_with_retry(
            method="GET",
            path=f"/v1/result/{task_id}",
        )
        if response.status_code == 404:
            _raise_for_result_404(task_id, response, last_status)
        if response.status_code != 200:
            _raise_for_generic_http_error(response, error_message)
        return response

    async def _fetch_convert_result_payload(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ConvertDocumentResponse:
        response = await self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return ConvertDocumentResponse.model_validate_json(response.text)

    async def _fetch_chunk_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ChunkDocumentResponse:
        response = await self._fetch_result_response(
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching chunk result for task {task_id} failed.",
        )
        return ChunkDocumentResponse.model_validate_json(response.text)

    def _url(self, path: str) -> str:
        if path.startswith("/"):
            return f"{self._base_url}{path}"
        return f"{self._base_url}/{path}"
