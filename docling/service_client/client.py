"""Synchronous client SDK for docling-serve."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Literal, TypeVar, cast, overload

import httpx
from docling_core.types.io import DocumentStream

from docling.datamodel.base_models import (
    ConversionStatus,
    OutputFormat,
)
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
from docling.datamodel.settings import DocumentLimits, PageRange
from docling.service_client._base import (
    DEFAULT_MAX_CONCURRENCY,
    HTTP_RETRY_BACKOFF_BASE_SECONDS,
    MAX_CONCURRENCY_LIMIT,
    SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS,
    SUCCESS_CONVERSION_STATUSES,
    ChunkerKind,
    ConversionItem,
    ExperimentalWarning,
    RawServiceResult,
    SourceType,
    StatusWatcherKind,
    _BaseClient,
    _ResolvedOptions,
    _SourceDescriptor,
)
from docling.service_client._scheduler import _run_bounded
from docling.service_client.exceptions import (
    ConversionError,
    ServiceUnavailableError,
    TaskNotFoundError,
    TaskTimeoutError,
)
from docling.service_client.job import ConversionJob, _JobHandlers
from docling.service_client.watchers import (
    PollingWatcher,
    StatusWatcher,
    WebSocketWatcher,
    _poll_sleep_duration,
    is_terminal_task_status,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "HTTP_RETRY_BACKOFF_BASE_SECONDS",
    "MAX_CONCURRENCY_LIMIT",
    "SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS",
    "SUCCESS_CONVERSION_STATUSES",
    "ChunkerKind",
    "ConversionItem",
    "DoclingServiceClient",
    "ExperimentalWarning",
    "RawServiceResult",
    "SourceType",
    "StatusWatcherKind",
]


@dataclass(slots=True)
class _ConvertAllItemMetadata:
    source_index: int
    descriptor: _SourceDescriptor


class DoclingServiceClient(_BaseClient):
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
        submit_options, _ = self._options_for_target_format(
            resolved.options, OutputFormat.JSON
        )
        return self._iterate_convert_all_sync(
            sources=sources,
            headers=headers,
            resolved=resolved,
            submit_options=submit_options,
            in_flight=effective_cap,
        )

    def submit_and_retrieve_many(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
    ) -> Iterator[tuple[ConversionItem, ConvertDocumentResponse | Exception]]:
        return self._run_submit_and_retrieve_many_async(
            item_list=items,
            max_in_flight=self._validate_concurrency(
                max_in_flight, name="max_in_flight"
            ),
            ordered=ordered,
        )

    def chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions | None = None,
    ) -> ChunkDocumentResponse:
        job = self.submit_chunk(source=source, chunker=chunker, options=options)
        return job.result(timeout=self._job_timeout)

    @overload
    def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: None | Literal["json"] = None,
        headers: dict[str, str] | None = None,
    ) -> ConversionJob[ConversionResult]: ...

    @overload
    def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: OutputFormat = ...,
        headers: dict[str, str] | None = None,
    ) -> ConversionJob[RawServiceResult]: ...

    def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        target_format: OutputFormat | Literal["json"] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ConversionJob[ConversionResult] | ConversionJob[RawServiceResult]:
        normalized_target_format: OutputFormat | None = (
            OutputFormat.JSON
            if target_format == "json"
            else cast(OutputFormat | None, target_format)
        )
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        submit_options, raw_result = self._options_for_target_format(
            resolved.options, normalized_target_format
        )
        return self._submit_conversion_job(
            source=source,
            source_headers=None,
            options=submit_options,
            limits=resolved.limits,
            raw_result=raw_result,
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

        submit_options, _ = self._options_for_target_format(
            resolved.options, OutputFormat.JSON
        )
        job = cast(
            ConversionJob[ConversionResult],
            self._submit_conversion_job(
                source=source,
                source_headers=headers,
                options=submit_options,
                limits=resolved.limits,
                raw_result=False,
                descriptor=descriptor,
            ),
        )
        result = job.result(timeout=self._job_timeout)
        return result

    def _submit_conversion_job(
        self,
        source: SourceType,
        source_headers: dict[str, str] | None,
        options: ConvertDocumentsRequestOptions,
        limits: DocumentLimits,
        raw_result: bool,
        descriptor: _SourceDescriptor | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> ConversionJob[ConversionResult] | ConversionJob[RawServiceResult]:
        descriptor = descriptor or self._describe_source(source)
        initial_status = self._submit_convert_task(
            source=source,
            source_headers=source_headers,
            options=options,
            raw_result=raw_result,
            request_headers=request_headers,
        )
        handlers = _JobHandlers[Any](
            poll=self._poll_task_status,
            watch=self._watch_task_updates,
            wait=self._wait_for_terminal_status,
            fetch_result=lambda task_id, last_status: self._fetch_convert_result(
                task_id=task_id,
                descriptor=descriptor,
                limits=limits,
                raw_result=raw_result,
                last_status=last_status,
            ),
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
        source_headers: dict[str, str] | None,
        options: ConvertDocumentsRequestOptions,
        raw_result: bool,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        target = ZipTarget() if raw_result else InBodyTarget()
        if isinstance(source, str):
            self._validate_http_source(source)
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
            response = self._request_with_retry(
                method="POST",
                path="/v1/convert/source/async",
                json=request.model_dump(mode="json"),
                headers=request_headers,
            )
        else:
            files = self._source_to_upload_files(source)
            data = options.model_dump(mode="json", exclude_none=True)
            data["target_type"] = "zip" if raw_result else "inbody"
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
        if isinstance(source, str):
            self._validate_http_source(source)
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

    def _fetch_convert_result(
        self,
        task_id: str,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        raw_result: bool,
        last_status: TaskStatusResponse | None,
    ) -> ConversionResult | RawServiceResult:
        if raw_result:
            response = self._request_with_retry(
                method="GET",
                path=f"/v1/result/{task_id}",
            )
            if response.status_code == 404:
                self._raise_for_result_404(
                    task_id=task_id, response=response, last_status=last_status
                )
            if response.status_code != 200:
                self._raise_for_generic_http_error(
                    response, f"Fetching result for task {task_id} failed."
                )
            return self._decode_raw_result(response)

        payload = self._fetch_convert_result_payload(
            task_id=task_id,
            last_status=last_status,
        )
        return self._build_conversion_result(
            payload=payload, descriptor=descriptor, limits=limits
        )

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
        return ConvertDocumentResponse.model_validate_json(response.text)

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
        return ChunkDocumentResponse.model_validate_json(response.text)

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
    ) -> Iterator[tuple[ConversionItem, ConvertDocumentResponse | Exception]]:
        self._ensure_sync_bridge_allowed()
        return self._iterate_submit_and_retrieve_many_sync(
            item_list=item_list,
            max_in_flight=max_in_flight,
            ordered=ordered,
        )

    def _iterate_submit_and_retrieve_many_sync(
        self,
        item_list: Iterable[ConversionItem],
        max_in_flight: int,
        ordered: bool,
    ) -> Iterator[tuple[ConversionItem, ConvertDocumentResponse | Exception]]:
        return self._iterate_async_generator_sync(
            self._submit_and_retrieve_many_async(
                item_list=item_list,
                max_in_flight=max_in_flight,
                ordered=ordered,
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
                    name = str(source) if isinstance(source, str) else source.name
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
                    source_headers=headers,
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
        ):
            metadata = cast(_ConvertAllItemMetadata, item.metadata)
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
    ) -> AsyncGenerator[
        tuple[ConversionItem, ConvertDocumentResponse | Exception], None
    ]:
        async with self._build_async_http_client() as async_client:

            async def process_one(
                _idx: int,
                item: ConversionItem,
                async_client: httpx.AsyncClient,
            ) -> ConvertDocumentResponse:
                resolved = self._resolve_options(
                    options=item.options,
                    max_num_pages=None,
                    max_file_size=None,
                    page_range=None,
                )
                submit_options, _ = self._options_for_target_format(
                    resolved.options, OutputFormat.JSON
                )
                initial_status = await self._submit_convert_task_async(
                    source=item.source,
                    source_headers=item.source_headers,
                    options=submit_options,
                    async_client=async_client,
                    request_headers=item.headers,
                )
                terminal_status = await self._wait_for_terminal_status_for_submit_and_retrieve_many_async(
                    task_id=initial_status.task_id,
                    timeout=self._job_timeout,
                    async_client=async_client,
                    max_in_flight=max_in_flight,
                )
                return await self._fetch_convert_result_payload_async(
                    task_id=initial_status.task_id,
                    last_status=terminal_status,
                    async_client=async_client,
                )

            buffered_results: dict[
                int, tuple[ConversionItem, ConvertDocumentResponse | Exception]
            ] = {}
            next_ordered_index = 0
            async for idx, item, outcome in _run_bounded(
                items=item_list,
                process_one=process_one,
                async_client=async_client,
                max_in_flight=max_in_flight,
            ):
                normalized: ConvertDocumentResponse | Exception
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
        source_headers: dict[str, str] | None,
        options: ConvertDocumentsRequestOptions,
        async_client: httpx.AsyncClient,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        if isinstance(source, str):
            self._validate_http_source(source)
            request = ConvertDocumentsRequest(
                options=options,
                sources=[
                    HttpSourceRequest(
                        url=source,
                        headers={} if source_headers is None else source_headers,
                    )
                ],
                target=InBodyTarget(),
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
            data["target_type"] = "inbody"
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
        return ConvertDocumentResponse.model_validate_json(response.text)

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
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = self._http_client.request(
                    method=method,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
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
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = await async_client.request(
                    method=method,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
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
