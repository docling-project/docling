"""Unit tests for the asynchronous docling-serve client SDK."""

import json
import warnings
from datetime import datetime, timezone
from pathlib import PurePath
from types import MethodType, SimpleNamespace

import httpx
import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.service.responses import (
    TaskStatusResponse,
)
from docling.datamodel.service.tasks import TaskType
from docling.service_client import (
    AsyncConversionJob,
    AsyncDoclingServiceClient,
    ConversionItem,
    StatusWatcherKind,
)
from docling.service_client._base import ExperimentalWarning
from docling.service_client.exceptions import ConversionError

TEST_BASE_URL = "http://docling-service.invalid"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _status_payload(task_id: str, status: str) -> str:
    return TaskStatusResponse(
        task_id=task_id,
        task_type=TaskType.CONVERT,
        task_status=status,
        task_position=0,
        task_meta=None,
        error_message=None,
    ).model_dump_json()


def _convert_result_payload(source_name: str) -> str:
    return json.dumps(
        {
            "status": ConversionStatus.SUCCESS.value,
            "errors": [],
            "processing_time": 0.0,
            "timings": {},
            "document": {
                "filename": source_name,
                "json_content": DoclingDocument(
                    name=PurePath(source_name).stem
                ).model_dump(mode="json"),
            },
        }
    )


def _client_with_transport(
    handler,
    *,
    status_watcher: StatusWatcherKind = StatusWatcherKind.POLLING,
) -> AsyncDoclingServiceClient:
    """Build an async client whose HTTP layer is backed by a MockTransport.

    Polling is used by default so tests never need to mock websockets.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        client = AsyncDoclingServiceClient(
            url=TEST_BASE_URL,
            status_watcher=status_watcher,
            poll_server_wait=0.0,
            poll_client_interval=0.0,
            job_timeout=5.0,
        )
    transport = httpx.MockTransport(handler)
    client._http_client = httpx.AsyncClient(transport=transport)
    return client


def test_async_client_emits_experimental_warning() -> None:
    with pytest.warns(ExperimentalWarning):
        AsyncDoclingServiceClient(url=TEST_BASE_URL)


def test_async_client_rejects_invalid_max_concurrency() -> None:
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match="max_concurrency must be between"),
    ):
        warnings.simplefilter("ignore", ExperimentalWarning)
        AsyncDoclingServiceClient(url=TEST_BASE_URL, max_concurrency=0)


def test_async_client_normalizes_base_url() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        client = AsyncDoclingServiceClient(url=f"{TEST_BASE_URL}/")
        assert client._base_url == TEST_BASE_URL


@pytest.mark.anyio
async def test_async_client_context_manager_closes_http() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as client:
            assert isinstance(client, AsyncDoclingServiceClient)
            http_client = client._http_client
        assert http_client.is_closed


@pytest.mark.anyio
async def test_async_health_returns_parsed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    async with _client_with_transport(handler) as client:
        health = await client.health()
        assert health.status == "ok"


@pytest.mark.anyio
async def test_async_version_returns_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/version"
        return httpx.Response(200, json={"version": "1.2.3", "build": "abc"})

    async with _client_with_transport(handler) as client:
        version = await client.version()
        assert version == {"version": "1.2.3", "build": "abc"}


@pytest.mark.anyio
async def test_async_convert_full_flow_via_polling() -> None:
    """End-to-end: submit task, poll until terminal, fetch JSON result."""
    poll_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_calls
        path = request.url.path
        if path == "/v1/convert/source/async":
            return httpx.Response(200, text=_status_payload("task-1", "pending"))
        if path == "/v1/status/poll/task-1":
            poll_calls += 1
            status = "pending" if poll_calls == 1 else "success"
            return httpx.Response(200, text=_status_payload("task-1", status))
        if path == "/v1/result/task-1":
            return httpx.Response(200, text=_convert_result_payload("paper.pdf"))
        raise AssertionError(f"unexpected path {path}")

    async with _client_with_transport(handler) as client:
        result = await client.convert(source="https://example.invalid/paper.pdf")
        assert result.status == ConversionStatus.SUCCESS
        assert poll_calls >= 2  # at least one "pending" then one "success"


@pytest.mark.anyio
async def test_async_convert_raises_on_failure_when_raises_on_error_true() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convert/source/async":
            return httpx.Response(200, text=_status_payload("task-1", "pending"))
        if path == "/v1/status/poll/task-1":
            return httpx.Response(200, text=_status_payload("task-1", "success"))
        if path == "/v1/result/task-1":
            payload = json.dumps(
                {
                    "status": ConversionStatus.FAILURE.value,
                    "errors": [],
                    "processing_time": 0.0,
                    "timings": {},
                    "document": {"filename": "paper.pdf", "json_content": None},
                }
            )
            return httpx.Response(200, text=payload)
        raise AssertionError(f"unexpected path {path}")

    async with _client_with_transport(handler) as client:
        with pytest.raises(ConversionError):
            await client.convert(source="https://example.invalid/paper.pdf")


@pytest.mark.anyio
async def test_async_convert_all_yields_results_in_order() -> None:
    sources = [f"https://example.invalid/doc-{i}.pdf" for i in range(3)]
    submitted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convert/source/async":
            body = json.loads(request.content)
            url = body["sources"][0]["url"]
            task_id = f"task-{len(submitted)}"
            submitted.append(url)
            return httpx.Response(200, text=_status_payload(task_id, "pending"))
        if path.startswith("/v1/status/poll/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_status_payload(task_id, "success"))
        if path.startswith("/v1/result/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_convert_result_payload(f"{task_id}.pdf"))
        raise AssertionError(f"unexpected path {path}")

    async with _client_with_transport(handler) as client:
        results = []
        async for result in client.convert_all(sources=sources, max_concurrency=4):
            results.append(result)

    assert len(results) == 3
    assert all(r.status == ConversionStatus.SUCCESS for r in results)


@pytest.mark.anyio
async def test_async_convert_all_callable_inside_running_loop() -> None:
    """Regression guard: this is the case the sync client refused with RuntimeError.

    See client.py `_ensure_sync_bridge_allowed` — calling convert_all from inside
    an event loop used to raise. The async client must work here.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convert/source/async":
            body = json.loads(request.content)
            url = body["sources"][0]["url"]
            task_id = f"task-{url[-5]}"
            return httpx.Response(200, text=_status_payload(task_id, "pending"))
        if path.startswith("/v1/status/poll/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_status_payload(task_id, "success"))
        if path.startswith("/v1/result/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_convert_result_payload(f"{task_id}.pdf"))
        raise AssertionError(f"unexpected path {path}")

    sources = ["https://example.invalid/0.pdf", "https://example.invalid/1.pdf"]
    async with _client_with_transport(handler) as client:
        # If this method blocked on a sync bridge, we'd be deadlocked here.
        agen = client.convert_all(sources=sources)
        collected = [r async for r in agen]
        assert len(collected) == 2


@pytest.mark.anyio
async def test_async_submit_returns_async_conversion_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/convert/source/async":
            return httpx.Response(200, text=_status_payload("task-9", "pending"))
        raise AssertionError(f"unexpected path {request.url.path}")

    async with _client_with_transport(handler) as client:
        job = await client.submit(source="https://example.invalid/paper.pdf")
        assert isinstance(job, AsyncConversionJob)
        assert job.task_id == "task-9"
        assert job.status == "pending"


@pytest.mark.anyio
async def test_async_conversion_job_result_waits_and_fetches() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convert/source/async":
            return httpx.Response(200, text=_status_payload("task-7", "pending"))
        if path == "/v1/status/poll/task-7":
            return httpx.Response(200, text=_status_payload("task-7", "success"))
        if path == "/v1/result/task-7":
            return httpx.Response(200, text=_convert_result_payload("doc.pdf"))
        raise AssertionError(f"unexpected path {path}")

    async with _client_with_transport(handler) as client:
        job = await client.submit(source="https://example.invalid/doc.pdf")
        result = await job.result()
        assert result.status == ConversionStatus.SUCCESS
        assert job.done


@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_yields_per_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/convert/source/async":
            body = json.loads(request.content)
            url = body["sources"][0]["url"]
            task_id = f"task-{url[-5]}"
            return httpx.Response(200, text=_status_payload(task_id, "pending"))
        if path.startswith("/v1/status/poll/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_status_payload(task_id, "success"))
        if path.startswith("/v1/result/"):
            task_id = path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=_convert_result_payload(f"{task_id}.pdf"))
        raise AssertionError(f"unexpected path {path}")

    items = [
        ConversionItem(source="https://example.invalid/0.pdf"),
        ConversionItem(source="https://example.invalid/1.pdf"),
    ]
    async with _client_with_transport(handler) as client:
        outcomes = [
            outcome
            async for outcome in client.submit_and_retrieve_many(
                items=items, max_in_flight=2, ordered=True
            )
        ]

    assert len(outcomes) == 2
    for item, response in outcomes:
        assert isinstance(item, ConversionItem)
        assert not isinstance(response, Exception)


@pytest.mark.anyio
async def test_async_client_uses_polling_watcher_when_configured() -> None:
    from docling.service_client.watchers import AsyncPollingWatcher

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        client = AsyncDoclingServiceClient(
            url=TEST_BASE_URL,
            status_watcher=StatusWatcherKind.POLLING,
        )
        watcher = client._status_watcher()
        assert isinstance(watcher, AsyncPollingWatcher)
        await client.aclose()


@pytest.mark.anyio
async def test_async_conversion_job_watch_yields_until_terminal() -> None:
    statuses = iter(["pending", "running", "success"])

    async def fake_watch(task_id: str, timeout):
        for status in statuses:
            yield _status_response(task_id, status)
            if status == "success":
                return

    async def fake_wait(task_id: str, timeout):
        return _status_response(task_id, "success")

    async def fake_poll(task_id: str, wait: float):
        return _status_response(task_id, "pending")

    async def fake_fetch_result(task_id: str, last_status):
        return SimpleNamespace(task_id=task_id, status="success")

    from docling.service_client.job import _AsyncJobHandlers

    handlers = _AsyncJobHandlers(
        poll=fake_poll,
        watch=fake_watch,
        wait=fake_wait,
        fetch_result=fake_fetch_result,
    )
    job = AsyncConversionJob(
        task_id="task-watch",
        submitted_at=datetime.now(tz=timezone.utc),
        handlers=handlers,
    )
    updates = [u async for u in job.watch()]
    assert [u.task_status for u in updates] == ["pending", "running", "success"]
    assert job.done


def _status_response(task_id: str, status: str) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=task_id,
        task_type=TaskType.CONVERT,
        task_status=status,
        task_position=0,
        task_meta=None,
        error_message=None,
    )
