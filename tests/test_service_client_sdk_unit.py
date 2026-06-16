import asyncio
import io
import json
import queue
import tempfile
import threading
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import MethodType, SimpleNamespace

import httpx
import pytest
from docling_core.types.doc import (
    DoclingDocument,
    ImageRef,
    ImageRefMode,
    PictureItem,
)
from docling_core.types.doc.document import BoundingBox, ProvenanceItem, Size
from PIL import Image as PILImage

import docling.service_client.client as client_module
import docling.service_client.watchers as watchers_module
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    HttpSourceRequest,
    S3SourceRequest,
)
from docling.datamodel.service.responses import (
    FailureCategory,
    FailurePhase,
    MessageKind,
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
    PublicFailureInfo,
    TaskFailureResult,
    TaskStatusResponse,
    WebsocketMessage,
)
from docling.datamodel.service.targets import InBodyTarget, PresignedUrlTarget, S3Target
from docling.service_client import (
    DEFAULT_MAX_CONCURRENCY,
    MAX_CONCURRENCY_LIMIT,
    ConversionItem,
    DoclingServiceClient,
)
from docling.service_client.exceptions import (
    ArtifactDownloadError,
    ConversionError,
    ResponseSchemaMismatchError,
    ResultExpiredError,
    ServiceError,
    ServiceUnavailableError,
    TaskExecutionError,
    UsageLimitExceededError,
)
from docling.service_client.job import ConversionJob, _JobHandlers
from docling.service_client.watchers import PollingWatcher

TEST_BASE_URL = "http://docling-service.invalid"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _status_response(task_id: str, status: str) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=task_id,
        task_type="convert",
        task_status=status,
        task_position=0,
        task_meta=None,
        error_message=None,
    )


def _convert_payload(source_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=ConversionStatus.SUCCESS,
        errors=[],
        timings={},
        document=SimpleNamespace(
            filename=None,
            json_content=DoclingDocument(name=PurePath(source_name).stem),
        ),
    )


def test_base_url_accepts_root_with_or_without_trailing_slash() -> None:
    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        assert client._base_url == TEST_BASE_URL
        assert client._max_concurrency == DEFAULT_MAX_CONCURRENCY

    with DoclingServiceClient(url=f"{TEST_BASE_URL}/") as client:
        assert client._base_url == TEST_BASE_URL


@pytest.mark.parametrize("value", [0, -1, MAX_CONCURRENCY_LIMIT + 1])
def test_client_rejects_invalid_default_max_concurrency(value: int) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"max_concurrency must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}."
        ),
    ):
        DoclingServiceClient(url=TEST_BASE_URL, max_concurrency=value)


@pytest.mark.parametrize(
    "url",
    [
        f"{TEST_BASE_URL}/v1",
        f"{TEST_BASE_URL}/v1/",
        "http://proxy.example.com/docling/v1",
        "http://proxy.example.com/a/b/v1/",
    ],
)
def test_base_url_rejects_v1_path(url: str) -> None:
    with pytest.raises(ValueError):
        DoclingServiceClient(url=url)


def test_ws_status_url_is_derived_from_base_url() -> None:
    with DoclingServiceClient(url="https://example.org") as client:
        assert (
            client._build_ws_status_url("task-123")
            == "wss://example.org/v1/status/ws/task-123"
        )

    with DoclingServiceClient(url="http://example.org", api_key="k") as client:
        assert (
            client._build_ws_status_url("task-123")
            == "ws://example.org/v1/status/ws/task-123?api_key=k"
        )


def test_guess_input_format_uses_docling_extension_map() -> None:
    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        assert client._guess_input_format("doc.asc") == InputFormat.ASCIIDOC
        assert client._guess_input_format("subtitles.vtt") == InputFormat.VTT
        assert client._guess_input_format("archive.tar.gz") == InputFormat.METS_GBS


def test_result_404_after_failed_status_raises_conversion_error() -> None:
    response = httpx.Response(
        404,
        json={"detail": "Task result not found. Please wait for a completion status."},
    )
    last_status = _status_response("task-1", "failure")
    last_status.error_message = "conversion failed upstream"

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        with pytest.raises(TaskExecutionError, match="conversion failed upstream"):
            client._raise_for_result_404(
                task_id="task-1",
                response=response,
                last_status=last_status,
            )


def test_fetch_result_response_raises_task_execution_error_for_failure_payload() -> (
    None
):
    response = httpx.Response(
        200,
        json=TaskFailureResult(
            failure=PublicFailureInfo(
                category=FailureCategory.INTERNAL,
                message="Internal processing error.",
                retryable=False,
                phase=FailurePhase.ORCHESTRATION,
            )
        ).model_dump(mode="json"),
        headers={"content-type": "application/json"},
    )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._request_with_retry = MethodType(  # type: ignore[method-assign]
            lambda self, **kwargs: response,
            client,
        )
        with pytest.raises(TaskExecutionError, match=r"Internal processing error\."):
            client._fetch_result_response(
                task_id="task-1",
                last_status=_status_response("task-1", "failure"),
                error_message="fetch failed",
            )


def test_result_404_after_success_status_raises_result_expired() -> None:
    response = httpx.Response(
        404,
        json={"detail": "Task result not found. Please wait for a completion status."},
    )
    last_status = _status_response("task-1", "success")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        with pytest.raises(ResultExpiredError, match="task-1"):
            client._raise_for_result_404(
                task_id="task-1",
                response=response,
                last_status=last_status,
            )


def test_conversion_job_poll_forwards_wait_and_updates_done() -> None:
    seen_waits: list[float] = []

    def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        seen_waits.append(wait)
        return _status_response(task_id=task_id, status="success")

    job = ConversionJob(
        task_id="task-1",
        submitted_at=datetime.now(timezone.utc),
        handlers=_JobHandlers(
            poll=fake_poll,
            watch=lambda task_id, timeout: iter(()),
            wait=lambda task_id, timeout: _status_response(
                task_id=task_id, status="success"
            ),
            fetch_result=lambda task_id, last_status: "done",
        ),
    )

    assert job.done is False
    update = job.poll(wait=1.25)

    assert update.task_status == "success"
    assert seen_waits == [1.25]
    assert job.done is True


def test_polling_watcher_enforces_minimum_client_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleep_calls: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(watchers_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(watchers_module.time, "sleep", clock.sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    watcher = PollingWatcher(
        poll_status=fake_poll,
        poll_server_wait=0.5,
        poll_client_interval=None,
        default_timeout=5.0,
    )
    updates = list(watcher.iter_updates(task_id="task-1"))

    assert [update.task_status for update in updates] == ["pending", "success"]
    assert wait_values == [0.5, 0.5]
    assert clock.sleep_calls == [pytest.approx(0.5)]


def test_polling_watcher_supports_explicit_client_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleep_calls: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(watchers_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(watchers_module.time, "sleep", clock.sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    watcher = PollingWatcher(
        poll_status=fake_poll,
        poll_server_wait=5.0,
        poll_client_interval=1.0,
        default_timeout=10.0,
    )
    updates = list(watcher.iter_updates(task_id="task-1"))

    assert [update.task_status for update in updates] == ["pending", "success"]
    assert wait_values == [5.0, 5.0]
    assert clock.sleep_calls == [pytest.approx(1.0)]


def test_websocket_watcher_treats_clean_close_on_next_as_end_of_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnectionClosedOK(Exception):
        pass

    class FakeWebSocket:
        def __init__(self) -> None:
            self._messages = iter(
                [
                    WebsocketMessage(
                        message=MessageKind.CONNECTION,
                        task=_status_response("task-1", "pending"),
                    ).model_dump_json(),
                    WebsocketMessage(
                        message=MessageKind.UPDATE,
                        task=_status_response("task-1", "pending"),
                    ).model_dump_json(),
                ]
            )

        def recv(self, timeout: float | None = None) -> str:
            return next(self._messages)

        def send(self, message: str) -> None:
            raise FakeConnectionClosedOK

    class FakeConnection:
        def __enter__(self) -> FakeWebSocket:
            return FakeWebSocket()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(watchers_module, "ConnectionClosedOK", FakeConnectionClosedOK)
    monkeypatch.setattr(
        watchers_module, "connect", lambda *args, **kwargs: FakeConnection()
    )

    watcher = watchers_module.WebSocketWatcher(
        ws_url_for_task=lambda task_id: f"ws://example.invalid/{task_id}",
        poll_fallback=None,
        fallback_to_poll=False,
        connect_timeout=1.0,
        default_timeout=10.0,
    )

    updates = list(watcher.iter_updates(task_id="task-1"))

    assert [update.task_status for update in updates] == ["pending", "pending"]


def test_websocket_watcher_reconnects_after_connection_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnectionClosedError(Exception):
        pass

    class FakeConnectionClosedOK(Exception):
        pass

    connection_calls: list[int] = []

    class FirstConnection:
        def __init__(self) -> None:
            self._consumed = False

        def __enter__(self) -> "FirstConnection":
            return self

        def recv(self, timeout: float | None = None) -> str:
            if self._consumed:
                raise FakeConnectionClosedError("connection reset")
            self._consumed = True
            return WebsocketMessage(
                message=MessageKind.CONNECTION,
                task=_status_response("task-1", "pending"),
            ).model_dump_json()

        def send(self, message: str) -> None:
            pass

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class SecondConnection:
        def __init__(self) -> None:
            self._messages = iter(
                [
                    WebsocketMessage(
                        message=MessageKind.CONNECTION,
                        task=_status_response("task-1", "pending"),
                    ).model_dump_json(),
                    WebsocketMessage(
                        message=MessageKind.UPDATE,
                        task=_status_response("task-1", "success"),
                    ).model_dump_json(),
                ]
            )

        def recv(self, timeout: float | None = None) -> str:
            return next(self._messages)

        def send(self, message: str) -> None:
            pass

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __enter__(self) -> "SecondConnection":
            return self

    connections = iter([FirstConnection(), SecondConnection()])

    def fake_connect(*args, **kwargs):
        connection_calls.append(1)
        return next(connections)

    monkeypatch.setattr(
        watchers_module, "ConnectionClosedError", FakeConnectionClosedError
    )
    monkeypatch.setattr(watchers_module, "ConnectionClosedOK", FakeConnectionClosedOK)
    monkeypatch.setattr(watchers_module, "connect", fake_connect)
    monkeypatch.setattr(watchers_module.time, "sleep", lambda _: None)

    watcher = watchers_module.WebSocketWatcher(
        ws_url_for_task=lambda task_id: f"ws://example.invalid/{task_id}",
        poll_fallback=None,
        fallback_to_poll=False,
        connect_timeout=1.0,
        default_timeout=10.0,
    )

    updates = list(watcher.iter_updates(task_id="task-1"))

    assert len(connection_calls) == 2
    assert [u.task_status for u in updates] == ["pending", "pending", "success"]


def test_websocket_watcher_raises_after_max_reconnect_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnectionClosedError(Exception):
        pass

    class FakeConnectionClosedOK(Exception):
        pass

    connection_calls: list[int] = []

    class DroppingConnection:
        def __enter__(self) -> "DroppingConnection":
            return self

        def recv(self, timeout: float | None = None) -> str:
            raise FakeConnectionClosedError("connection reset")

        def send(self, message: str) -> None:
            pass

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_connect(*args, **kwargs):
        connection_calls.append(1)
        return DroppingConnection()

    monkeypatch.setattr(
        watchers_module, "ConnectionClosedError", FakeConnectionClosedError
    )
    monkeypatch.setattr(watchers_module, "ConnectionClosedOK", FakeConnectionClosedOK)
    monkeypatch.setattr(watchers_module, "connect", fake_connect)
    monkeypatch.setattr(watchers_module.time, "sleep", lambda _: None)

    watcher = watchers_module.WebSocketWatcher(
        ws_url_for_task=lambda task_id: f"ws://example.invalid/{task_id}",
        poll_fallback=None,
        fallback_to_poll=False,
        connect_timeout=1.0,
        default_timeout=10.0,
    )

    with pytest.raises(watchers_module.ServiceUnavailableError):
        list(watcher.iter_updates(task_id="task-1"))

    assert len(connection_calls) == watchers_module.WS_MAX_RECONNECT_ATTEMPTS + 1


@pytest.mark.anyio
async def test_async_wait_for_terminal_enforces_minimum_client_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleep_calls: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock.monotonic)

    async def fake_async_sleep(seconds: float) -> None:
        clock.advance(seconds)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_async_sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    async def fake_poll(
        self,
        task_id: str,
        wait: float,
        async_client,
    ) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    with DoclingServiceClient(url=TEST_BASE_URL, poll_server_wait=0.5) as client:
        client._poll_task_status_async = MethodType(fake_poll, client)
        result = await client._wait_for_terminal_status_async(
            task_id="task-1",
            timeout=5.0,
            async_client=None,  # type: ignore[arg-type]
        )

    assert result.task_status == "success"
    assert wait_values == [0.5, 0.5]
    assert clock.sleep_calls == [pytest.approx(0.5)]


@pytest.mark.anyio
async def test_async_wait_for_terminal_supports_explicit_client_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleep_calls: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.sleep_calls.append(seconds)
            self.now += seconds

    clock = _Clock()
    monkeypatch.setattr(client_module.time, "monotonic", clock.monotonic)

    async def fake_async_sleep(seconds: float) -> None:
        clock.advance(seconds)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_async_sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    async def fake_poll(
        self,
        task_id: str,
        wait: float,
        async_client,
    ) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        poll_server_wait=5.0,
        poll_client_interval=1.0,
    ) as client:
        client._poll_task_status_async = MethodType(fake_poll, client)
        result = await client._wait_for_terminal_status_async(
            task_id="task-1",
            timeout=10.0,
            async_client=None,  # type: ignore[arg-type]
        )

    assert result.task_status == "success"
    assert wait_values == [5.0, 5.0]
    assert clock.sleep_calls == [pytest.approx(1.0)]


def test_convert_all_uses_async_pipeline_and_preserves_order(tmp_path) -> None:
    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        calls: list[dict[str, object]] = []

        async def fake_submit_and_retrieve_many_async(
            self,
            item_list,
            max_in_flight,
            ordered,
            target=InBodyTarget(),
        ):
            items = list(item_list)
            calls.append(
                {
                    "count": len(items),
                    "max_in_flight": max_in_flight,
                    "ordered": ordered,
                    "request_headers": [item.headers for item in items],
                }
            )
            for item in items:
                yield item, _convert_payload(Path(item.source).name)

        client._submit_and_retrieve_many_async = MethodType(
            fake_submit_and_retrieve_many_async, client
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p3 = tmp_path / "c.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")
        p3.write_bytes(b"%PDF-1.4\n")

        results = list(
            client.convert_all(
                [p1, p2, p3],
                headers={"Authorization": "Bearer source-token"},
                options=ConvertDocumentsRequestOptions(),
                max_concurrency=64,
            )
        )

    assert calls == [
        {
            "count": 3,
            "max_in_flight": 64,
            "ordered": True,
            "request_headers": [
                {"Authorization": "Bearer source-token"},
                {"Authorization": "Bearer source-token"},
                {"Authorization": "Bearer source-token"},
            ],
        }
    ]
    assert [result.input.file.name for result in results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(result.status == ConversionStatus.SUCCESS for result in results)


def test_convert_all_returns_iterator_and_yields_before_batch_completion(
    tmp_path: Path,
) -> None:
    release_first = threading.Event()
    release_third = threading.Event()

    with DoclingServiceClient(url=TEST_BASE_URL) as client:

        async def fake_submit_and_retrieve_many_async(
            self,
            item_list,
            max_in_flight,
            ordered,
            target=InBodyTarget(),
        ):
            items = list(item_list)
            assert max_in_flight == 2
            assert ordered is True

            await asyncio.to_thread(release_first.wait)
            yield items[0], _convert_payload(Path(items[0].source).name)
            yield items[1], _convert_payload(Path(items[1].source).name)

            await asyncio.to_thread(release_third.wait)
            yield items[2], _convert_payload(Path(items[2].source).name)

        client._submit_and_retrieve_many_async = MethodType(
            fake_submit_and_retrieve_many_async, client
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p3 = tmp_path / "c.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")
        p3.write_bytes(b"%PDF-1.4\n")

        iterator_queue: queue.Queue[object] = queue.Queue(maxsize=1)

        def build_iterator() -> None:
            try:
                iterator_queue.put(
                    client.convert_all(
                        [p1, p2, p3],
                        options=ConvertDocumentsRequestOptions(),
                        max_concurrency=2,
                    )
                )
            except BaseException as exc:
                iterator_queue.put(exc)

        build_thread = threading.Thread(target=build_iterator)
        build_thread.start()

        iterator_or_exc = iterator_queue.get(timeout=0.2)
        build_thread.join(timeout=0.2)

        if isinstance(iterator_or_exc, BaseException):
            raise iterator_or_exc

        assert build_thread.is_alive() is False
        iterator = iterator_or_exc

        first_result_queue: queue.Queue[object] = queue.Queue(maxsize=1)

        def consume_first_result() -> None:
            try:
                first_result_queue.put(next(iterator))
            except BaseException as exc:
                first_result_queue.put(exc)

        first_thread = threading.Thread(target=consume_first_result)
        first_thread.start()
        time.sleep(0.05)
        assert first_result_queue.empty()

        release_first.set()
        first_result_or_exc = first_result_queue.get(timeout=0.2)
        first_thread.join(timeout=0.2)

        if isinstance(first_result_or_exc, BaseException):
            raise first_result_or_exc

        first_result = first_result_or_exc
        assert first_thread.is_alive() is False
        assert first_result.input.file.name == "a.pdf"
        assert next(iterator).input.file.name == "b.pdf"

        release_third.set()
        assert next(iterator).input.file.name == "c.pdf"

        with pytest.raises(StopIteration):
            next(iterator)


def test_convert_all_interleaves_preflight_skips_correctly(tmp_path: Path) -> None:
    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        submitted_names: list[str] = []

        async def fake_submit_and_retrieve_many_async(
            self,
            item_list,
            max_in_flight,
            ordered,
            target=InBodyTarget(),
        ):
            items = list(item_list)
            assert max_in_flight == DEFAULT_MAX_CONCURRENCY
            assert ordered is True

            for item in items:
                submitted_names.append(Path(item.source).name)
                yield item, _convert_payload(Path(item.source).name)

        client._submit_and_retrieve_many_async = MethodType(
            fake_submit_and_retrieve_many_async, client
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p3 = tmp_path / "c.pdf"
        p4 = tmp_path / "d.pdf"
        p1.write_bytes(b"aa")
        p2.write_bytes(b"b")
        p3.write_bytes(b"cc")
        p4.write_bytes(b"d")

        results = list(client.convert_all([p1, p2, p3, p4], max_file_size=1))

    assert submitted_names == ["b.pdf", "d.pdf"]
    assert [result.input.file.name for result in results] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
        "d.pdf",
    ]
    assert [result.status for result in results] == [
        ConversionStatus.SKIPPED,
        ConversionStatus.SUCCESS,
        ConversionStatus.SKIPPED,
        ConversionStatus.SUCCESS,
    ]
    assert "max_file_size" in results[0].errors[0].error_message
    assert "max_file_size" in results[2].errors[0].error_message


@pytest.mark.parametrize("value", [0, -1, MAX_CONCURRENCY_LIMIT + 1])
def test_convert_all_rejects_invalid_max_concurrency(
    tmp_path: Path, value: int
) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        with pytest.raises(
            ValueError,
            match=(
                f"max_concurrency must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}."
            ),
        ):
            list(client.convert_all([source], max_concurrency=value))


def test_submit_and_retrieve_many_yields_completion_order_and_ordered_mode(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            if task_id == "task-a.pdf":
                await asyncio.sleep(0.01)
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")

        completion_order = [
            Path(item.source).name
            for item, _ in client.submit_and_retrieve_each(
                [ConversionItem(source=p1), ConversionItem(source=p2)],
                max_in_flight=2,
                target=InBodyTarget(),
            )
        ]
        ordered_names = [
            Path(item.source).name
            for item, _ in client.submit_and_retrieve_each(
                [ConversionItem(source=p1), ConversionItem(source=p2)],
                max_in_flight=2,
                ordered=True,
                target=InBodyTarget(),
            )
        ]

    assert completion_order == ["b.pdf", "a.pdf"]
    assert ordered_names == ["a.pdf", "b.pdf"]


def test_submit_and_retrieve_many_forwards_per_item_request_headers(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    seen_headers: list[dict[str, str] | None] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            seen_headers.append(request_headers)
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")

        list(
            client.submit_and_retrieve_each(
                [
                    ConversionItem(source=p1, headers={"X-Tenant-Id": "tenant-a"}),
                    ConversionItem(source=p2, headers={"X-Tenant-Id": "tenant-b"}),
                ],
                max_in_flight=2,
                target=InBodyTarget(),
            )
        )

    assert seen_headers == [
        {"X-Tenant-Id": "tenant-a"},
        {"X-Tenant-Id": "tenant-b"},
    ]


def test_submit_and_retrieve_many_forwards_http_source_request_headers() -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    seen_sources: list[HttpSourceRequest] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            assert isinstance(source, HttpSourceRequest)
            seen_sources.append(source)
            return _status_response(
                f"task-{Path(str(source.url)).name}",
                "pending",
            )

        async def fake_wait(self, task_id, timeout, async_client):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        list(
            client.submit_and_retrieve_each(
                [
                    ConversionItem(
                        source=HttpSourceRequest(
                            url="https://example.org/a.pdf",
                            headers={"Authorization": "Bearer a"},
                        ),
                    ),
                    ConversionItem(
                        source=HttpSourceRequest(
                            url="https://example.org/b.pdf",
                            headers={"Authorization": "Bearer b"},
                        ),
                    ),
                ],
                max_in_flight=2,
                target=InBodyTarget(),
            )
        )

    assert [item.headers for item in seen_sources] == [
        {"Authorization": "Bearer a"},
        {"Authorization": "Bearer b"},
    ]


def test_submit_and_retrieve_many_isolates_failures_per_item(tmp_path: Path) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            if source.name == "bad.pdf":
                raise ValueError("submit failed")
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        good = tmp_path / "good.pdf"
        bad = tmp_path / "bad.pdf"
        good.write_bytes(b"%PDF-1.4\n")
        bad.write_bytes(b"%PDF-1.4\n")

        outcomes = sorted(
            client.submit_and_retrieve_each(
                [ConversionItem(source=bad), ConversionItem(source=good)],
                max_in_flight=2,
                target=InBodyTarget(),
            ),
            key=lambda entry: Path(entry[0].source).name,
        )

    assert isinstance(outcomes[0][1], Exception)
    assert str(outcomes[0][1]) == "submit failed"
    assert getattr(outcomes[1][1], "status") == ConversionStatus.SUCCESS


def test_submit_and_retrieve_many_respects_max_in_flight(tmp_path: Path) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    state = {"active": 0, "max_seen": 0, "submitted": 0}
    release = asyncio.Event()

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            state["active"] += 1
            state["submitted"] += 1
            state["max_seen"] = max(state["max_seen"], state["active"])
            if state["submitted"] >= 2:
                release.set()
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            if task_id in {"task-a.pdf", "task-b.pdf"}:
                await release.wait()
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            state["active"] -= 1
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        paths = []
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            path = tmp_path / name
            path.write_bytes(b"%PDF-1.4\n")
            paths.append(path)

        list(
            client.submit_and_retrieve_each(
                [ConversionItem(source=path) for path in paths],
                max_in_flight=2,
                target=InBodyTarget(),
            )
        )

    assert state["max_seen"] == 2


def test_submit_and_retrieve_many_consumes_iterable_incrementally(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    generated: list[str] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            await asyncio.sleep(
                0
            )  # yield so the event loop can process completed results
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        def item_iter():
            for idx in range(50):
                path = tmp_path / f"{idx}.pdf"
                path.write_bytes(b"%PDF-1.4\n")
                generated.append(path.name)
                yield ConversionItem(source=path)

        iterator = client.submit_and_retrieve_each(
            item_iter(),
            max_in_flight=1,
            target=InBodyTarget(),
        )
        assert generated == []

        first_item, _ = next(iterator)
        assert Path(first_item.source).name == "0.pdf"
        assert len(generated) < 50

        remaining_names = [Path(item.source).name for item, _ in iterator]

    assert remaining_names[-1] == "49.pdf"
    assert len(generated) == 50


@pytest.mark.parametrize("value", [0, -1, MAX_CONCURRENCY_LIMIT + 1])
def test_submit_and_retrieve_many_rejects_invalid_max_in_flight(
    tmp_path: Path, value: int
) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        with pytest.raises(
            ValueError,
            match=(
                f"max_in_flight must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}."
            ),
        ):
            client.submit_and_retrieve_each(
                [ConversionItem(source=source)],
                max_in_flight=value,
                target=InBodyTarget(),
            )


@pytest.mark.anyio
async def test_submit_and_retrieve_many_prefers_websocket_wait_at_or_below_threshold() -> (
    None
):
    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="websocket",
    ) as client:
        seen: list[tuple[str, float | None]] = []

        def fake_wait_for_terminal(
            task_id: str, timeout: float | None
        ) -> TaskStatusResponse:
            seen.append((task_id, timeout))
            return _status_response(task_id, "success")

        client._ws_watcher.wait_for_terminal = fake_wait_for_terminal  # type: ignore[method-assign]

        result = await client._wait_for_terminal_status_for_submit_and_retrieve_many_async(
            task_id="task-1",
            timeout=3.0,
            async_client=None,  # type: ignore[arg-type]
            max_in_flight=client_module.SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS,
        )

    assert result.task_status == "success"
    assert seen == [("task-1", 3.0)]


@pytest.mark.anyio
async def test_submit_and_retrieve_many_uses_polling_above_websocket_threshold() -> (
    None
):
    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="websocket",
    ) as client:
        seen: list[tuple[str, float, object]] = []

        async def fake_wait_async(self, task_id, timeout, async_client):
            seen.append((task_id, timeout, async_client))
            return _status_response(task_id, "success")

        def fail_wait_for_terminal(
            task_id: str, timeout: float | None
        ) -> TaskStatusResponse:
            raise AssertionError("websocket wait should not be used above threshold")

        client._wait_for_terminal_status_async = MethodType(fake_wait_async, client)
        client._ws_watcher.wait_for_terminal = fail_wait_for_terminal  # type: ignore[method-assign]

        marker = object()
        result = (
            await client._wait_for_terminal_status_for_submit_and_retrieve_many_async(
                task_id="task-2",
                timeout=4.0,
                async_client=marker,  # type: ignore[arg-type]
                max_in_flight=(
                    client_module.SUBMIT_AND_RETRIEVE_MANY_MAX_IN_FLIGHT_WEBSOCKETS + 1
                ),
            )
        )

    assert result.task_status == "success"
    assert seen == [("task-2", 4.0, marker)]


@pytest.mark.anyio
async def test_submit_and_retrieve_many_respects_explicit_polling_watcher_for_waits() -> (
    None
):
    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:
        seen: list[tuple[str, float, object]] = []

        async def fake_wait_async(self, task_id, timeout, async_client):
            seen.append((task_id, timeout, async_client))
            return _status_response(task_id, "success")

        def fail_wait_for_terminal(
            task_id: str, timeout: float | None
        ) -> TaskStatusResponse:
            raise AssertionError(
                "websocket wait should not be used for polling watcher"
            )

        client._wait_for_terminal_status_async = MethodType(fake_wait_async, client)
        client._ws_watcher.wait_for_terminal = fail_wait_for_terminal  # type: ignore[method-assign]

        marker = object()
        result = (
            await client._wait_for_terminal_status_for_submit_and_retrieve_many_async(
                task_id="task-3",
                timeout=5.0,
                async_client=marker,  # type: ignore[arg-type]
                max_in_flight=1,
            )
        )

    assert result.task_status == "success"
    assert seen == [("task-3", 5.0, marker)]


@pytest.mark.anyio
async def test_submit_and_retrieve_many_ordered_mode_yields_before_batch_completion(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    release_first = asyncio.Event()
    release_third = asyncio.Event()

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            if task_id == "task-a.pdf":
                await release_first.wait()
            if task_id == "task-c.pdf":
                await release_third.wait()
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        items: list[ConversionItem] = []
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            path = tmp_path / name
            path.write_bytes(b"%PDF-1.4\n")
            items.append(ConversionItem(source=path))

        async_iterator = client._submit_and_retrieve_many_async(
            item_list=items,
            max_in_flight=2,
            ordered=True,
        )

        first_result_task = asyncio.create_task(anext(async_iterator))
        await asyncio.sleep(0)
        release_first.set()

        first_item, _ = await asyncio.wait_for(first_result_task, timeout=0.2)
        second_item, _ = await asyncio.wait_for(anext(async_iterator), timeout=0.2)

        assert Path(first_item.source).name == "a.pdf"
        assert Path(second_item.source).name == "b.pdf"

        release_third.set()
        third_item, _ = await asyncio.wait_for(anext(async_iterator), timeout=0.2)
        assert Path(third_item.source).name == "c.pdf"

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(async_iterator), timeout=0.2)


def test_submit_url_forwards_request_headers() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["header_tenant"] = request.headers.get("X-Tenant-Id")
        captured["header_api"] = request.headers.get("X-Api-Key")
        captured["payload"] = request.content.decode("utf-8")
        return httpx.Response(
            200, json=_status_response("task-1", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL, api_key="base-key") as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            headers={"X-Api-Key": "base-key"},
            timeout=client._http_client.timeout,
        )
        job = client.submit(
            source="https://example.org/sample.pdf",
            options=ConvertDocumentsRequestOptions(),
            headers={"X-Tenant-Id": "tenant-a"},
        )

    assert job.task_id == "task-1"
    assert captured["path"] == "/v1/convert/source/async"
    assert captured["header_tenant"] == "tenant-a"
    assert captured["header_api"] == "base-key"


def test_submit_file_forwards_request_headers(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["header_tenant"] = request.headers.get("X-Tenant-Id")
        captured["header_api"] = request.headers.get("X-Api-Key")
        return httpx.Response(
            200, json=_status_response("task-2", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL, api_key="base-key") as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            headers={"X-Api-Key": "base-key"},
            timeout=client._http_client.timeout,
        )
        job = client.submit(
            source=sample,
            options=ConvertDocumentsRequestOptions(),
            headers={"X-Tenant-Id": "tenant-b"},
        )

    assert job.task_id == "task-2"
    assert captured["path"] == "/v1/convert/file/async"
    assert captured["header_tenant"] == "tenant-b"
    assert captured["header_api"] == "base-key"


def test_serialize_convert_options_omits_defaults_and_none() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with DoclingServiceClient(url=TEST_BASE_URL) as client:
            payload = client._serialize_convert_options(
                ConvertDocumentsRequestOptions(
                    page_range=(3, 7),
                    document_timeout=None,
                )
            )

    assert payload == {"page_range": [3, 7]}
    assert all(
        "PydanticSerializationUnexpectedValue" not in str(warning.message)
        for warning in caught
    )


def test_submit_source_serializes_convert_options_without_defaults() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, json=_status_response("task-source", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            timeout=client._http_client.timeout,
        )
        job = client.submit(
            source="https://example.org/sample.pdf",
            options=ConvertDocumentsRequestOptions(
                page_range=(3, 7),
                document_timeout=None,
            ),
        )

    assert job.task_id == "task-source"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["options"] == {"page_range": [3, 7]}


def test_submit_file_serializes_convert_options_without_defaults(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def fake_request_with_retry(**kw: object) -> httpx.Response:
        captured["data"] = kw["data"]
        captured["files"] = kw["files"]
        return httpx.Response(
            200, json=_status_response("task-file", "pending").model_dump(mode="json")
        )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._request_with_retry = fake_request_with_retry  # type: ignore[method-assign]
        status = client._submit_convert_task(
            source=sample,
            options=ConvertDocumentsRequestOptions(
                page_range=(3, 7),
                document_timeout=None,
            ),
            target=InBodyTarget(),
        )

    assert status.task_id == "task-file"
    assert captured["data"] == {
        "page_range": [3, 7],
        "target_type": "inbody",
    }


def test_submit_accepts_http_source_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.content.decode("utf-8")
        return httpx.Response(
            200,
            json=_status_response("task-http-source", "pending").model_dump(
                mode="json"
            ),
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            timeout=client._http_client.timeout,
        )
        job = client.submit(
            source=HttpSourceRequest(
                url="https://example.org/sample.pdf",
                headers={"Authorization": "Bearer source-token"},
            ),
            target=InBodyTarget(),
        )

    assert job.task_id == "task-http-source"
    assert '"headers":{"Authorization":"Bearer source-token"}' in str(
        captured["payload"]
    )


def test_submit_auto_falls_back_to_inbody_when_presigned_is_rejected() -> None:
    seen_targets: list[object] = []
    seen_formats: list[list[OutputFormat]] = []

    def fake_submit_convert_task(
        self,
        source,
        options,
        target,
        request_headers=None,
    ):
        seen_targets.append(target)
        seen_formats.append(list(options.to_formats))
        if isinstance(target, PresignedUrlTarget):
            raise ServiceError(
                "Task submission failed.",
                status_code=422,
                detail=(
                    "Presigned URL target requires artifact storage to be configured "
                    "and enabled on the server."
                ),
            )
        return _status_response("task-fallback", "pending")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit_convert_task, client)
        job = client.submit(
            source="https://example.org/sample.pdf",
            output_formats=[OutputFormat.MARKDOWN],
        )

    assert job.task_id == "task-fallback"
    assert [type(item) for item in seen_targets] == [PresignedUrlTarget, InBodyTarget]
    assert seen_formats == [
        [OutputFormat.MARKDOWN],
        [OutputFormat.MARKDOWN, OutputFormat.JSON],
    ]


def test_submit_and_retrieve_each_auto_target_returns_presigned_when_supported(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    seen_targets: list[type[object]] = []
    presigned_result = PresignedUrlConvertResponse(
        num_converted=1,
        num_succeeded=1,
        num_partially_succeeded=0,
        num_failed=0,
        processing_time=0.25,
        documents=[
            {
                "source_index": 0,
                "source_uri": "https://example.org/sample.pdf",
                "filename": "sample.pdf",
                "status": "success",
                "artifacts": [
                    {
                        "artifact_type": "markdown",
                        "mime_type": "text/markdown",
                        "uri": "https://download.example.org/sample.md",
                    }
                ],
            }
        ],
    )

    with DoclingServiceClient(url=TEST_BASE_URL, status_watcher="polling") as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            seen_targets.append(type(target))
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            return _status_response(task_id, "success")

        async def fake_fetch_presigned(self, task_id, last_status, async_client):
            return presigned_result

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_presigned_result_async = MethodType(fake_fetch_presigned, client)

        source = tmp_path / "a.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        outcomes = list(
            client.submit_and_retrieve_each(
                [ConversionItem(source=source)],
                max_in_flight=1,
            )
        )

    assert seen_targets == [PresignedUrlTarget]
    assert outcomes[0][1] is presigned_result


def test_submit_and_retrieve_each_auto_target_falls_back_to_inbody_when_presigned_is_rejected(
    tmp_path: Path,
) -> None:
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    seen_targets: list[type[object]] = []

    with DoclingServiceClient(url=TEST_BASE_URL, status_watcher="polling") as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, options, target, async_client, request_headers=None
        ):
            seen_targets.append(type(target))
            if isinstance(target, PresignedUrlTarget):
                raise ServiceError(
                    "Task submission failed.",
                    status_code=422,
                    detail=(
                        "Presigned URL target requires artifact storage to be configured "
                        "and enabled on the server."
                    ),
                )
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(self, task_id, timeout, async_client):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(self, task_id, last_status, async_client):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_http_client = MethodType(
            fake_build_async_http_client, client
        )
        client._submit_convert_task_async = MethodType(fake_submit, client)
        client._wait_for_terminal_status_async = MethodType(fake_wait, client)
        client._fetch_convert_result_payload_async = MethodType(
            fake_fetch_payload, client
        )

        source = tmp_path / "a.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        outcomes = list(
            client.submit_and_retrieve_each(
                [ConversionItem(source=source)],
                max_in_flight=1,
            )
        )

    assert seen_targets == [PresignedUrlTarget, InBodyTarget]
    assert getattr(outcomes[0][1], "status") == ConversionStatus.SUCCESS


def test_submit_batch_posts_batch_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = request.content.decode("utf-8")
        return httpx.Response(
            200, json=_status_response("task-batch", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            timeout=client._http_client.timeout,
        )
        job = client.submit_batch(
            sources=[
                HttpSourceRequest(
                    url="https://example.org/sample.pdf",
                    headers={"Authorization": "Bearer source-token"},
                )
            ],
            output_formats=[OutputFormat.MARKDOWN],
            target=PresignedUrlTarget(),
        )

    assert job.task_id == "task-batch"
    assert captured["path"] == "/v1/convert/source/batch"
    assert '"kind":"presigned_url"' in str(captured["payload"])
    assert '"headers":{"Authorization":"Bearer source-token"}' in str(
        captured["payload"]
    )


def test_submit_batch_returns_presigned_result_for_presigned_target() -> None:
    presigned_result = PresignedUrlConvertResponse(
        num_converted=1,
        num_succeeded=1,
        num_partially_succeeded=0,
        num_failed=0,
        processing_time=0.25,
        documents=[
            {
                "source_index": 0,
                "source_uri": "https://example.org/sample.pdf",
                "filename": "sample.pdf",
                "status": "success",
                "artifacts": [
                    {
                        "artifact_type": "markdown",
                        "mime_type": "text/markdown",
                        "uri": "https://download.example.org/sample.md",
                    }
                ],
            }
        ],
    )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_batch_task = MethodType(
            lambda self, sources, options, target, request_headers=None: (
                _status_response("task-presigned-batch", "pending")
            ),
            client,
        )
        client._wait_for_terminal_status = MethodType(
            lambda self, task_id, timeout: _status_response(task_id, "success"),
            client,
        )
        client._fetch_presigned_result = MethodType(
            lambda self, task_id, last_status: presigned_result,
            client,
        )

        job = client.submit_batch(
            sources=[AnyHttpSourceRequest(url="https://example.org/sample.pdf")],
            target=PresignedUrlTarget(),
        )
        result = job.result(timeout=1.0)

    assert result is presigned_result


def test_submit_batch_returns_counts_result_for_s3_target() -> None:
    s3_result = PresignedUrlConvertDocumentResponse(
        num_converted=1,
        num_succeeded=1,
        num_partially_succeeded=0,
        num_failed=0,
        processing_time=0.25,
    )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_batch_task = MethodType(
            lambda self, sources, options, target, request_headers=None: (
                _status_response("task-s3", "pending")
            ),
            client,
        )
        client._wait_for_terminal_status = MethodType(
            lambda self, task_id, timeout: _status_response(task_id, "success"),
            client,
        )
        client._fetch_presigned_document_result = MethodType(
            lambda self, task_id, last_status: s3_result,
            client,
        )

        job = client.submit_batch(
            sources=[
                S3SourceRequest(
                    endpoint="s3.example.org",
                    access_key="a",
                    secret_key="b",
                    bucket="input",
                    key_prefix="docs/",
                )
            ],
            target=S3Target(
                endpoint="s3.example.org",
                access_key="a",
                secret_key="b",
                bucket="output",
                key_prefix="converted/",
            ),
        )
        result = job.result(timeout=1.0)

    assert result is s3_result


def test_submit_batch_forwards_request_headers() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["header_tenant"] = request.headers.get("X-Tenant-Id")
        captured["header_api"] = request.headers.get("X-Api-Key")
        return httpx.Response(
            200,
            json=_status_response("task-batch-headers", "pending").model_dump(
                mode="json"
            ),
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL, api_key="base-key") as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            headers={"X-Api-Key": "base-key"},
            timeout=client._http_client.timeout,
        )
        job = client.submit_batch(
            sources=[AnyHttpSourceRequest(url="https://example.org/sample.pdf")],
            target=PresignedUrlTarget(),
            headers={"X-Tenant-Id": "tenant-batch"},
        )

    assert job.task_id == "task-batch-headers"
    assert captured["path"] == "/v1/convert/source/batch"
    assert captured["header_tenant"] == "tenant-batch"
    assert captured["header_api"] == "base-key"


def test_fetch_convert_result_payload_wraps_schema_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/result/task-bad-result"
        return httpx.Response(200, json={"unexpected": "shape"})

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            timeout=client._http_client.timeout,
        )
        with pytest.raises(
            ResponseSchemaMismatchError,
        ) as exc_info:
            client._fetch_convert_result_payload("task-bad-result", None)

    assert exc_info.value.status_code == 200
    assert exc_info.value.detail is not None
    assert exc_info.value.message == (
        "Response schema mismatch — client and server versions may differ."
    )


@pytest.mark.anyio
async def test_fetch_presigned_result_async_wraps_schema_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/result/task-bad-presigned"
        return httpx.Response(200, json={"unexpected": "shape"})

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=client._http_client.timeout,
        ) as async_client:
            with pytest.raises(
                ResponseSchemaMismatchError,
            ) as exc_info:
                await client._fetch_presigned_result_async(
                    "task-bad-presigned",
                    None,
                    async_client,
                )

    assert exc_info.value.status_code == 200
    assert exc_info.value.detail is not None
    assert exc_info.value.message == (
        "Response schema mismatch — client and server versions may differ."
    )


def test_submit_and_retrieve_many_is_deprecated_alias(tmp_path: Path) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client.submit_and_retrieve_each = MethodType(
            lambda self, items, max_in_flight=DEFAULT_MAX_CONCURRENCY, ordered=False, target=None: (
                iter(())
            ),
            client,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            list(client.submit_and_retrieve_many([ConversionItem(source=source)]))

    assert any("deprecated" in str(item.message).lower() for item in captured)


def test_request_with_retry_allows_request_headers_to_override_defaults() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("X-Api-Key")
        seen["tenant_id"] = request.headers.get("X-Tenant-Id")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL, api_key="base-key") as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport,
            headers={"X-Api-Key": "base-key"},
            timeout=client._http_client.timeout,
        )
        response = client._request_with_retry(
            method="GET",
            path="/health",
            headers={"X-Api-Key": "override-key", "X-Tenant-Id": "tenant-c"},
            retries=0,
        )

    assert response.status_code == 200
    assert seen == {"api_key": "override-key", "tenant_id": "tenant-c"}


# --- Retry policy tests ---


def test_503_with_retry_after_header_retries_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                503, headers={"Retry-After": "2"}, json={"detail": "backpressure"}
            )
        return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        response = client._request_with_retry(
            method="POST", path="/v1/convert/source/async", retries=1
        )

    assert response.status_code == 200
    assert sleep_calls == [2.0]


def test_429_with_retry_after_header_retries_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429, headers={"Retry-After": "4"}, json={"detail": "rate limited"}
            )
        return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        response = client._request_with_retry(
            method="POST", path="/v1/convert/source/async", retries=1
        )

    assert response.status_code == 200
    assert sleep_calls == [4.0]


def test_503_without_retry_after_header_does_not_retry() -> None:
    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, json={"detail": "backpressure"})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(ServiceUnavailableError, match="Task submission failed"):
            client._submit_convert_task(
                source="https://example.com/test.pdf",
                options=ConvertDocumentsRequestOptions(),
                target=InBodyTarget(),
            )

    assert call_count == 1


def test_429_without_retry_after_header_does_not_retry() -> None:
    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"detail": "rate limited"})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(ServiceError, match="Task submission failed"):
            client._submit_convert_task(
                source="https://example.com/test.pdf",
                options=ConvertDocumentsRequestOptions(),
                target=InBodyTarget(),
            )

    assert call_count == 1


def test_402_usage_limit_exceeded_raises_explicit_exception() -> None:
    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            402,
            json={
                "error": "usage_limit_exceeded",
                "message": "Your page processing limit has been exceeded. Please upgrade your plan.",
                "details": {"currentUsage": 101183, "limit": 100000},
            },
        )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(UsageLimitExceededError) as exc_info:
            client._submit_convert_task(
                source="https://example.com/test.pdf",
                options=ConvertDocumentsRequestOptions(),
                target=InBodyTarget(),
            )

    assert call_count == 1
    assert exc_info.value.status_code == 402
    assert (
        exc_info.value.detail
        == "Your page processing limit has been exceeded. Please upgrade your plan."
    )
    assert exc_info.value.current_usage == 101183
    assert exc_info.value.limit == 100000


def test_402_usage_limit_exceeded_with_invalid_payload_omits_detail() -> None:
    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            402,
            json={
                "error": "usage_limit_exceeded",
                "message": "Your page processing limit has been exceeded. Please upgrade your plan.",
                "details": "not-an-object",
            },
        )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(UsageLimitExceededError) as exc_info:
            client._submit_convert_task(
                source="https://example.com/test.pdf",
                options=ConvertDocumentsRequestOptions(),
                target=InBodyTarget(),
            )

    assert call_count == 1
    assert exc_info.value.status_code == 402
    assert exc_info.value.detail is None
    assert exc_info.value.current_usage is None
    assert exc_info.value.limit is None


def test_500_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            return httpx.Response(500, json={"detail": "server error"})
        return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        response = client._request_with_retry(
            method="POST", path="/v1/convert/source/async", retries=3
        )

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0, 4.0]


def test_502_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            return httpx.Response(502, json={"detail": "bad gateway"})
        return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        response = client._request_with_retry(
            method="POST", path="/v1/convert/source/async", retries=3
        )

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0, 4.0]


def test_503_after_all_retries_raises_service_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = lambda **kw: httpx.Response(
            503, headers={"Retry-After": "1"}
        )  # type: ignore[method-assign]
        with pytest.raises(ServiceUnavailableError):
            client._request_with_retry(
                method="POST", path="/v1/convert/source/async", retries=2
            )


def test_get_transport_error_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ConnectTimeout("connect timed out")
        return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        response = client._request_with_retry(method="GET", path="/v1/result/task-123")

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0]


def test_post_transport_error_does_not_retry() -> None:
    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("connect timed out")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(
            ServiceUnavailableError,
            match="Service transport request failed",
        ):
            client._request_with_retry(
                method="POST", path="/v1/convert/source/async", retries=3
            )

    assert call_count == 1


def test_get_transport_error_after_all_retries_raises_service_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def fake_request(**kw: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("connect timed out")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.request = fake_request  # type: ignore[method-assign]
        with pytest.raises(
            ServiceUnavailableError,
            match="Service transport request failed after retries",
        ):
            client._request_with_retry(
                method="GET", path="/v1/result/task-123", retries=2
            )

    assert call_count == 3
    assert sleep_calls == [1.0, 2.0]


@pytest.mark.anyio
async def test_503_with_retry_after_header_retries_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    503, headers={"Retry-After": "3"}, json={"detail": "backpressure"}
                )
            return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        response = await client._request_with_retry_async(
            async_client=FakeAsyncClient(),  # type: ignore[arg-type]
            method="POST",
            path="/v1/convert/source/async",
            retries=1,
        )

    assert response.status_code == 200
    assert sleep_calls == [3.0]


@pytest.mark.anyio
async def test_429_with_retry_after_header_retries_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429, headers={"Retry-After": "5"}, json={"detail": "rate limited"}
                )
            return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        response = await client._request_with_retry_async(
            async_client=FakeAsyncClient(),  # type: ignore[arg-type]
            method="POST",
            path="/v1/convert/source/async",
            retries=1,
        )

    assert response.status_code == 200
    assert sleep_calls == [5.0]


@pytest.mark.anyio
async def test_502_retries_with_exponential_backoff_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                return httpx.Response(502, json={"detail": "bad gateway"})
            return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        response = await client._request_with_retry_async(
            async_client=FakeAsyncClient(),  # type: ignore[arg-type]
            method="POST",
            path="/v1/convert/source/async",
            retries=3,
        )

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0, 4.0]


@pytest.mark.anyio
async def test_get_transport_error_retries_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectTimeout("connect timed out")
            return httpx.Response(200, json={})

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        response = await client._request_with_retry_async(
            async_client=FakeAsyncClient(),  # type: ignore[arg-type]
            method="GET",
            path="/v1/result/task-123",
        )

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0]


@pytest.mark.anyio
async def test_post_transport_error_does_not_retry_async() -> None:
    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectTimeout("connect timed out")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        with pytest.raises(
            ServiceUnavailableError,
            match="Service transport request failed",
        ):
            await client._request_with_retry_async(
                async_client=FakeAsyncClient(),  # type: ignore[arg-type]
                method="POST",
                path="/v1/convert/source/async",
                retries=3,
            )

    assert call_count == 1


# --- Path-prefix URL tests ---


def test_base_url_accepts_path_prefix() -> None:
    with DoclingServiceClient(url="http://proxy.example.com/docling") as client:
        assert client._base_url == "http://proxy.example.com/docling"

    with DoclingServiceClient(url="http://proxy.example.com/docling/") as client:
        assert client._base_url == "http://proxy.example.com/docling"

    with DoclingServiceClient(url="http://proxy.example.com/a/b/c") as client:
        assert client._base_url == "http://proxy.example.com/a/b/c"


def test_api_url_includes_base_path_prefix() -> None:
    with DoclingServiceClient(url="http://proxy.example.com/docling") as client:
        assert (
            client._url("/v1/convert/source/async")
            == "http://proxy.example.com/docling/v1/convert/source/async"
        )


def test_ws_status_url_includes_base_path_prefix() -> None:
    with DoclingServiceClient(url="https://proxy.example.com/docling") as client:
        assert (
            client._build_ws_status_url("task-123")
            == "wss://proxy.example.com/docling/v1/status/ws/task-123"
        )

    with DoclingServiceClient(url="http://proxy.example.com/a/b") as client:
        assert (
            client._build_ws_status_url("task-99")
            == "ws://proxy.example.com/a/b/v1/status/ws/task-99"
        )


def test_ws_status_url_includes_api_key_query_param_with_base_path_prefix() -> None:
    with DoclingServiceClient(
        url="https://proxy.example.com/docling", api_key="secret-key"
    ) as client:
        assert (
            client._build_ws_status_url("task-123")
            == "wss://proxy.example.com/docling/v1/status/ws/task-123"
            "?api_key=secret-key"
        )


def test_page_range_json_serialization_is_warning_free() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = ConvertDocumentsRequestOptions(page_range=(3, 7)).model_dump(
            mode="json",
            exclude_defaults=True,
            exclude_none=True,
        )

    assert payload["page_range"] == [3, 7]
    assert all(
        "PydanticSerializationUnexpectedValue" not in str(warning.message)
        for warning in caught
    )


# ---------------------------------------------------------------------------
# Presigned materialization for high-level convert()/convert_all()
# ---------------------------------------------------------------------------


def _sample_document_with_images() -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    page_img = PILImage.new("RGB", (40, 30), (10, 20, 30))
    doc.add_page(
        page_no=1,
        size=Size(width=40, height=30),
        image=ImageRef.from_pil(page_img, dpi=72),
    )
    pic_img = PILImage.new("RGB", (12, 8), (200, 100, 50))
    prov = ProvenanceItem(
        page_no=1, bbox=BoundingBox(l=0, t=0, r=12, b=8), charspan=(0, 0)
    )
    doc.add_picture(image=ImageRef.from_pil(pic_img, dpi=72), prov=prov)
    return doc


def _referenced_bundle_bytes() -> bytes:
    """Build a resource-bundle ZIP the way the server does: doc JSON at the root
    plus the externalized images under artifacts/ (REFERENCED mode)."""
    doc = _sample_document_with_images()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        doc.save_as_json(
            base / "sample.json",
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("artifacts"),
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle_zip:
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    bundle_zip.write(path, arcname=path.relative_to(base).as_posix())
        return buf.getvalue()


def _embedded_json_bytes() -> bytes:
    return _sample_document_with_images().model_dump_json().encode("utf-8")


def _presigned_response(
    artifacts: list[dict], *, filename: str = "sample.pdf", status: str = "success"
) -> PresignedUrlConvertResponse:
    return PresignedUrlConvertResponse(
        num_converted=1,
        num_succeeded=1 if status == "success" else 0,
        num_partially_succeeded=0,
        num_failed=0,
        processing_time=0.1,
        documents=[
            {
                "source_index": 0,
                "source_uri": filename,
                "filename": filename,
                "status": status,
                "artifacts": artifacts,
            }
        ],
    )


def test_convert_presigned_bundle_materializes_embedded_images(tmp_path: Path) -> None:
    bundle_bytes = _referenced_bundle_bytes()
    seen_targets: list[type[object]] = []
    downloaded: list[str] = []

    def fake_submit(self, source, options, target, request_headers=None):
        seen_targets.append(type(target))
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return _presigned_response(
            [
                {
                    "artifact_type": "json",
                    "mime_type": "application/json",
                    "uri": "https://dl.example.org/sample.json",
                },
                {
                    "artifact_type": "resource_bundle",
                    "mime_type": "application/zip",
                    "uri": "https://dl.example.org/sample_bundle.zip",
                },
            ]
        )

    def fake_download(self, uri):
        downloaded.append(uri)
        return bundle_bytes

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)
        client._download_artifact_bytes = MethodType(fake_download, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        result = client.convert(source)

    # Presigned is attempted first; the bundle (not the bare JSON) is downloaded.
    assert seen_targets == [PresignedUrlTarget]
    assert downloaded == ["https://dl.example.org/sample_bundle.zip"]
    assert result.status == ConversionStatus.SUCCESS
    assert result.input.file.name == "sample.pdf"

    pictures = [
        item
        for item, _ in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]
    assert pictures and pictures[0].image is not None
    assert str(pictures[0].image.uri).startswith("data:image")
    assert pictures[0].image.pil_image.size == (12, 8)

    page_image = result.document.pages[1].image
    assert page_image is not None
    assert str(page_image.uri).startswith("data:image")
    assert page_image.pil_image.size == (40, 30)


def test_convert_presigned_json_only_is_self_contained(tmp_path: Path) -> None:
    json_bytes = _embedded_json_bytes()
    downloaded: list[str] = []

    def fake_submit(self, source, options, target, request_headers=None):
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return _presigned_response(
            [
                {
                    "artifact_type": "json",
                    "mime_type": "application/json",
                    "uri": "https://dl.example.org/sample.json",
                }
            ]
        )

    def fake_download(self, uri):
        downloaded.append(uri)
        return json_bytes

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)
        client._download_artifact_bytes = MethodType(fake_download, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        result = client.convert(source)

    assert downloaded == ["https://dl.example.org/sample.json"]
    assert result.status == ConversionStatus.SUCCESS
    pictures = [
        item
        for item, _ in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]
    assert pictures and str(pictures[0].image.uri).startswith("data:image")


def test_convert_falls_back_to_inbody_when_presigned_is_rejected(
    tmp_path: Path,
) -> None:
    seen_targets: list[type[object]] = []

    def fake_submit(self, source, options, target, request_headers=None):
        seen_targets.append(type(target))
        if isinstance(target, PresignedUrlTarget):
            raise ServiceError(
                "Task submission failed.",
                status_code=422,
                detail=(
                    "Presigned URL target requires artifact storage to be configured "
                    "and enabled on the server."
                ),
            )
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_payload(self, task_id, last_status):
        return _convert_payload("sample.pdf")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_convert_result_payload = MethodType(fake_fetch_payload, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        result = client.convert(source)

    assert seen_targets == [PresignedUrlTarget, InBodyTarget]
    assert result.status == ConversionStatus.SUCCESS


def test_convert_download_failure_degrades_gracefully(tmp_path: Path) -> None:
    def fake_submit(self, source, options, target, request_headers=None):
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return _presigned_response(
            [
                {
                    "artifact_type": "resource_bundle",
                    "mime_type": "application/zip",
                    "uri": "https://dl.example.org/sample_bundle.zip",
                }
            ]
        )

    def fake_download(self, uri):
        raise ArtifactDownloadError("Artifact download failed: boom")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)
        client._download_artifact_bytes = MethodType(fake_download, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")

        result = client.convert(source, raises_on_error=False)
        assert result.status == ConversionStatus.FAILURE
        assert result.errors
        assert "boom" in result.errors[0].error_message

        with pytest.raises(ConversionError):
            client.convert(source)


def test_convert_presigned_failed_item_preserves_server_errors(tmp_path: Path) -> None:
    # A server-side FAILURE arrives with the real status/errors and no document
    # artifacts; materialization must surface those, not a generic "no artifact".
    server_error = ErrorItem(
        component_type=DoclingComponentType.MODEL,
        module_name="docling.pipeline",
        error_message="OCR engine crashed on page 3",
    )

    def fake_submit(self, source, options, target, request_headers=None):
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return PresignedUrlConvertResponse(
            num_converted=1,
            num_succeeded=0,
            num_partially_succeeded=0,
            num_failed=1,
            processing_time=0.1,
            documents=[
                {
                    "source_index": 0,
                    "source_uri": "sample.pdf",
                    "filename": "sample.pdf",
                    "status": "failure",
                    "errors": [server_error.model_dump()],
                    "artifacts": [],
                }
            ],
        )

    def fake_download(self, uri):
        raise AssertionError("must not download artifacts for a failed item")

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)
        client._download_artifact_bytes = MethodType(fake_download, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        result = client.convert(source, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert [e.error_message for e in result.errors] == ["OCR engine crashed on page 3"]


def test_convert_presigned_bundle_rejects_image_outside_bundle(tmp_path: Path) -> None:
    # A malicious bundle whose JSON references an image outside the extract dir
    # must be rejected rather than reading arbitrary local files.
    doc = _sample_document_with_images()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        doc.save_as_json(
            base / "sample.json",
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("artifacts"),
        )
        json_path = base / "sample.json"
        json_path.write_text(
            json_path.read_text().replace("artifacts/", "../../../etc/")
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle_zip:
            bundle_zip.write(json_path, arcname="sample.json")
        bundle_bytes = buf.getvalue()

    def fake_submit(self, source, options, target, request_headers=None):
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return _presigned_response(
            [
                {
                    "artifact_type": "resource_bundle",
                    "mime_type": "application/zip",
                    "uri": "https://dl.example.org/sample_bundle.zip",
                }
            ]
        )

    def fake_download(self, uri):
        return bundle_bytes

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)
        client._download_artifact_bytes = MethodType(fake_download, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        result = client.convert(source, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert "outside the bundle" in result.errors[0].error_message


def test_convert_all_materializes_presigned_results_in_order(tmp_path: Path) -> None:
    bundle_bytes = _referenced_bundle_bytes()

    async def fake_many(self, item_list, max_in_flight, ordered, target=None):
        assert target is None
        for item in list(item_list):
            name = Path(item.source).name
            yield (
                item,
                _presigned_response(
                    [
                        {
                            "artifact_type": "resource_bundle",
                            "mime_type": "application/zip",
                            "uri": f"https://dl.example.org/{name}.zip",
                        }
                    ],
                    filename=name,
                ),
            )

    async def fake_download_async(self, uri):
        return bundle_bytes

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_and_retrieve_many_async = MethodType(fake_many, client)
        client._download_artifact_bytes_async = MethodType(fake_download_async, client)

        paths = []
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            p = tmp_path / name
            p.write_bytes(b"%PDF-1.4\n")
            paths.append(p)

        results = list(client.convert_all(paths))

    assert [r.input.file.name for r in results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(r.status == ConversionStatus.SUCCESS for r in results)
    for result in results:
        pictures = [
            item
            for item, _ in result.document.iterate_items()
            if isinstance(item, PictureItem)
        ]
        assert pictures and str(pictures[0].image.uri).startswith("data:image")


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://8.8.8.8/artifact.zip", True),
        ("https://127.0.0.1/artifact.zip", False),
        ("http://10.1.2.3/artifact.zip", False),
        ("https://169.254.169.254/artifact.zip", False),
        ("http://[::1]/artifact.zip", False),
        ("ftp://8.8.8.8/artifact.zip", False),
        ("not-a-url", False),
    ],
)
def test_is_safe_artifact_url(url: str, expected: bool) -> None:
    assert client_module._is_safe_artifact_url(url) is expected


def test_convert_rejects_private_artifact_url_by_default(tmp_path: Path) -> None:
    def fake_submit(self, source, options, target, request_headers=None):
        return _status_response("task-x", "pending")

    def fake_wait(self, task_id, timeout):
        return _status_response(task_id, "success")

    def fake_fetch_presigned(self, task_id, last_status):
        return _presigned_response(
            [
                {
                    "artifact_type": "resource_bundle",
                    "mime_type": "application/zip",
                    "uri": "http://127.0.0.1:9000/sample_bundle.zip",
                }
            ]
        )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._submit_convert_task = MethodType(fake_submit, client)
        client._wait_for_terminal_status = MethodType(fake_wait, client)
        client._fetch_presigned_result = MethodType(fake_fetch_presigned, client)

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"%PDF-1.4\n")

        # The SSRF guard fires before any network call, degrading to FAILURE.
        result = client.convert(source, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert result.errors
    assert "non-public URL" in result.errors[0].error_message
