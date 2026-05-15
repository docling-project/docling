import asyncio
import queue
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import SimpleNamespace

import httpx
import pytest
from docling_core.types.doc import DoclingDocument

import docling.service_client._async_client as _async_client_module
import docling.service_client.client as client_module
import docling.service_client.watchers as watchers_module
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.responses import (
    MessageKind,
    TaskStatusResponse,
    WebsocketMessage,
)
from docling.service_client import (
    DEFAULT_MAX_CONCURRENCY,
    MAX_CONCURRENCY_LIMIT,
    ConversionItem,
    DoclingServiceClient,
)
from docling.service_client.exceptions import (
    ConversionError,
    ResultExpiredError,
    ServiceError,
    ServiceUnavailableError,
    UsageLimitExceededError,
)
from docling.service_client.job import ConversionJob, _JobHandlers
from docling.service_client.watchers import AsyncPollingWatcher, PollingWatcher

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


def _make_sam_ctx(submit_fn, wait_fn, fetch_fn):
    """Build a fake async service client context for submit_and_retrieve_many tests.

    Returns a context manager whose submit_and_retrieve_many delegates to
    _run_bounded with the supplied fake submit/wait/fetch callables, exercising
    the same ordering and concurrency guarantees as the real implementation.
    """
    from docling.service_client._scheduler import _run_bounded

    class _FakeAsyncCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def submit_and_retrieve_many(self, items, max_in_flight=8, ordered=False):
            async def process_one(_idx, item, _client):
                status = await submit_fn(item.source, item.source_headers, item.headers)
                terminal = await wait_fn(status.task_id, None)
                return await fetch_fn(status.task_id, terminal)

            buffered: dict = {}
            next_ordered = 0
            async for idx, item, outcome in _run_bounded(
                items, process_one, object(), max_in_flight
            ):
                normalized: object = (
                    outcome
                    if isinstance(outcome, Exception)
                    else RuntimeError(str(outcome))
                    if isinstance(outcome, BaseException)
                    else outcome
                )
                if ordered:
                    buffered[idx] = (item, normalized)
                    while next_ordered in buffered:
                        yield buffered.pop(next_ordered)
                        next_ordered += 1
                    continue
                yield item, normalized

    return _FakeAsyncCtx()


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
    assert client_module._guess_input_format("doc.asc") == InputFormat.ASCIIDOC
    assert client_module._guess_input_format("subtitles.vtt") == InputFormat.VTT
    assert client_module._guess_input_format("archive.tar.gz") == InputFormat.METS_GBS


def test_result_404_after_failed_status_raises_conversion_error() -> None:
    response = httpx.Response(
        404,
        json={"detail": "Task result not found. Please wait for a completion status."},
    )
    last_status = _status_response("task-1", "failure")
    last_status.error_message = "conversion failed upstream"

    with pytest.raises(ConversionError, match="conversion failed upstream"):
        client_module._raise_for_result_404(
            task_id="task-1",
            response=response,
            last_status=last_status,
        )


def test_result_404_after_success_status_raises_result_expired() -> None:
    response = httpx.Response(
        404,
        json={"detail": "Task result not found. Please wait for a completion status."},
    )
    last_status = _status_response("task-1", "success")

    with pytest.raises(ResultExpiredError, match="task-1"):
        client_module._raise_for_result_404(
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
    sleep_calls: list[float] = []
    clock_now = [0.0]

    monkeypatch.setattr(watchers_module.time, "monotonic", lambda: clock_now[0])

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock_now[0] += seconds

    monkeypatch.setattr(watchers_module.asyncio, "sleep", fake_sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    async def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    watcher = AsyncPollingWatcher(
        poll_status=fake_poll,
        poll_server_wait=0.5,
        poll_client_interval=None,
        default_timeout=5.0,
    )
    updates = [u async for u in watcher.iter_updates(task_id="task-1")]

    assert [u.task_status for u in updates] == ["pending", "success"]
    assert wait_values == [0.5, 0.5]
    assert sleep_calls == [pytest.approx(0.5)]


@pytest.mark.anyio
async def test_async_wait_for_terminal_supports_explicit_client_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    clock_now = [0.0]

    monkeypatch.setattr(watchers_module.time, "monotonic", lambda: clock_now[0])

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock_now[0] += seconds

    monkeypatch.setattr(watchers_module.asyncio, "sleep", fake_sleep)

    statuses = iter(["pending", "success"])
    wait_values: list[float] = []

    async def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        wait_values.append(wait)
        return _status_response(task_id=task_id, status=next(statuses))

    watcher = AsyncPollingWatcher(
        poll_status=fake_poll,
        poll_server_wait=5.0,
        poll_client_interval=1.0,
        default_timeout=10.0,
    )
    updates = [u async for u in watcher.iter_updates(task_id="task-1")]

    assert [u.task_status for u in updates] == ["pending", "success"]
    assert wait_values == [5.0, 5.0]
    assert sleep_calls == [pytest.approx(1.0)]


def test_convert_all_uses_async_pipeline_and_preserves_order(tmp_path) -> None:
    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        calls: list[dict[str, object]] = []

        class _FakeAsyncCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def submit_and_retrieve_many(self, item_list, max_in_flight, ordered):
                items = list(item_list)
                calls.append(
                    {
                        "count": len(items),
                        "max_in_flight": max_in_flight,
                        "ordered": ordered,
                        "source_headers": [item.source_headers for item in items],
                        "request_headers": [item.headers for item in items],
                    }
                )
                for item in items:
                    yield item, _convert_payload(Path(item.source).name)

        client._build_async_service_client = lambda: _FakeAsyncCtx()  # type: ignore[method-assign]

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
            "source_headers": [
                {"Authorization": "Bearer source-token"},
                {"Authorization": "Bearer source-token"},
                {"Authorization": "Bearer source-token"},
            ],
            "request_headers": [None, None, None],
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

        class _FakeAsyncCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def submit_and_retrieve_many(self, item_list, max_in_flight, ordered):
                items = list(item_list)
                assert max_in_flight == 2
                assert ordered is True

                await asyncio.to_thread(release_first.wait)
                yield items[0], _convert_payload(Path(items[0].source).name)
                yield items[1], _convert_payload(Path(items[1].source).name)

                await asyncio.to_thread(release_third.wait)
                yield items[2], _convert_payload(Path(items[2].source).name)

        client._build_async_service_client = lambda: _FakeAsyncCtx()  # type: ignore[method-assign]

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

        class _FakeAsyncCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def submit_and_retrieve_many(self, item_list, max_in_flight, ordered):
                items = list(item_list)
                assert max_in_flight == DEFAULT_MAX_CONCURRENCY
                assert ordered is True

                for item in items:
                    submitted_names.append(Path(item.source).name)
                    yield item, _convert_payload(Path(item.source).name)

        client._build_async_service_client = lambda: _FakeAsyncCtx()  # type: ignore[method-assign]

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
    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(task_id, timeout):
            if task_id == "task-a.pdf":
                await asyncio.sleep(0.01)
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")

        completion_order = [
            Path(item.source).name
            for item, _ in client.submit_and_retrieve_many(
                [ConversionItem(source=p1), ConversionItem(source=p2)],
                max_in_flight=2,
            )
        ]
        ordered_names = [
            Path(item.source).name
            for item, _ in client.submit_and_retrieve_many(
                [ConversionItem(source=p1), ConversionItem(source=p2)],
                max_in_flight=2,
                ordered=True,
            )
        ]

    assert completion_order == ["b.pdf", "a.pdf"]
    assert ordered_names == ["a.pdf", "b.pdf"]


def test_submit_and_retrieve_many_forwards_per_item_request_headers(
    tmp_path: Path,
) -> None:
    seen_headers: list[dict[str, str] | None] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            seen_headers.append(request_headers)
            return _status_response(f"task-{source.name}", "pending")

        async def fake_wait(task_id, timeout):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        p1 = tmp_path / "a.pdf"
        p2 = tmp_path / "b.pdf"
        p1.write_bytes(b"%PDF-1.4\n")
        p2.write_bytes(b"%PDF-1.4\n")

        list(
            client.submit_and_retrieve_many(
                [
                    ConversionItem(source=p1, headers={"X-Tenant-Id": "tenant-a"}),
                    ConversionItem(source=p2, headers={"X-Tenant-Id": "tenant-b"}),
                ],
                max_in_flight=2,
            )
        )

    assert seen_headers == [
        {"X-Tenant-Id": "tenant-a"},
        {"X-Tenant-Id": "tenant-b"},
    ]


def test_submit_and_retrieve_many_forwards_per_item_source_headers() -> None:
    seen_source_headers: list[dict[str, str] | None] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            seen_source_headers.append(source_headers)
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(task_id, timeout):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        list(
            client.submit_and_retrieve_many(
                [
                    ConversionItem(
                        source="https://example.org/a.pdf",
                        source_headers={"Authorization": "Bearer a"},
                    ),
                    ConversionItem(
                        source="https://example.org/b.pdf",
                        source_headers={"Authorization": "Bearer b"},
                    ),
                ],
                max_in_flight=2,
            )
        )

    assert seen_source_headers == [
        {"Authorization": "Bearer a"},
        {"Authorization": "Bearer b"},
    ]


def test_submit_and_retrieve_many_isolates_failures_per_item(tmp_path: Path) -> None:
    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            if Path(source).name == "bad.pdf":
                raise ValueError("submit failed")
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(task_id, timeout):
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        good = tmp_path / "good.pdf"
        bad = tmp_path / "bad.pdf"
        good.write_bytes(b"%PDF-1.4\n")
        bad.write_bytes(b"%PDF-1.4\n")

        outcomes = sorted(
            client.submit_and_retrieve_many(
                [ConversionItem(source=bad), ConversionItem(source=good)],
                max_in_flight=2,
            ),
            key=lambda entry: Path(entry[0].source).name,
        )

    assert isinstance(outcomes[0][1], Exception)
    assert str(outcomes[0][1]) == "submit failed"
    assert getattr(outcomes[1][1], "status") == ConversionStatus.SUCCESS


def test_submit_and_retrieve_many_respects_max_in_flight(tmp_path: Path) -> None:
    state = {"active": 0, "max_seen": 0, "submitted": 0}
    release = asyncio.Event()

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            state["active"] += 1
            state["submitted"] += 1
            state["max_seen"] = max(state["max_seen"], state["active"])
            if state["submitted"] >= 2:
                release.set()
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(task_id, timeout):
            if task_id in {"task-a.pdf", "task-b.pdf"}:
                await release.wait()
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            state["active"] -= 1
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        paths = []
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            path = tmp_path / name
            path.write_bytes(b"%PDF-1.4\n")
            paths.append(path)

        list(
            client.submit_and_retrieve_many(
                [ConversionItem(source=path) for path in paths],
                max_in_flight=2,
            )
        )

    assert state["max_seen"] == 2


def test_submit_and_retrieve_many_consumes_iterable_incrementally(
    tmp_path: Path,
) -> None:
    generated: list[str] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        async def fake_submit(source, source_headers, request_headers=None):
            return _status_response(f"task-{Path(source).name}", "pending")

        async def fake_wait(task_id, timeout):
            await asyncio.sleep(
                0
            )  # yield so the event loop can process completed results
            return _status_response(task_id, "success")

        async def fake_fetch_payload(task_id, last_status):
            return _convert_payload(task_id.removeprefix("task-"))

        client._build_async_service_client = lambda: _make_sam_ctx(  # type: ignore[method-assign]
            fake_submit, fake_wait, fake_fetch_payload
        )

        def item_iter():
            for idx in range(50):
                path = tmp_path / f"{idx}.pdf"
                path.write_bytes(b"%PDF-1.4\n")
                generated.append(path.name)
                yield ConversionItem(source=path)

        iterator = client.submit_and_retrieve_many(
            item_iter(),
            max_in_flight=1,
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
            client.submit_and_retrieve_many(
                [ConversionItem(source=source)],
                max_in_flight=value,
            )


@pytest.mark.anyio
async def test_submit_and_retrieve_many_ordered_mode_yields_before_batch_completion(
    tmp_path: Path,
) -> None:
    from docling.service_client._async_client import AsyncDoclingServiceClient

    release_first = asyncio.Event()
    release_third = asyncio.Event()

    async def fake_submit(source, source_headers, request_headers=None):
        return _status_response(f"task-{Path(source).name}", "pending")

    async def fake_wait(task_id, timeout):
        if task_id == "task-a.pdf":
            await release_first.wait()
        if task_id == "task-c.pdf":
            await release_third.wait()
        return _status_response(task_id, "success")

    async def fake_fetch_payload(task_id, last_status):
        return _convert_payload(task_id.removeprefix("task-"))

    items: list[ConversionItem] = []
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.4\n")
        items.append(ConversionItem(source=path))

    async with _make_sam_ctx(
        fake_submit, fake_wait, fake_fetch_payload
    ) as async_client:
        async_iterator = async_client.submit_and_retrieve_many(
            items,
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
                source_headers=None,
                options=ConvertDocumentsRequestOptions(),
                raw_result=False,
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
                source_headers=None,
                options=ConvertDocumentsRequestOptions(),
                raw_result=False,
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
                source_headers=None,
                options=ConvertDocumentsRequestOptions(),
                raw_result=False,
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
                source_headers=None,
                options=ConvertDocumentsRequestOptions(),
                raw_result=False,
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
    from docling.service_client._async_client import AsyncDoclingServiceClient

    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(_async_client_module.asyncio, "sleep", fake_sleep)

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

        async def aclose(self) -> None:
            pass

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as async_client:
        async_client._async_client = FakeAsyncClient()  # type: ignore[assignment]
        response = await async_client._request_with_retry(
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
    from docling.service_client._async_client import AsyncDoclingServiceClient

    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(_async_client_module.asyncio, "sleep", fake_sleep)

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

        async def aclose(self) -> None:
            pass

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as async_client:
        async_client._async_client = FakeAsyncClient()  # type: ignore[assignment]
        response = await async_client._request_with_retry(
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
    from docling.service_client._async_client import AsyncDoclingServiceClient

    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(_async_client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                return httpx.Response(502, json={"detail": "bad gateway"})
            return httpx.Response(200, json={})

        async def aclose(self) -> None:
            pass

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as async_client:
        async_client._async_client = FakeAsyncClient()  # type: ignore[assignment]
        response = await async_client._request_with_retry(
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
    from docling.service_client._async_client import AsyncDoclingServiceClient

    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(_async_client_module.asyncio, "sleep", fake_sleep)

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectTimeout("connect timed out")
            return httpx.Response(200, json={})

        async def aclose(self) -> None:
            pass

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as async_client:
        async_client._async_client = FakeAsyncClient()  # type: ignore[assignment]
        response = await async_client._request_with_retry(
            method="GET",
            path="/v1/result/task-123",
        )

    assert response.status_code == 200
    assert sleep_calls == [1.0, 2.0]


@pytest.mark.anyio
async def test_post_transport_error_does_not_retry_async() -> None:
    from docling.service_client._async_client import AsyncDoclingServiceClient

    call_count = 0

    class FakeAsyncClient:
        async def request(self, **kw: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectTimeout("connect timed out")

        async def aclose(self) -> None:
            pass

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as async_client:
        async_client._async_client = FakeAsyncClient()  # type: ignore[assignment]
        with pytest.raises(
            ServiceUnavailableError,
            match="Service transport request failed",
        ):
            await async_client._request_with_retry(
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
            exclude_none=True,
        )

    assert payload["page_range"] == [3, 7]
    assert all(
        "PydanticSerializationUnexpectedValue" not in str(warning.message)
        for warning in caught
    )


# ---------------------------------------------------------------------------
# AsyncDoclingServiceClient — unit tests
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402 (needed at module level)

from docling.service_client._async_client import AsyncDoclingServiceClient  # noqa: E402
from docling.service_client.client import StatusWatcherKind  # noqa: E402
from docling.service_client.job import (  # noqa: E402
    AsyncConversionJob,
    _AsyncJobHandlers,
)


@asynccontextmanager
async def _make_async_http_client(handler, **kwargs):
    """Open an AsyncDoclingServiceClient whose HTTP client is replaced by a fake."""
    async with AsyncDoclingServiceClient(url=TEST_BASE_URL, **kwargs) as client:

        class _FakeHttp:
            async def request(self, method, url, **kw):
                return handler(method, url, **kw)

            async def aclose(self) -> None:
                pass

        client._async_client = _FakeHttp()  # type: ignore[assignment]
        yield client


@asynccontextmanager
async def _make_async_sam_client(submit_fn, wait_fn, fetch_fn):
    """Open an AsyncDoclingServiceClient with patched submit/wait/fetch for SAM tests."""
    async with AsyncDoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher=StatusWatcherKind.POLLING,
    ) as client:

        class _FakeHttp:
            async def request(self, *a, **kw):
                raise AssertionError("unexpected HTTP call in SAM test")

            async def aclose(self) -> None:
                pass

        client._async_client = _FakeHttp()  # type: ignore[assignment]
        client._submit_convert_task = submit_fn  # type: ignore[method-assign]
        assert client._polling_watcher is not None
        client._polling_watcher.wait_for_terminal = wait_fn  # type: ignore[method-assign]
        client._fetch_convert_result_payload = fetch_fn  # type: ignore[method-assign]
        yield client


# --- constructor ---


@pytest.mark.parametrize("value", [0, -1, MAX_CONCURRENCY_LIMIT + 1])
def test_async_client_rejects_invalid_default_max_concurrency(value: int) -> None:
    with pytest.raises(
        ValueError,
        match=f"max_concurrency must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}.",
    ):
        AsyncDoclingServiceClient(url=TEST_BASE_URL, max_concurrency=value)


def test_async_client_constructor_matches_sync_defaults() -> None:
    """Verify AsyncDoclingServiceClient has the same default signature as DoclingServiceClient."""
    import inspect

    from docling.service_client.client import DEFAULT_MAX_CONCURRENCY

    sync_sig = inspect.signature(DoclingServiceClient.__init__)
    async_sig = inspect.signature(AsyncDoclingServiceClient.__init__)

    shared_params = [
        "url",
        "api_key",
        "options",
        "status_watcher",
        "ws_fallback_to_poll",
        "poll_server_wait",
        "poll_client_interval",
        "job_timeout",
        "max_concurrency",
        "http_retries",
        "http_connect_timeout",
        "http_read_timeout",
    ]
    for name in shared_params:
        assert name in sync_sig.parameters, f"sync missing {name}"
        assert name in async_sig.parameters, f"async missing {name}"
        s_default = sync_sig.parameters[name].default
        a_default = async_sig.parameters[name].default
        assert s_default == a_default, (
            f"default mismatch for {name!r}: sync={s_default!r} async={a_default!r}"
        )


# --- AsyncConversionJob lifecycle ---


@pytest.mark.anyio
async def test_async_conversion_job_poll_updates_state() -> None:
    seen_waits: list[float] = []

    async def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        seen_waits.append(wait)
        return _status_response(task_id=task_id, status="success")

    async def fake_fetch(task_id: str, last: TaskStatusResponse | None) -> str:
        return "result"

    job: AsyncConversionJob[str] = AsyncConversionJob(
        task_id="task-1",
        submitted_at=datetime.now(timezone.utc),
        handlers=_AsyncJobHandlers(
            poll=fake_poll,
            watch=lambda tid, t: (x async for x in []),
            wait=lambda tid, t: fake_poll(tid, 0.0),
            fetch_result=fake_fetch,
        ),
    )

    assert not job.done
    status = await job.poll(wait=3.7)
    assert status.task_status == "success"
    assert seen_waits == [3.7]
    assert job.done


@pytest.mark.anyio
async def test_async_conversion_job_watch_streams_updates() -> None:
    emitted = ["pending", "success"]
    idx = 0

    async def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        nonlocal idx
        s = emitted[idx]
        idx += 1
        return _status_response(task_id=task_id, status=s)

    async def gen(task_id: str, timeout: float | None):
        for s in emitted:
            yield _status_response(task_id=task_id, status=s)

    async def fake_fetch(task_id: str, last: TaskStatusResponse | None) -> str:
        return "done"

    job: AsyncConversionJob[str] = AsyncConversionJob(
        task_id="task-1",
        submitted_at=datetime.now(timezone.utc),
        handlers=_AsyncJobHandlers(
            poll=fake_poll,
            watch=gen,
            wait=lambda tid, t: fake_poll(tid, 0.0),
            fetch_result=fake_fetch,
        ),
    )

    updates = [u async for u in job.watch()]
    assert [u.task_status for u in updates] == ["pending", "success"]
    assert job.done


@pytest.mark.anyio
async def test_async_conversion_job_result_waits_and_fetches() -> None:
    waited: list[str] = []
    fetched: list[str] = []

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        waited.append(task_id)
        return _status_response(task_id=task_id, status="success")

    async def fake_fetch(task_id: str, last: TaskStatusResponse | None) -> str:
        fetched.append(task_id)
        return "the-result"

    job: AsyncConversionJob[str] = AsyncConversionJob(
        task_id="task-1",
        submitted_at=datetime.now(timezone.utc),
        handlers=_AsyncJobHandlers(
            poll=lambda tid, w: fake_wait(tid, None),
            watch=lambda tid, t: (x async for x in []),
            wait=fake_wait,
            fetch_result=fake_fetch,
        ),
    )

    result = await job.result(timeout=10.0)
    assert result == "the-result"
    assert waited == ["task-1"]
    assert fetched == ["task-1"]


@pytest.mark.anyio
async def test_async_conversion_job_result_skips_wait_when_already_done() -> None:
    wait_count = [0]

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        wait_count[0] += 1
        return _status_response(task_id=task_id, status="success")

    async def fake_fetch(task_id: str, last: TaskStatusResponse | None) -> str:
        return "done"

    initial = _status_response("task-1", "success")
    job: AsyncConversionJob[str] = AsyncConversionJob(
        task_id="task-1",
        submitted_at=datetime.now(timezone.utc),
        handlers=_AsyncJobHandlers(
            poll=lambda tid, w: fake_wait(tid, None),
            watch=lambda tid, t: (x async for x in []),
            wait=fake_wait,
            fetch_result=fake_fetch,
        ),
        initial_status=initial,
    )

    assert job.done
    await job.result()
    assert wait_count[0] == 0


# --- submit ---


@pytest.mark.anyio
async def test_async_submit_url_source_posts_correct_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        captured["method"] = method
        captured["url"] = url
        return httpx.Response(
            200, json=_status_response("task-1", "pending").model_dump(mode="json")
        )

    async with _make_async_http_client(handler) as client:
        job = await client.submit(source="https://example.org/sample.pdf")

    assert captured["method"] == "POST"
    assert "/v1/convert/source/async" in str(captured["url"])
    assert job.task_id == "task-1"


@pytest.mark.anyio
async def test_async_submit_file_source_posts_correct_endpoint(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        captured["method"] = method
        captured["url"] = url
        return httpx.Response(
            200, json=_status_response("task-2", "pending").model_dump(mode="json")
        )

    async with _make_async_http_client(handler) as client:
        job = await client.submit(source=sample)

    assert captured["method"] == "POST"
    assert "/v1/convert/file/async" in str(captured["url"])
    assert job.task_id == "task-2"


@pytest.mark.anyio
async def test_async_submit_forwards_request_headers() -> None:
    captured: dict[str, object] = {}

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        captured["headers"] = kw.get("headers")
        return httpx.Response(
            200, json=_status_response("task-1", "pending").model_dump(mode="json")
        )

    async with _make_async_http_client(handler, api_key="base-key") as client:
        job = await client.submit(
            source="https://example.org/sample.pdf",
            headers={"X-Tenant-Id": "tenant-a"},
        )

    assert captured["headers"] == {"X-Tenant-Id": "tenant-a"}
    assert job.task_id == "task-1"


@pytest.mark.anyio
async def test_async_submit_non_json_target_format_uses_zip_target(
    tmp_path: Path,
) -> None:
    from docling.datamodel.base_models import OutputFormat

    calls: list[str] = []

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        calls.append(url)
        data = kw.get("data") or {}
        assert data.get("target_type") == "zip", (
            f"expected zip, got {data.get('target_type')}"
        )
        return httpx.Response(
            200, json=_status_response("task-raw", "pending").model_dump(mode="json")
        )

    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    async with _make_async_http_client(handler) as client:
        job = await client.submit(
            source=sample,
            target_format=OutputFormat.MARKDOWN,
        )

    assert job.task_id == "task-raw"


# --- submit_chunk ---


@pytest.mark.anyio
async def test_async_submit_chunk_url_source_posts_correct_endpoint() -> None:
    from docling.service_client.client import ChunkerKind

    captured: dict[str, object] = {}

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        captured["url"] = url
        return httpx.Response(
            200, json=_status_response("chunk-1", "pending").model_dump(mode="json")
        )

    async with _make_async_http_client(handler) as client:
        job = await client.submit_chunk(
            source="https://example.org/doc.pdf",
            chunker=ChunkerKind.HYBRID,
        )

    assert "/v1/chunk/hybrid/source/async" in str(captured["url"])
    assert job.task_id == "chunk-1"


# --- health / version ---


@pytest.mark.anyio
async def test_async_health_returns_parsed_response() -> None:
    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with _make_async_http_client(handler) as client:
        result = await client.health()

    assert result.status == "ok"


@pytest.mark.anyio
async def test_async_version_returns_dict() -> None:
    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        return httpx.Response(200, json={"version": "1.2.3", "api_version": "v1"})

    async with _make_async_http_client(handler) as client:
        result = await client.version()

    assert result["version"] == "1.2.3"


# --- async WebSocket watcher fallback ---


@pytest.mark.anyio
async def test_async_websocket_watcher_falls_back_to_polling_on_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop_sleep(s: float) -> None:
        pass

    monkeypatch.setattr(watchers_module.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(
        watchers_module,
        "async_ws_connect",
        lambda *a, **kw: _raise_os_error_ctx(),
    )

    async def fake_poll(task_id: str, wait: float) -> TaskStatusResponse:
        return _status_response(task_id=task_id, status="success")

    fallback = AsyncPollingWatcher(
        poll_status=fake_poll,
        poll_server_wait=5.0,
        poll_client_interval=None,
        default_timeout=30.0,
    )
    watcher = watchers_module.AsyncWebSocketWatcher(
        ws_url_for_task=lambda tid: f"ws://fake/{tid}",
        poll_fallback=fallback,
        fallback_to_poll=True,
        connect_timeout=0.1,
        default_timeout=30.0,
    )

    updates = [u async for u in watcher.iter_updates("task-1")]
    assert [u.task_status for u in updates] == ["success"]


class _raise_os_error_ctx:
    """Async context manager that raises OSError on enter, simulating a broken WS."""

    async def __aenter__(self):
        raise OSError("connection refused")

    async def __aexit__(self, *a):
        pass


# --- usage limit exceeded (async) ---


@pytest.mark.anyio
async def test_async_402_usage_limit_exceeded_raises_explicit_exception() -> None:
    payload = {
        "error": "usage_limit_exceeded",
        "message": "Your page limit has been exceeded.",
        "details": {"currentUsage": 100, "limit": 50},
    }

    def handler(method: str, url: str, **kw: object) -> httpx.Response:
        return httpx.Response(402, json=payload)

    async with _make_async_http_client(handler) as client:
        with pytest.raises(UsageLimitExceededError) as exc_info:
            await client.submit(source="https://example.org/doc.pdf")

    assert exc_info.value.status_code == 402
    assert exc_info.value.current_usage == 100
    assert exc_info.value.limit == 50


# --- submit_and_retrieve_many (direct async client) ---


@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_forwards_per_item_request_headers(
    tmp_path: Path,
) -> None:
    seen_headers: list[dict[str, str] | None] = []

    async def fake_submit(
        source, source_headers, options, raw_result, request_headers=None
    ):
        seen_headers.append(request_headers)
        return _status_response(f"task-{Path(source).name}", "pending")

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        return _status_response(task_id, "success")

    async def fake_fetch(task_id: str, last_status: TaskStatusResponse | None):
        return _convert_payload(task_id.removeprefix("task-"))

    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    p1.write_bytes(b"%PDF-1.4\n")
    p2.write_bytes(b"%PDF-1.4\n")

    async with _make_async_sam_client(fake_submit, fake_wait, fake_fetch) as client:
        results = [
            r
            async for r in client.submit_and_retrieve_many(
                [
                    ConversionItem(source=p1, headers={"X-Tenant-Id": "tenant-a"}),
                    ConversionItem(source=p2, headers={"X-Tenant-Id": "tenant-b"}),
                ],
                max_in_flight=2,
            )
        ]

    assert len(results) == 2
    assert set(map(str, seen_headers)) == {
        str({"X-Tenant-Id": "tenant-a"}),
        str({"X-Tenant-Id": "tenant-b"}),
    }


@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_forwards_per_item_source_headers() -> (
    None
):
    seen_source_headers: list[dict[str, str] | None] = []

    async def fake_submit(
        source, source_headers, options, raw_result, request_headers=None
    ):
        seen_source_headers.append(source_headers)
        return _status_response(f"task-{Path(source).name}", "pending")

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        return _status_response(task_id, "success")

    async def fake_fetch(task_id: str, last_status: TaskStatusResponse | None):
        return _convert_payload(task_id.removeprefix("task-"))

    async with _make_async_sam_client(fake_submit, fake_wait, fake_fetch) as client:
        results = [
            r
            async for r in client.submit_and_retrieve_many(
                [
                    ConversionItem(
                        source="https://example.org/a.pdf",
                        source_headers={"Authorization": "Bearer a"},
                    ),
                    ConversionItem(
                        source="https://example.org/b.pdf",
                        source_headers={"Authorization": "Bearer b"},
                    ),
                ],
                max_in_flight=2,
            )
        ]

    assert len(results) == 2
    assert set(map(str, seen_source_headers)) == {
        str({"Authorization": "Bearer a"}),
        str({"Authorization": "Bearer b"}),
    }


@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_isolates_failures_per_item(
    tmp_path: Path,
) -> None:
    async def fake_submit(
        source, source_headers, options, raw_result, request_headers=None
    ):
        if Path(source).name == "bad.pdf":
            raise ValueError("submit failed")
        return _status_response(f"task-{Path(source).name}", "pending")

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        return _status_response(task_id, "success")

    async def fake_fetch(task_id: str, last_status: TaskStatusResponse | None):
        return _convert_payload(task_id.removeprefix("task-"))

    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1.4\n")
    bad.write_bytes(b"%PDF-1.4\n")

    async with _make_async_sam_client(fake_submit, fake_wait, fake_fetch) as client:
        outcomes = sorted(
            [
                r
                async for r in client.submit_and_retrieve_many(
                    [ConversionItem(source=bad), ConversionItem(source=good)],
                    max_in_flight=2,
                )
            ],
            key=lambda entry: Path(entry[0].source).name,
        )

    assert isinstance(outcomes[0][1], Exception)
    assert str(outcomes[0][1]) == "submit failed"
    assert getattr(outcomes[1][1], "status") == ConversionStatus.SUCCESS


@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_respects_max_in_flight(
    tmp_path: Path,
) -> None:
    state = {"active": 0, "max_seen": 0, "submitted": 0}
    release = asyncio.Event()

    async def fake_submit(
        source, source_headers, options, raw_result, request_headers=None
    ):
        state["active"] += 1
        state["submitted"] += 1
        state["max_seen"] = max(state["max_seen"], state["active"])
        if state["submitted"] >= 2:
            release.set()
        return _status_response(f"task-{Path(source).name}", "pending")

    async def fake_wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
        if task_id in {"task-a.pdf", "task-b.pdf"}:
            await release.wait()
        return _status_response(task_id, "success")

    async def fake_fetch(task_id: str, last_status: TaskStatusResponse | None):
        state["active"] -= 1
        return _convert_payload(task_id.removeprefix("task-"))

    paths = []
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.4\n")
        paths.append(path)

    async with _make_async_sam_client(fake_submit, fake_wait, fake_fetch) as client:
        results = [
            r
            async for r in client.submit_and_retrieve_many(
                [ConversionItem(source=p) for p in paths],
                max_in_flight=2,
            )
        ]

    assert len(results) == 3
    assert state["max_seen"] == 2


@pytest.mark.parametrize("value", [0, -1, MAX_CONCURRENCY_LIMIT + 1])
@pytest.mark.anyio
async def test_async_submit_and_retrieve_many_rejects_invalid_max_in_flight(
    tmp_path: Path, value: int
) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    async with AsyncDoclingServiceClient(url=TEST_BASE_URL) as client:

        class _FakeHttp:
            async def request(self, *a, **kw):
                raise AssertionError("should not be called")

            async def aclose(self) -> None:
                pass

        client._async_client = _FakeHttp()  # type: ignore[assignment]
        with pytest.raises(
            ValueError,
            match=f"max_in_flight must be between 1 and {MAX_CONCURRENCY_LIMIT}, got {value}.",
        ):
            async for _ in client.submit_and_retrieve_many(
                [ConversionItem(source=source)],
                max_in_flight=value,
            ):
                pass
