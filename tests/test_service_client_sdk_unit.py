import asyncio
import warnings
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import MethodType, SimpleNamespace

import httpx
import pytest
from docling_core.types.doc import DoclingDocument

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
from docling.service_client import ConversionItem, DoclingServiceClient
from docling.service_client.exceptions import (
    ConversionError,
    ResultExpiredError,
    ServiceError,
    ServiceUnavailableError,
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

    with DoclingServiceClient(url=f"{TEST_BASE_URL}/") as client:
        assert client._base_url == TEST_BASE_URL


@pytest.mark.parametrize(
    "url",
    [
        f"{TEST_BASE_URL}/v1",
        f"{TEST_BASE_URL}/v1/",
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
        with pytest.raises(ConversionError, match="conversion failed upstream"):
            client._raise_for_result_404(
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
        ):
            calls.append(
                {
                    "count": len(item_list),
                    "max_in_flight": max_in_flight,
                    "ordered": ordered,
                    "source_headers": [item.source_headers for item in item_list],
                    "request_headers": [item.headers for item in item_list],
                }
            )
            for item in item_list:
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
            self, source, source_headers, options, async_client, request_headers=None
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
            self, source, source_headers, options, async_client, request_headers=None
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
    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    seen_source_headers: list[dict[str, str] | None] = []

    with DoclingServiceClient(
        url=TEST_BASE_URL,
        status_watcher="polling",
    ) as client:

        def fake_build_async_http_client(self):
            return _DummyAsyncClient()

        async def fake_submit(
            self, source, source_headers, options, async_client, request_headers=None
        ):
            seen_source_headers.append(source_headers)
            return _status_response(f"task-{Path(source).name}", "pending")

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
            self, source, source_headers, options, async_client, request_headers=None
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
            self, source, source_headers, options, async_client, request_headers=None
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
            client.submit_and_retrieve_many(
                [ConversionItem(source=path) for path in paths],
                max_in_flight=2,
            )
        )

    assert state["max_seen"] == 2


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


# ── submit_chunk: ChunkerKind and ChunkerOptions ──────────────────────────────


def test_submit_chunk_with_chunker_kind_uses_defaults(tmp_path: Path) -> None:
    """ChunkerKind.HYBRID still works (backward compat) and sends default options."""
    from docling.service_client import ChunkerKind

    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["fields"] = dict(
            (k, v) for k, v in request.headers.items() if k.startswith("content-")
        )
        return httpx.Response(
            200, json=_status_response("task-chunk-1", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport, timeout=client._http_client.timeout
        )
        job = client.submit_chunk(source=sample, chunker=ChunkerKind.HYBRID)

    assert job.task_id == "task-chunk-1"
    assert captured["path"] == "/v1/chunk/hybrid/file/async"


def test_submit_chunk_with_hybrid_options_sends_custom_params(tmp_path: Path) -> None:
    """HybridChunkerOptions with custom tokenizer/max_tokens/merge_peers are transmitted."""
    from docling.datamodel.service.chunking import HybridChunkerOptions

    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        return httpx.Response(
            200, json=_status_response("task-chunk-2", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    chunker_opts = HybridChunkerOptions(
        tokenizer="BAAI/bge-m3",
        max_tokens=2500,
        merge_peers=True,
        include_raw_text=True,
    )

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport, timeout=client._http_client.timeout
        )
        job = client.submit_chunk(source=sample, chunker=chunker_opts)

    assert job.task_id == "task-chunk-2"
    assert captured["path"] == "/v1/chunk/hybrid/file/async"
    body = captured["body"].decode("utf-8")  # type: ignore[union-attr]
    assert "BAAI/bge-m3" in body
    assert "2500" in body
    assert "include_raw_text" in body


def test_submit_chunk_with_hierarchical_options(tmp_path: Path) -> None:
    """HierarchicalChunkerOptions routes to the hierarchical endpoint."""
    from docling.datamodel.service.chunking import HierarchicalChunkerOptions

    captured: dict[str, object] = {}
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200, json=_status_response("task-chunk-3", "pending").model_dump(mode="json")
        )

    transport = httpx.MockTransport(handler)

    with DoclingServiceClient(url=TEST_BASE_URL) as client:
        client._http_client.close()
        client._http_client = httpx.Client(
            transport=transport, timeout=client._http_client.timeout
        )
        job = client.submit_chunk(
            source=sample, chunker=HierarchicalChunkerOptions()
        )

    assert job.task_id == "task-chunk-3"
    assert captured["path"] == "/v1/chunk/hierarchical/file/async"
