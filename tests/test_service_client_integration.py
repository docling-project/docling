import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pytest
from docling_core.types.doc.base import ImageRefMode

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import AnyHttpSourceRequest
from docling.datamodel.service.responses import TaskStatusResponse
from docling.datamodel.service.targets import (
    InBodyTarget,
    PresignedUrlTarget,
    ZipTarget,
)
from docling.service_client import (
    AsyncDoclingServiceClient,
    DoclingServiceClient,
    RawServiceResult,
    ServiceUnavailableError,
    TaskTimeoutError,
)

SERVICE_URL_ENV = "DOCLING_SERVICE_URL"
SERVICE_API_KEY_ENV = "DOCLING_SERVICE_API_KEY"
SERVICE_URL = os.environ.get(SERVICE_URL_ENV)
SERVICE_API_KEY = os.environ.get(SERVICE_API_KEY_ENV)
FIXTURES_DIR = Path(__file__).resolve().parent / "data" / "pdf"
BATCH_SAMPLE_SOURCES = [
    "https://arxiv.org/pdf/2206.01062",
]


class _RedactedSecret(str):
    def __repr__(self) -> str:
        return "'<redacted>'"


class _WatchableJob(Protocol):
    def watch(self, timeout: float | None = None) -> Iterator[TaskStatusResponse]: ...


pytestmark = [
    pytest.mark.skipif(
        bool(os.environ.get("CI")),
        reason="requires a running external docling-serve host; disabled in CI",
    ),
    pytest.mark.skipif(
        not SERVICE_URL,
        reason=f"requires a running docling-serve host; set {SERVICE_URL_ENV} to run",
    ),
]


@pytest.fixture(scope="module")
def live_service_url() -> str:
    assert SERVICE_URL is not None
    return SERVICE_URL.rstrip("/")


@pytest.fixture(scope="module")
def service_api_key() -> str | None:
    if SERVICE_API_KEY is None:
        return None
    return _RedactedSecret(SERVICE_API_KEY)


def _json_options() -> ConvertDocumentsRequestOptions:
    return ConvertDocumentsRequestOptions(
        do_ocr=False,
        do_table_structure=False,
        include_images=False,
        to_formats=[OutputFormat.JSON],
        abort_on_error=False,
        image_export_mode=ImageRefMode.REFERENCED,
    )


def _watch_terminal_without_poll_fallback(
    job: _WatchableJob,
) -> list[TaskStatusResponse]:
    updates: list[TaskStatusResponse] = []
    try:
        for update in job.watch(timeout=30.0):
            updates.append(update)
    except (ServiceUnavailableError, TaskTimeoutError) as exc:
        statuses = [update.task_status.value for update in updates]
        if len(statuses) > 12:
            status_summary = (
                f"{len(statuses)} updates; "
                f"first={statuses[:4]}; last={statuses[-4:]}; "
                f"unique={sorted(set(statuses))}"
            )
        else:
            status_summary = repr(statuses)
        pytest.fail(
            "WebSocket watcher did not emit terminal status without poll fallback; "
            f"received statuses: {status_summary}; error: {exc}",
            pytrace=False,
        )
    return updates


def test_convert_and_submit_with_polling_watcher(
    live_service_url: str, service_api_key: str | None, tmp_path: Path
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
        options=_json_options(),
    ) as client:
        health = client.health()
        assert health.status == "ok"

        converted = client.convert(source=source)
        assert converted.status.value in {"success", "partial_success"}
        assert converted.document.name == "2206.01062"

        job = client.submit(source=source, target=InBodyTarget())
        submitted = job.result(timeout=300.0)
        assert submitted.status.value in {"success", "partial_success"}
        assert submitted.document.name == "2206.01062"


def test_submit_non_json_returns_raw_payload(
    live_service_url: str, service_api_key: str | None, tmp_path: Path
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        options = ConvertDocumentsRequestOptions(
            do_ocr=False,
            do_table_structure=False,
            include_images=False,
            to_formats=[OutputFormat.MARKDOWN],
            abort_on_error=False,
        )
        job = client.submit(
            source=source,
            options=options,
            output_formats=[OutputFormat.MARKDOWN],
            target=ZipTarget(),
        )
        raw_result = job.result(timeout=300.0)

        assert isinstance(raw_result, RawServiceResult)
        assert len(raw_result.content) > 0
        assert "zip" in raw_result.content_type


def test_convert_all_preserves_input_order(
    live_service_url: str, service_api_key: str | None, tmp_path: Path
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()
    source_a = tmp_path / "order-a.pdf"
    source_b = tmp_path / "order-b.pdf"
    source_a.write_bytes(source.read_bytes())
    source_b.write_bytes(source.read_bytes())

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        results = list(
            client.convert_all(
                source=[source_a, source_b],
                options=_json_options(),
                max_concurrency=2,
            )
        )

    assert len(results) == 2
    assert results[0].input.file.name == "order-a.pdf"
    assert results[1].input.file.name == "order-b.pdf"


def test_websocket_watcher_end_to_end(
    live_service_url: str, service_api_key: str | None, tmp_path: Path
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="websocket",
        ws_fallback_to_poll=True,
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        result = client.convert(source=source, options=_json_options())

    assert result.status.value in {"success", "partial_success"}
    assert result.document.name == "2206.01062"


def test_websocket_watcher_reaches_terminal_without_poll_fallback(
    live_service_url: str, service_api_key: str | None
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="websocket",
        ws_fallback_to_poll=False,
        job_timeout=30.0,
    ) as client:
        job = client.submit(
            source=source,
            options=_json_options(),
            output_formats=[OutputFormat.JSON],
            target=PresignedUrlTarget(),
        )
        updates = _watch_terminal_without_poll_fallback(job)

    assert updates
    assert updates[-1].task_status.value == "success"


def test_submit_batch_websocket_watcher_reaches_terminal_without_poll_fallback(
    live_service_url: str, service_api_key: str | None
) -> None:
    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="websocket",
        ws_fallback_to_poll=False,
        job_timeout=30.0,
    ) as client:
        job = client.submit_batch(
            sources=[
                AnyHttpSourceRequest(url=source) for source in BATCH_SAMPLE_SOURCES
            ],
            target=PresignedUrlTarget(),
            output_formats=[OutputFormat.JSON],
            options=_json_options(),
        )
        updates = _watch_terminal_without_poll_fallback(job)
        result = job.result(timeout=1.0)

    assert updates
    assert updates[-1].task_status.value == "success"
    assert result.num_succeeded == 1
    assert result.num_failed == 0


def test_submit_accepts_custom_request_headers(
    live_service_url: str,
    service_api_key: str | None,
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    with DoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        job = client.submit(
            source=source,
            options=_json_options(),
            headers={"X-Tenant-Id": "tenant-integration"},
        )
        result = job.result(timeout=300.0)

    assert result.status.value in {"success", "partial_success"}


# ---------------------------------------------------------------------------
# Async integration tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_convert_with_polling_watcher(
    live_service_url: str, service_api_key: str | None
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    async with AsyncDoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
        options=_json_options(),
    ) as client:
        health = await client.health()
        assert health.status == "ok"

        version = await client.version()
        assert isinstance(version, dict)

        job = await client.submit(source=source, target=InBodyTarget())
        result = await job.result(timeout=300.0)
        assert result.status.value in {"success", "partial_success"}
        assert result.document.name == "2206.01062"


@pytest.mark.anyio
async def test_async_submit_non_json_returns_raw_payload(
    live_service_url: str, service_api_key: str | None
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    options = ConvertDocumentsRequestOptions(
        do_ocr=False,
        do_table_structure=False,
        include_images=False,
        to_formats=[OutputFormat.MARKDOWN],
        abort_on_error=False,
    )
    async with AsyncDoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        job = await client.submit(
            source=source,
            options=options,
            output_formats=[OutputFormat.MARKDOWN],
            target=ZipTarget(),
        )
        raw_result = await job.result(timeout=300.0)

    assert isinstance(raw_result, RawServiceResult)
    assert len(raw_result.content) > 0
    assert "zip" in raw_result.content_type


@pytest.mark.anyio
async def test_async_websocket_watcher_end_to_end(
    live_service_url: str, service_api_key: str | None
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    async with AsyncDoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="websocket",
        ws_fallback_to_poll=True,
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        job = await client.submit(source=source, options=_json_options())
        result = await job.result(timeout=300.0)

    assert result.status.value in {"success", "partial_success"}
    assert result.document.name == "2206.01062"


@pytest.mark.anyio
async def test_async_submit_accepts_custom_request_headers(
    live_service_url: str, service_api_key: str | None
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()

    async with AsyncDoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        job = await client.submit(
            source=source,
            options=_json_options(),
            headers={"X-Tenant-Id": "tenant-async-integration"},
        )
        result = await job.result(timeout=300.0)

    assert result.status.value in {"success", "partial_success"}


@pytest.mark.anyio
async def test_async_submit_and_retrieve_each_preserves_per_item_results(
    live_service_url: str, service_api_key: str | None, tmp_path: Path
) -> None:
    source = FIXTURES_DIR / "2206.01062.pdf"
    assert source.exists()
    source_a = tmp_path / "async-order-a.pdf"
    source_b = tmp_path / "async-order-b.pdf"
    source_a.write_bytes(source.read_bytes())
    source_b.write_bytes(source.read_bytes())

    from docling.service_client import ConversionItem

    items = [
        ConversionItem(source=source_a, options=_json_options()),
        ConversionItem(source=source_b, options=_json_options()),
    ]

    async with AsyncDoclingServiceClient(
        url=live_service_url,
        api_key=service_api_key,
        status_watcher="polling",
        poll_server_wait=0.2,
        job_timeout=300.0,
    ) as client:
        pairs = [
            pair
            async for pair in client.submit_and_retrieve_each(
                items=items, max_in_flight=2
            )
        ]

    assert len(pairs) == 2
    for item, result_or_exc in pairs:
        assert not isinstance(result_or_exc, Exception)
        assert result_or_exc.status.value in {"success", "partial_success"}
