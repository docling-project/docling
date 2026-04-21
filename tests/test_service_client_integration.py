import os
from pathlib import Path

import pytest

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.service_client import DoclingServiceClient, RawServiceResult

SERVICE_URL_ENV = "DOCLING_SERVICE_URL"
SERVICE_API_KEY_ENV = "DOCLING_SERVICE_API_KEY"
SERVICE_URL = os.environ.get(SERVICE_URL_ENV)
SERVICE_API_KEY = os.environ.get(SERVICE_API_KEY_ENV)
FIXTURES_DIR = Path(__file__).resolve().parent / "data" / "pdf"

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
    return SERVICE_API_KEY


def _json_options() -> ConvertDocumentsRequestOptions:
    return ConvertDocumentsRequestOptions(
        do_ocr=False,
        do_table_structure=False,
        include_images=False,
        to_formats=[OutputFormat.JSON],
        abort_on_error=False,
    )


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

        job = client.submit(source=source, target_format=OutputFormat.JSON)
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
            source=source, options=options, target_format=OutputFormat.MARKDOWN
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
                sources=[source_a, source_b],
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
