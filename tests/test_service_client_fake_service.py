"""Service client tests driven against a real in-process docling-serve fake.

The existing unit tests reach into ``client._http_client`` to install an
``httpx.MockTransport``, which cannot exercise anything below the request
layer. Here the client talks to a real socket, so submission, the polling
loop, target negotiation, retries, artifact download and the error taxonomy
all run as they would against a live service.

Assertions target outcomes that survive a wire-format change -- the resulting
document, the status, the exception type -- not the exact bytes exchanged.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

import docling.service_client.client as client_module
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.targets import InBodyTarget, PresignedUrlTarget
from docling.service_client import (
    AsyncDoclingServiceClient,
    ConversionError,
    DoclingServiceClient,
    ServiceUnavailableError,
)
from docling.service_client.client import ConversionItem, StatusWatcherKind
from docling.service_client.exceptions import ServiceError, TaskNotFoundError
from tests.fakes.docling_serve import FakeDoclingServe
from tests.fakes.http_service import FakeHttpService, Response

SOURCE = "https://example.com/report.pdf"


@pytest.fixture
def serve() -> Iterator[FakeDoclingServe]:
    """A running fake docling-serve; also exposes the underlying HTTP service."""
    fake_service = FakeHttpService()
    base_url = fake_service.start()
    try:
        yield FakeDoclingServe(fake_service, base_url)
    finally:
        fake_service.stop()


@pytest.fixture
def service(serve: FakeDoclingServe) -> FakeHttpService:
    return serve.service


@pytest.fixture
def client(service: FakeHttpService) -> Iterator[DoclingServiceClient]:
    with DoclingServiceClient(
        url=service.base_url,
        # The fake answers polls immediately rather than long-polling, so drop
        # the client-side cadence that would otherwise pace each poll by 5s.
        status_watcher=StatusWatcherKind.POLLING,
        poll_server_wait=0.01,
        poll_client_interval=0.01,
    ) as remote:
        yield remote


@pytest.fixture
def allow_loopback_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let artifact downloads reach the loopback fake.

    ``_is_safe_artifact_url`` is an SSRF guard that rejects non-routable hosts
    by design, which includes the test server. The guard itself is covered by
    ``test_presigned_artifact_on_a_loopback_host_is_refused``.
    """
    monkeypatch.setattr(client_module, "_is_safe_artifact_url", lambda url: True)


# -- basic endpoints -----------------------------------------------------


def test_health_and_version_reach_the_service(client, service):
    assert client.health().status == "ok"
    assert client.version()["version"] == "0.0.0-fake"
    assert [r.path for r in service.requests] == ["/health", "/version"]


def test_base_url_including_v1_is_rejected(service):
    """The client owns the /v1 prefix; callers pass the service root."""
    with pytest.raises(ValueError, match="not include /v1"):
        DoclingServiceClient(url=f"{service.base_url}/v1")


@pytest.mark.parametrize("bad_url", ["ftp://example.com", "not-a-url", ""])
def test_non_http_base_urls_are_rejected(bad_url):
    with pytest.raises(ValueError, match="absolute http"):
        DoclingServiceClient(url=bad_url)


def test_base_url_with_query_or_fragment_is_rejected(service):
    with pytest.raises(ValueError, match="query or fragment"):
        DoclingServiceClient(url=f"{service.base_url}/?token=abc")


# -- the convert flow ----------------------------------------------------


def test_convert_with_an_inbody_target_returns_the_document(client, service):
    job = client.submit(SOURCE, target=InBodyTarget())
    result = job.result()

    assert result.status == ConversionStatus.SUCCESS
    assert "Fake service result" in result.document.export_to_markdown()


def test_convert_polls_until_the_task_reaches_a_terminal_status(
    client, service, serve, allow_loopback_artifacts
):
    serve.polls_before_success = 3

    result = client.convert(SOURCE)

    assert result.status == ConversionStatus.SUCCESS
    polls = service.requests_for("GET", r"/v1/status/poll/.*")
    # pending -> started x3 -> success: the loop genuinely iterated.
    assert len(polls) == 4


def test_convert_downloads_presigned_artifacts(
    client, service, allow_loopback_artifacts
):
    result = client.convert(SOURCE)

    assert result.status == ConversionStatus.SUCCESS
    assert "Fake service result" in result.document.export_to_markdown()
    assert service.requests_for("GET", r"/artifacts/.*/json")


def test_presigned_artifact_on_a_loopback_host_is_refused(client):
    """The SSRF guard must reject artifact URLs that are not globally routable."""
    result = client.convert(SOURCE, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert any("non-public URL" in item.error_message for item in result.errors), (
        result.errors
    )


def test_convert_falls_back_to_inbody_when_presigned_is_unsupported(client, service):
    """A service without artifact storage rejects presigned; the client retargets."""
    # Only the first submission is rejected, so the InBody retry succeeds.
    service.respond_once(
        "POST",
        r"/v1/convert/source/async",
        Response(
            status=422,
            body={
                "detail": "This deployment requires artifact storage to be configured"
            },
        ),
    )

    result = client.convert(SOURCE)

    assert result.status == ConversionStatus.SUCCESS
    assert len(service.requests_for("POST", r"/v1/convert/source/async")) == 2


# -- failure handling ----------------------------------------------------


def test_task_failure_raises_conversion_error(client, serve):
    serve.terminal_status = ConversionStatus.FAILURE

    with pytest.raises(ConversionError):
        client.convert(SOURCE)


def test_task_failure_is_returned_when_raises_on_error_is_false(client, serve):
    serve.terminal_status = ConversionStatus.FAILURE

    result = client.convert(SOURCE, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE


def test_unknown_task_poll_raises_task_not_found(client, service):
    service.add_route(
        "GET",
        r"/v1/status/poll/.*",
        lambda request, match: Response(status=404, body={"detail": "gone"}),
    )

    with pytest.raises(TaskNotFoundError):
        client.convert(SOURCE)


def test_server_error_on_submit_raises_a_service_error(client, service):
    service.add_route(
        "POST",
        r"/v1/convert/source/async",
        lambda request, match: Response(status=500, body={"detail": "boom"}),
    )

    with pytest.raises(ServiceError):
        client.convert(SOURCE)


def test_unreachable_service_raises_service_unavailable(service):
    base_url = service.base_url
    service.stop()

    with DoclingServiceClient(
        url=base_url, status_watcher=StatusWatcherKind.POLLING
    ) as remote:
        with pytest.raises(ServiceUnavailableError):
            remote.health()


# -- retries -------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502])
def test_server_errors_on_poll_are_retried(
    client, service, allow_loopback_artifacts, status
):
    """500 and 502 retry on an exponential backoff, with no header needed."""
    service.respond_once(
        "GET", r"/v1/status/poll/.*", Response(status=status, body={"detail": "boom"})
    )

    result = client.convert(SOURCE)

    assert result.status == ConversionStatus.SUCCESS
    assert len(service.requests_for("GET", r"/v1/status/poll/.*")) >= 3


@pytest.mark.parametrize("status", [429, 503])
def test_throttling_responses_are_retried_when_retry_after_is_present(
    client, service, allow_loopback_artifacts, status
):
    service.respond_once(
        "GET",
        r"/v1/status/poll/.*",
        Response(
            status=status, body={"detail": "slow down"}, headers={"Retry-After": "0"}
        ),
    )

    result = client.convert(SOURCE)

    assert result.status == ConversionStatus.SUCCESS
    assert len(service.requests_for("GET", r"/v1/status/poll/.*")) >= 3


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        # 4xx and 5xx map to different exception types in the taxonomy.
        (429, ServiceError),
        (503, ServiceUnavailableError),
    ],
)
def test_throttling_without_retry_after_is_surfaced_immediately(
    client, service, status, expected
):
    """Without the header the client cannot know how long to wait, so it gives up."""
    service.add_route(
        "GET",
        r"/v1/status/poll/.*",
        lambda request, match: Response(status=status, body={"detail": "no header"}),
    )

    with pytest.raises(expected):
        client.convert(SOURCE)

    assert len(service.requests_for("GET", r"/v1/status/poll/.*")) == 1


def test_client_error_on_poll_is_not_retried(client, service):
    service.add_route(
        "GET",
        r"/v1/status/poll/.*",
        lambda request, match: Response(status=400, body={"detail": "bad request"}),
    )

    with pytest.raises(ServiceError):
        client.convert(SOURCE)

    assert len(service.requests_for("GET", r"/v1/status/poll/.*")) == 1


# -- multiple documents --------------------------------------------------


def test_convert_all_yields_one_result_per_source(
    client, service, allow_loopback_artifacts
):
    sources = [
        "https://example.com/a.pdf",
        "https://example.com/b.pdf",
        "https://example.com/c.pdf",
    ]

    results = list(client.convert_all(sources))

    assert len(results) == 3
    assert all(r.status == ConversionStatus.SUCCESS for r in results)
    assert len(service.requests_for("POST", r"/v1/convert/source/async")) == 3


def test_submit_returns_a_job_whose_status_can_be_polled(client, serve):
    serve.polls_before_success = 2

    job = client.submit(SOURCE, target=InBodyTarget())
    statuses = [update.task_status for update in job.watch(timeout=30)]

    assert statuses[-1] == ConversionStatus.SUCCESS


# -- the async client ----------------------------------------------------
#
# The same fake serves AsyncDoclingServiceClient unchanged: a real socket does
# not care which client library or event loop is on the other end.


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def async_client_kwargs(service: FakeHttpService) -> dict[str, object]:
    return {
        "url": service.base_url,
        "status_watcher": StatusWatcherKind.POLLING,
        "poll_server_wait": 0.01,
        "poll_client_interval": 0.01,
    }


@pytest.mark.anyio
async def test_async_client_reaches_health_and_version(async_client_kwargs, service):
    async with AsyncDoclingServiceClient(**async_client_kwargs) as remote:
        assert (await remote.health()).status == "ok"
        assert (await remote.version())["version"] == "0.0.0-fake"


@pytest.mark.anyio
async def test_async_submit_awaits_the_converted_document(async_client_kwargs, serve):
    serve.polls_before_success = 2

    async with AsyncDoclingServiceClient(**async_client_kwargs) as remote:
        job = await remote.submit(SOURCE, target=InBodyTarget())
        result = await job.result(timeout=30)

    assert result.status == ConversionStatus.SUCCESS
    assert "Fake service result" in result.document.export_to_markdown()


@pytest.mark.anyio
async def test_async_watch_yields_every_status_transition(async_client_kwargs, serve):
    serve.polls_before_success = 2

    async with AsyncDoclingServiceClient(**async_client_kwargs) as remote:
        job = await remote.submit(SOURCE, target=InBodyTarget())
        statuses = [update.task_status async for update in job.watch(timeout=30)]

    assert statuses[-1] == ConversionStatus.SUCCESS
    assert ConversionStatus.STARTED in statuses


@pytest.mark.anyio
async def test_async_submit_and_retrieve_each_covers_every_item(
    async_client_kwargs, service
):
    items = [
        ConversionItem(source="https://example.com/a.pdf", metadata="a"),
        ConversionItem(source="https://example.com/b.pdf", metadata="b"),
    ]

    async with AsyncDoclingServiceClient(**async_client_kwargs) as remote:
        pairs = [
            pair
            async for pair in remote.submit_and_retrieve_each(
                items, target=InBodyTarget()
            )
        ]

    assert len(pairs) == 2
    assert {item.metadata for item, _ in pairs} == {"a", "b"}
    assert not [result for _, result in pairs if isinstance(result, Exception)]
    assert len(service.requests_for("POST", r"/v1/convert/source/async")) == 2


@pytest.mark.anyio
async def test_async_client_surfaces_submit_errors(async_client_kwargs, service):
    service.add_route(
        "POST",
        r"/v1/convert/source/async",
        lambda request, match: Response(status=500, body={"detail": "boom"}),
    )

    async with AsyncDoclingServiceClient(**async_client_kwargs) as remote:
        with pytest.raises(ServiceError):
            await remote.submit(SOURCE, target=InBodyTarget())


BATCH_SOURCES = [
    {"kind": "http", "url": "https://example.com/a.pdf"},
    {"kind": "http", "url": "https://example.com/b.pdf"},
]


def test_submit_batch_reaches_the_batch_endpoint(client, service):
    """Batch submission uses its own route, distinct from per-source submits."""
    job = client.submit_batch(sources=BATCH_SOURCES, target=PresignedUrlTarget())

    assert job.task_id
    assert len(service.requests_for("POST", r"/v1/convert/source/batch")) == 1
    # The per-source route must not have been used for a batch submission.
    assert not service.requests_for("POST", r"/v1/convert/source/async")


def test_submit_batch_rejects_a_non_storage_target(client):
    """Batch results go to storage, so InBody is not a valid batch target."""
    with pytest.raises(ValidationError):
        client.submit_batch(sources=BATCH_SOURCES, target=InBodyTarget())


def test_submit_batch_requires_exactly_one_of_target_or_targets(client):
    with pytest.raises(ValueError, match="requires either"):
        client.submit_batch(sources=BATCH_SOURCES)

    with pytest.raises(ValueError, match="only one"):
        client.submit_batch(
            sources=BATCH_SOURCES,
            target=PresignedUrlTarget(),
            targets=[PresignedUrlTarget()],
        )
