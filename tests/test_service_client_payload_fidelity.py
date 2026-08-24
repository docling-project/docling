"""Does the server receive everything the caller configured on the client?

Motivated by a shipped regression: ``engine_options`` is declared as its base
class, so Pydantic serialised it against the *base* schema and silently
dropped every subclass field. A caller who configured a custom engine got a
request carrying only ``engine_type``; the URL, params and timeout never left
the process, and nothing failed loudly. The fix was to annotate the field
``SerializeAsAny``.

These tests assert the shape of the *request* the client puts on the wire, so
the fake here is only a recorder -- it models no part of the docling-serve
response contract and cannot drift from it. The one response it does return
is built from this repo's own ``TaskStatusResponse`` model.
"""

from __future__ import annotations

import json
import re
import typing
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import AnyUrl, BaseModel

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.responses import TaskStatusResponse
from docling.datamodel.service.targets import InBodyTarget
from docling.datamodel.service.tasks import TaskType
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
    VllmVlmEngineOptions,
    VlmEngineType,
)
from docling.service_client import DoclingServiceClient
from docling.service_client.client import StatusWatcherKind
from tests.fakes.http_service import FakeHttpService, RecordedRequest, Response

SOURCE = "https://example.com/report.pdf"


@pytest.fixture
def recorder() -> Iterator[FakeHttpService]:
    """A service that accepts a submission and records nothing else."""
    service = FakeHttpService()
    service.start()

    def _accept(request: RecordedRequest, match: re.Match[str]) -> Response:
        status = TaskStatusResponse(
            task_id="task-1",
            task_type=TaskType.CONVERT,
            task_status=ConversionStatus.PENDING,
        )
        return Response(body=json.loads(status.model_dump_json()))

    service.add_route("POST", r"/v1/convert/source/async", _accept)
    service.add_route("POST", r"/v1/convert/file/async", _accept)
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def client(recorder: FakeHttpService) -> Iterator[DoclingServiceClient]:
    with DoclingServiceClient(
        url=recorder.base_url, status_watcher=StatusWatcherKind.POLLING
    ) as remote:
        yield remote


def _submitted_options(service: FakeHttpService) -> dict[str, Any]:
    """The options block of the JSON submission the server actually received."""
    requests = service.requests_for("POST", r"/v1/convert/source/async")
    assert requests, "no submission reached the server"
    return requests[-1].json()["options"]


def _api_engine() -> ApiVlmEngineOptions:
    return ApiVlmEngineOptions(
        engine_type=VlmEngineType.API_OPENAI,
        url=AnyUrl("https://vlm.example.com/v1/chat/completions"),
        params={"model": "custom-vlm", "temperature": 0.1},
        timeout=123.0,
        headers={"X-Tenant": "acme"},
        concurrency=4,
    )


# -- the regression this module exists for -------------------------------


def test_custom_vlm_engine_options_reach_the_server(client, recorder):
    """Every field of a custom engine must survive serialisation.

    Without ``SerializeAsAny`` on the field, only ``engine_type`` arrives and
    the rest is dropped without error -- the caller's configuration is lost.
    """
    engine = _api_engine()
    client.submit(
        SOURCE,
        options=ConvertDocumentsOptions(
            vlm_pipeline_custom_config=VlmConvertOptions.from_preset(
                "smoldocling", engine_options=engine
            )
        ),
        target=InBodyTarget(),
    )

    received = _submitted_options(recorder)["vlm_pipeline_custom_config"][
        "engine_options"
    ]

    assert received["url"] == "https://vlm.example.com/v1/chat/completions"
    assert received["params"] == {"model": "custom-vlm", "temperature": 0.1}
    assert received["timeout"] == 123.0
    assert received["headers"] == {"X-Tenant": "acme"}
    assert received["concurrency"] == 4
    assert received["engine_type"] == VlmEngineType.API_OPENAI.value


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        pytest.param(
            TransformersVlmEngineOptions(
                engine_type=VlmEngineType.TRANSFORMERS,
                torch_dtype="bfloat16",
                quantized=True,
                load_in_8bit=False,
                trust_remote_code=True,
            ),
            {
                "torch_dtype": "bfloat16",
                "quantized": True,
                "load_in_8bit": False,
                "trust_remote_code": True,
            },
            id="transformers",
        ),
        pytest.param(
            MlxVlmEngineOptions(engine_type=VlmEngineType.MLX, trust_remote_code=True),
            {"trust_remote_code": True},
            id="mlx",
        ),
        pytest.param(
            VllmVlmEngineOptions(
                engine_type=VlmEngineType.VLLM,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.75,
                trust_remote_code=True,
            ),
            {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.75,
                "trust_remote_code": True,
            },
            id="vllm",
        ),
    ],
)
def test_every_engine_variant_keeps_its_own_fields(client, recorder, engine, expected):
    """Each engine subclass carries different fields; none may be flattened."""
    client.submit(
        SOURCE,
        options=ConvertDocumentsOptions(
            vlm_pipeline_custom_config=VlmConvertOptions.from_preset(
                "smoldocling", engine_options=engine
            )
        ),
        target=InBodyTarget(),
    )

    received = _submitted_options(recorder)["vlm_pipeline_custom_config"][
        "engine_options"
    ]

    for key, value in expected.items():
        assert received.get(key) == value, f"{key} was lost in serialisation"


def test_free_form_custom_config_reaches_the_server_intact(client, recorder):
    """Nested dict options must not be flattened or stringified on the way out."""
    ocr_config = {"engine": "custom-ocr", "nested": {"lang": ["en", "de"], "dpi": 300}}
    client.submit(
        SOURCE,
        options=ConvertDocumentsOptions(
            ocr_custom_config=ocr_config,
            layout_custom_config={"repo_id": "acme/layout", "threshold": 0.42},
        ),
        target=InBodyTarget(),
    )

    received = _submitted_options(recorder)

    assert received["ocr_custom_config"] == ocr_config
    assert received["layout_custom_config"] == {
        "repo_id": "acme/layout",
        "threshold": 0.42,
    }


def test_secret_values_are_sent_unwrapped(client, recorder):
    """SecretStr must reach the server as its value, not as '**********'."""
    client.submit(
        SOURCE,
        options=ConvertDocumentsOptions(
            vlm_pipeline_custom_config=VlmConvertOptions.from_preset(
                "smoldocling",
                engine_options=ApiVlmEngineOptions(
                    engine_type=VlmEngineType.API_OPENAI,
                    url=AnyUrl("https://vlm.example.com/v1"),
                    headers={"Authorization": "Bearer sk-not-a-real-key"},
                ),
            )
        ),
        target=InBodyTarget(),
    )

    received = _submitted_options(recorder)["vlm_pipeline_custom_config"][
        "engine_options"
    ]

    assert received["headers"]["Authorization"] == "Bearer sk-not-a-real-key"
    assert "*" not in received["headers"]["Authorization"]


# -- the file-upload route uses a different encoder ----------------------


def test_file_uploads_carry_nested_options_as_json_fields(client, recorder, tmp_path):
    """Multipart fields take only primitives, so nested options are JSON encoded."""
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    ocr_config = {"engine": "custom-ocr", "nested": {"lang": ["en"]}}

    client.submit(
        source,
        options=ConvertDocumentsOptions(ocr_custom_config=ocr_config),
        target=InBodyTarget(),
    )

    uploads = recorder.requests_for("POST", r"/v1/convert/file/async")
    assert uploads, "the file route was not used for a local path"
    body = uploads[-1].body.decode("utf-8", errors="replace")
    # The field is present and holds the JSON encoding of the nested value.
    assert "ocr_custom_config" in body
    assert json.dumps(ocr_config) in body


# -- a guard against the next occurrence ---------------------------------


def _polymorphic_option_fields() -> list[tuple[str, str, str, bool]]:
    """Walk the request options graph for fields that can hold a subclass.

    Returns ``(model, field, declared base, is SerializeAsAny)`` for every
    field whose declared type is a Pydantic model that has subclasses.
    """

    def descendants(cls: type) -> set[type]:
        found: set[type] = set()
        for sub in cls.__subclasses__():
            found.add(sub)
            found |= descendants(sub)
        return found

    seen: set[type] = set()
    todo: list[type] = [ConvertDocumentsOptions]
    findings: set[tuple[str, str, str, bool]] = set()
    while todo:
        model = todo.pop()
        if model in seen or not (
            isinstance(model, type) and issubclass(model, BaseModel)
        ):
            continue
        seen.add(model)
        for name, field in model.model_fields.items():
            # SerializeAsAny is recorded as a metadata marker; the annotation
            # itself still reports the bare base class.
            protected = any(
                type(marker).__name__ == "SerializeAsAny" for marker in field.metadata
            )
            for arg in typing.get_args(field.annotation) or (field.annotation,):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    todo.append(arg)
                    if descendants(arg):
                        findings.add((model.__name__, name, arg.__name__, protected))
    return sorted(findings)


def test_polymorphic_option_fields_are_serialized_as_any():
    """Any option field holding a subclass must serialise the subclass schema.

    Declaring such a field without ``SerializeAsAny`` makes Pydantic dump it
    against the base schema, dropping subclass fields silently. Walking the
    options graph means a newly added field is covered without editing this
    test -- which is the point, since the failure mode is silent.
    """
    fields = _polymorphic_option_fields()
    assert fields, (
        "no polymorphic option fields found - this check would pass trivially "
        "and must be revisited"
    )

    offenders = [
        f"{model}.{field} (declared as {base})"
        for model, field, base, protected in fields
        if not protected
    ]
    assert not offenders, (
        "these option fields hold subclasses but are not SerializeAsAny, so "
        "subclass fields will be dropped from the request: " + ", ".join(offenders)
    )
