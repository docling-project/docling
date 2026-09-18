# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Offline SDK/wire and source extraction client contracts."""

import json
import subprocess
import sys
from copy import deepcopy

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from docling.datamodel.base_models import (
    ConversionStatus,
    VlmPredictionToken,
    VlmStopReason,
)
from docling.datamodel.extraction import (
    ExtractionItem,
    ExtractionTarget,
    ExtractionTemplate,
    PageScope,
)
from docling.datamodel.extraction_options import GRANITE_VISION_4_1_SPEC
from docling.datamodel.service import ExtractionDocumentResult, ExtractionTaskResult
from docling.datamodel.service.options import ExtractDocumentsOptions
from docling.datamodel.service.requests import ExtractSourcesRequest
from docling.datamodel.service.responses import (
    DoclingTaskResult,
    ExtractDocumentResponse,
)
from docling.models.extraction.prompt_utils import prepare_target
from docling.service_client import AsyncDoclingServiceClient, DoclingServiceClient
from docling.service_client.client import RawServiceResult
from docling.service_client.exceptions import ResponseSchemaMismatchError, ServiceError


class Invoice(BaseModel):
    invoice: str
    total: float


def request(*, storage=False):
    return ExtractSourcesRequest(
        options=ExtractDocumentsOptions(
            target=ExtractionTarget.from_pydantic(
                Invoice,
                template=ExtractionTemplate(
                    format="example_json", value={"invoice": "INV-42", "total": 4.2}
                ),
                instructions="Copy the invoice identifier exactly",
            ),
            extraction_preset="granite_vision_4_1",
            input_channels="image",
            page_range=(2, 3),
        ),
        sources=[
            {
                "kind": "http",
                "url": "https://example.com/report.pdf",
                "headers": {"Authorization": "secret"},
            }
        ],
        target={"kind": "presigned_url"} if storage else {"kind": "inbody"},
    )


def response():
    return ExtractDocumentResponse(
        documents=[
            ExtractionDocumentResult(
                source_index=2,
                source_uri="s3://bucket/archive/report.pdf",
                filename="report.pdf",
                status=ConversionStatus.PARTIAL_SUCCESS,
                items=[
                    ExtractionItem(
                        scope=PageScope(page_no=3),
                        extracted_data={"invoice": "INV-42", "total": 4.2},
                        raw_text='{"invoice":"INV-42","total":4.2}',
                        validation_status="passed",
                        generated_tokens=[VlmPredictionToken(token=42, logprob=-0.2)],
                        generation_time=0.4,
                        num_tokens=12,
                        usage={"completion_tokens": 12},
                        stop_reason=VlmStopReason.END_OF_SEQUENCE,
                    ),
                    ExtractionItem(
                        scope={"kind": "document"},
                        raw_text='{"total":"bad"}',
                        validation_status="failed",
                        errors=["Schema validation at $.total: not a number"],
                    ),
                    ExtractionItem(
                        scope=PageScope(page_no=2),
                        validation_status="not_run",
                        errors=["Inference failed"],
                    ),
                ],
            )
        ],
        processing_time=0.5,
        num_converted=1,
        num_succeeded=0,
        num_partially_succeeded=1,
        num_failed=0,
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"invoice": "INV-42", "total": 4.2},
        {"invoice": "INV-42", "total": "4.2"},
        {"total": 4.2},
    ],
)
def test_sdk_and_wire_validate_original_schema(answer):
    sdk = request().options.target
    wire = ExtractSourcesRequest.model_validate_json(
        request().model_dump_json()
    ).options.target
    before = deepcopy(wire.model_dump())
    outcomes = []
    for target in (sdk, wire):
        prepared = prepare_target(target, GRANITE_VISION_4_1_SPEC)
        assert prepared.validator.schema == target.output_schema
        assert (
            "INV-42" in prepared.prompt
            and "Copy the invoice identifier exactly" in prepared.prompt
        )
        outcomes.append(
            [
                (error.json_path, error.message)
                for error in prepared.validator.iter_errors(answer)
            ]
        )
    assert outcomes[0] == outcomes[1]
    assert bool(outcomes[0]) == (answer != {"invoice": "INV-42", "total": 4.2})
    assert wire.model_dump() == before


@pytest.mark.parametrize(
    "options",
    [
        {"template": {"invoice": "string"}},
        {"target": {}},
        {"target": {"instructions": "extract"}},
        {"target": {"template": {"format": "unknown", "value": {}}}},
        {"target": {"template": {"format": "example_json", "value": "{}"}}},
        {
            "target": {
                "template": {"format": "example_json", "value": {}},
                "validator": "python",
            }
        },
        {
            "target": {"template": {"format": "example_json", "value": {}}},
            "grouping": "whole_document",
        },
        {
            "target": {"template": {"format": "example_json", "value": {}}},
            "output_mode": "unknown",
        },
    ],
)
def test_old_unknown_and_empty_options_rejected(options):
    with pytest.raises(ValidationError):
        ExtractDocumentsOptions.model_validate(options)


def test_document_and_task_json_roundtrips():
    inline = response()
    assert (
        ExtractDocumentResponse.model_validate_json(inline.model_dump_json()) == inline
    )
    task = DoclingTaskResult(
        result=ExtractionTaskResult(documents=inline.documents),
        **inline.model_dump(exclude={"documents"}),
    )
    restored = DoclingTaskResult.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.result.documents[0].items[0].scope.page_no == 3
    assert restored.result.documents[0].source_uri == "s3://bucket/archive/report.pdf"
    payload = inline.documents[0].model_dump()
    for old_field in ("pages", "input", "backend"):
        with pytest.raises(ValidationError):
            ExtractionDocumentResult.model_validate({**payload, old_field: []})


@pytest.mark.parametrize(
    "scope",
    [
        {"kind": "page", "page_no": 0},
        {"kind": "page", "page_no": True},
        {"kind": "document", "page_no": 1},
        {"kind": "chunk"},
    ],
)
def test_malformed_wire_scope_rejected(scope):
    payload = response().model_dump()
    payload["documents"][0]["items"][0]["scope"] = scope
    with pytest.raises(ValidationError):
        ExtractDocumentResponse.model_validate(payload)


def transport(storage, *, admission=200, malformed=False):
    calls = []

    def handle(req):
        calls.append(req)
        if req.method == "POST":
            return httpx.Response(
                admission,
                json={
                    "task_id": "extract-1",
                    "task_type": "extract",
                    "task_status": "success",
                }
                if admission == 200
                else {"detail": "denied"},
            )
        assert req.url.path == "/v1/result/extract-1"
        return httpx.Response(
            200,
            json={"num_converted": 1}
            if malformed
            else (
                {
                    "num_converted": 1,
                    "num_succeeded": 1,
                    "num_failed": 0,
                    "processing_time": 0.1,
                }
                if storage
                else response().model_dump(mode="json")
            ),
        )

    return httpx.MockTransport(handle), calls


def assert_client(calls, value, original, storage):
    assert calls[0].url.path == "/v1/extract/source/async"
    received = json.loads(calls[0].content)
    assert received["options"]["target"] == original.options.target.model_dump(
        mode="json", exclude_none=True
    )
    assert received["target"]["kind"] == ("presigned_url" if storage else "inbody")
    assert received["options"]["page_range"] == [2, 3]
    assert received["sources"][0]["headers"]["Authorization"] == "secret"
    if storage:
        assert value.content_type == "application/json"
    else:
        assert value == response()
    assert isinstance(value, RawServiceResult if storage else ExtractDocumentResponse)


@pytest.mark.parametrize("storage", [False, True])
def test_sync_extraction_payload_and_result(storage):
    tr, calls = transport(storage)
    original = request(storage=storage)
    before = original.model_dump()
    with DoclingServiceClient(url="https://service.example") as client:
        client._http_client.close()
        client._http_client = httpx.Client(transport=tr)
        value = client.submit_extract(original).result()
        changed = original.model_copy(deep=True)
        changed.options.output_mode = "schema_constrained"
        changed.options.target = ExtractionTarget(
            output_schema={
                "type": "object",
                "properties": {"account": {"type": "integer"}},
            },
            template=ExtractionTemplate(format="example_json", value={"account": 17}),
            instructions="Copy account only",
        )
        client.submit_extract(changed)
        assert json.loads(calls[-1].content)["options"][
            "target"
        ] == changed.options.target.model_dump(mode="json", exclude_none=True)
        assert (
            json.loads(calls[-1].content)["options"]["output_mode"]
            == "schema_constrained"
        )
        assert "INV-42" not in calls[-1].content.decode()

    assert_client(calls, value, original, storage)
    assert original.model_dump() == before


@pytest.mark.anyio
@pytest.mark.parametrize("storage", [False, True])
async def test_async_extraction_payload_and_result(storage):
    tr, calls = transport(storage)
    original = request(storage=storage)
    before = original.model_dump()
    async with AsyncDoclingServiceClient(url="https://service.example") as client:
        await client._async_client.aclose()
        client._async_client = httpx.AsyncClient(transport=tr)
        value = await (await client.submit_extract(original)).result()
        changed = original.model_copy(deep=True)
        changed.options.output_mode = "schema_constrained"
        changed.options.target = ExtractionTarget(
            output_schema={
                "type": "object",
                "properties": {"account": {"type": "integer"}},
            },
            template=ExtractionTemplate(format="example_json", value={"account": 17}),
            instructions="Copy account only",
        )
        await client.submit_extract(changed)
        assert json.loads(calls[-1].content)["options"][
            "target"
        ] == changed.options.target.model_dump(mode="json", exclude_none=True)
        assert (
            json.loads(calls[-1].content)["options"]["output_mode"]
            == "schema_constrained"
        )
        assert "INV-42" not in calls[-1].content.decode()

    assert_client(calls, value, original, storage)
    assert original.model_dump() == before


@pytest.mark.parametrize(
    "admission,malformed,exception",
    [(403, False, ServiceError), (200, True, ResponseSchemaMismatchError)],
)
def test_client_admission_and_result_schema_errors(admission, malformed, exception):
    tr, _ = transport(False, admission=admission, malformed=malformed)
    with DoclingServiceClient(url="https://service.example") as client:
        client._http_client.close()
        client._http_client = httpx.Client(transport=tr)
        with pytest.raises(exception):
            client.submit_extract(request()).result()


def test_service_contract_imports_with_slim_dependencies():
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'docling_parse', 'pypdfium2', 'qwen_vl_utils', 'jsonschema'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from docling.datamodel.service import ExtractDocumentsOptions, ExtractSourcesRequest, ExtractionDocumentResult, ExtractionTaskResult
from docling.service_client import DoclingServiceClient, AsyncDoclingServiceClient
from docling.datamodel.extraction import ExtractionItem
options = ExtractDocumentsOptions(target={'template': {'format': 'example_json', 'value': {'invoice': 'INV-42'}}})
assert options.target.template.value['invoice'] == 'INV-42'
assert ExtractionDocumentResult.model_fields['items'].annotation == list[ExtractionItem]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_storage_credentials_are_not_redacted_on_submission():
    from docling.datamodel.service.sources import GoogleDriveCredentials
    from docling.datamodel.service.targets import GoogleDriveTarget

    target = GoogleDriveTarget(
        path_id="folder-42",
        refresh_token="refresh-42",
        credentials=GoogleDriveCredentials(
            client_id="id",
            project_id="project",
            auth_uri="https://accounts.example/auth",
            token_uri="https://accounts.example/token",
            auth_provider_x509_cert_url="https://accounts.example/certs",
            client_secret="caller-secret",
            redirect_uris=["http://localhost"],
        ),
    )
    original = request().model_copy(update={"target": target})
    tr, calls = transport(True)
    with DoclingServiceClient(url="https://service.example") as client:
        client._http_client.close()
        client._http_client = httpx.Client(transport=tr)
        client.submit_extract(original)
    assert (
        json.loads(calls[0].content)["target"]["credentials"]["client_secret"]
        == "caller-secret"
    )
    assert (
        original.target.credentials.client_secret.get_secret_value() == "caller-secret"
    )
