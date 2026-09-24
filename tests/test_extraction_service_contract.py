# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Offline SDK/wire and source extraction client contracts."""

import base64
import json
import subprocess
import sys
from copy import deepcopy

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    InputFormat,
    VlmStopReason,
)
from docling.datamodel.extraction import (
    DocumentExtractionResult,
    ExtractionItem,
    ExtractionTarget,
    ExtractionTemplate,
    PageScope,
    VlmInferenceMetadata,
)
from docling.datamodel.extraction_options import GRANITE_VISION_4_1_SPEC
from docling.datamodel.service import ExtractionDocumentResult, ExtractionTaskResult
from docling.datamodel.service.options import ExtractDocumentsOptions
from docling.datamodel.service.requests import ExtractSourcesRequest, S3SourceRequest
from docling.datamodel.service.responses import (
    DoclingTaskResult,
    ExtractDocumentResponse,
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
)
from docling.models.extraction.prompt_utils import prepare_target
from docling.service_client import AsyncDoclingServiceClient, DoclingServiceClient
from docling.service_client.exceptions import (
    ExtractionError,
    ResponseSchemaMismatchError,
    ServiceError,
)


class Invoice(BaseModel):
    invoice: str
    total: float


def request(*, storage=False):
    return ExtractSourcesRequest(
        extraction_target=ExtractionTarget.from_pydantic(
            Invoice,
            template=ExtractionTemplate(
                format="example_json", value={"invoice": "INV-42", "total": 4.2}
            ),
            instructions="Copy the invoice identifier exactly",
        ),
        options=ExtractDocumentsOptions(
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


def _item_error(message, page_no=None):
    return ErrorItem(
        component_type=DoclingComponentType.MODEL,
        module_name="ExtractionVlmPipeline",
        error_message=message,
        category=FailureCategory.INFERENCE_FAILURE,
        page_no=page_no,
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
                        inference_metadata=VlmInferenceMetadata(
                            generation_time=0.4,
                            num_tokens=12,
                            usage={"completion_tokens": 12},
                            stop_reason=VlmStopReason.END_OF_SEQUENCE,
                        ),
                    ),
                    ExtractionItem(
                        scope={"kind": "document"},
                        raw_text='{"total":"bad"}',
                        validation_status="failed",
                        errors=[
                            _item_error("Schema validation at $.total: not a number")
                        ],
                    ),
                    ExtractionItem(
                        scope=PageScope(page_no=2),
                        validation_status="not_run",
                        errors=[_item_error("Inference failed", page_no=2)],
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
    sdk = request().extraction_target
    wire = ExtractSourcesRequest.model_validate_json(
        request().model_dump_json()
    ).extraction_target
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
                    "documents": [],
                }
                if storage
                else response().model_dump(mode="json")
            ),
        )

    return httpx.MockTransport(handle), calls


def _submit(client, req):
    # Unpack a prebuilt request into the friendly submit_extract signature.
    return client.submit_extract(
        source=list(req.sources),
        extraction_target=req.extraction_target,
        options=req.options,
        target=req.target,
        callbacks=req.callbacks,
    )


def _asubmit(client, req):
    return client.submit_extract(
        source=list(req.sources),
        extraction_target=req.extraction_target,
        options=req.options,
        target=req.target,
        callbacks=req.callbacks,
    )


def assert_client(calls, value, original, storage):
    assert calls[0].url.path == "/v1/extract/source/async"
    received = json.loads(calls[0].content)
    assert received["extraction_target"] == original.extraction_target.model_dump(
        mode="json", exclude_none=True
    )
    assert received["target"]["kind"] == ("presigned_url" if storage else "inbody")
    assert received["options"]["page_range"] == [2, 3]
    assert received["sources"][0]["headers"]["Authorization"] == "secret"
    if storage:
        assert isinstance(value, PresignedUrlConvertResponse)
        assert value.num_succeeded == 1
    else:
        assert value == response()


@pytest.mark.parametrize("storage", [False, True])
def test_sync_extraction_payload_and_result(storage):
    tr, calls = transport(storage)
    original = request(storage=storage)
    before = original.model_dump()
    with DoclingServiceClient(url="https://service.example") as client:
        client._http_client.close()
        client._http_client = httpx.Client(transport=tr)
        value = _submit(client, original).result()
        changed = original.model_copy(deep=True)
        changed.options.output_mode = "schema_constrained"
        changed.extraction_target = ExtractionTarget(
            output_schema={
                "type": "object",
                "properties": {"account": {"type": "integer"}},
            },
            template=ExtractionTemplate(format="example_json", value={"account": 17}),
            instructions="Copy account only",
        )
        _submit(client, changed)
        assert json.loads(calls[-1].content)[
            "extraction_target"
        ] == changed.extraction_target.model_dump(mode="json", exclude_none=True)
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
        value = await (await _asubmit(client, original)).result()
        changed = original.model_copy(deep=True)
        changed.options.output_mode = "schema_constrained"
        changed.extraction_target = ExtractionTarget(
            output_schema={
                "type": "object",
                "properties": {"account": {"type": "integer"}},
            },
            template=ExtractionTemplate(format="example_json", value={"account": 17}),
            instructions="Copy account only",
        )
        await _asubmit(client, changed)
        assert json.loads(calls[-1].content)[
            "extraction_target"
        ] == changed.extraction_target.model_dump(mode="json", exclude_none=True)
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
            _submit(client, request()).result()


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
req = ExtractSourcesRequest(extraction_target={'template': {'format': 'example_json', 'value': {'invoice': 'INV-42'}}}, sources=[{'kind': 'http', 'url': 'https://example.com/r.pdf'}])
assert req.extraction_target.template.value['invoice'] == 'INV-42'
assert req.options == ExtractDocumentsOptions()
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
        value = _submit(client, original).result()
    assert type(value) is PresignedUrlConvertDocumentResponse
    assert (
        json.loads(calls[0].content)["target"]["credentials"]["client_secret"]
        == "caller-secret"
    )
    assert (
        original.target.credentials.client_secret.get_secret_value() == "caller-secret"
    )


def _doc(status=ConversionStatus.SUCCESS, source_index=0, filename="report.pdf"):
    return ExtractionDocumentResult(
        source_index=source_index,
        source_uri=f"file://{filename}",
        filename=filename,
        status=status,
        items=[],
    )


def _result_transport(documents):
    calls = []

    def handle(req):
        calls.append(req)
        if req.method == "POST":
            return httpx.Response(
                200,
                json={
                    "task_id": "extract-1",
                    "task_type": "extract",
                    "task_status": "success",
                },
            )
        payload = ExtractDocumentResponse(
            num_converted=len(documents),
            num_succeeded=sum(
                d.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS)
                for d in documents
            ),
            num_failed=sum(d.status == ConversionStatus.FAILURE for d in documents),
            processing_time=0.1,
            documents=documents,
        )
        return httpx.Response(200, json=payload.model_dump(mode="json"))

    return httpx.MockTransport(handle), calls


def _extract_client(documents):
    tr, calls = _result_transport(documents)
    client = DoclingServiceClient(url="https://service.example")
    client._http_client.close()
    client._http_client = httpx.Client(transport=tr)
    return client, calls


TARGET = ExtractionTarget(
    template=ExtractionTemplate(format="example_json", value={"invoice": "INV-1"})
)


def test_extract_returns_single_document_and_uploads_file_inline(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    client, calls = _extract_client([_doc()])
    with client:
        document = client.extract(pdf, target=TARGET, page_range=(2, 3))
    # Same result type as the local DocumentExtractor.
    assert isinstance(document, DocumentExtractionResult)
    assert document.input.file.name == "report.pdf"
    assert document.input.format == InputFormat.PDF
    assert json.loads(calls[0].content)["options"]["page_range"] == [2, 3]
    # Local files ride inline as a base64 file source, not multipart.
    source = json.loads(calls[0].content)["sources"][0]
    assert source["kind"] == "file"
    assert base64.b64decode(source["base64_string"]) == b"%PDF-1.4 fake"
    assert json.loads(calls[0].content)["target"]["kind"] == "inbody"


def test_extract_raises_when_source_expands_server_side():
    client, _ = _extract_client([_doc(source_index=0), _doc(source_index=1)])
    with client, pytest.raises(ExtractionError, match="expanded to 2"):
        client.extract("https://example.com/report.pdf", TARGET)


@pytest.mark.parametrize(
    "source",
    [
        ["https://example.com/a.pdf"],
        S3SourceRequest(
            endpoint="s3.example.com",
            access_key="key",
            secret_key="secret",
            bucket="docs",
        ),
    ],
)
def test_extract_rejects_expandable_sources_before_submission(source):
    client, calls = _extract_client([_doc()])
    with client, pytest.raises(TypeError, match="extract_all"):
        client.extract(source, TARGET)
    assert calls == []


S3_DICT = {
    "kind": "s3",
    "endpoint": "s3.example.com",
    "access_key": "key",
    "secret_key": "secret",
    "bucket": "docs",
}


@pytest.mark.parametrize("dict_sources", [S3_DICT, [S3_DICT, S3_DICT]])
def test_submit_extract_accepts_dict_sources_like_submit_batch(dict_sources):
    client, calls = _extract_client([_doc()])
    with client:
        client.submit_extract(dict_sources, TARGET)
    sent = json.loads(calls[0].content)["sources"]
    expected = dict_sources if isinstance(dict_sources, list) else [dict_sources]
    assert [(item["kind"], item["bucket"]) for item in sent] == [
        ("s3", "docs") for _ in expected
    ]
    assert sent[0]["secret_key"] == "secret"


def test_zip_sources_are_rejected_before_submission(tmp_path):
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(b"PK\x03\x04")
    client, calls = _extract_client([_doc()])
    with client:
        for source in ("https://example.com/bundle.zip", archive):
            with pytest.raises(ValidationError, match="ZIP archives"):
                client.submit_extract(source, TARGET)
        # convert() shares the URL path; it used to fail as a missing local file.
        with pytest.raises(ValidationError, match="ZIP archives"):
            client.convert("https://example.com/bundle.zip")
    assert calls == []


def test_extract_failure_status_respects_raises_on_error():
    client, _ = _extract_client([_doc(status=ConversionStatus.FAILURE)])
    with client:
        with pytest.raises(ExtractionError):
            client.extract("https://example.com/report.pdf", TARGET)
    client, _ = _extract_client([_doc(status=ConversionStatus.FAILURE)])
    with client:
        document = client.extract(
            "https://example.com/report.pdf", TARGET, raises_on_error=False
        )
    assert document.status == ConversionStatus.FAILURE


def test_extract_all_runs_one_job_per_source(monkeypatch):
    # Serve rejects more than 3 sources per request by default; one source is
    # rejected on its own to check that a failed job does not end the iterator.
    calls = []

    def handle(req):
        calls.append(req)
        if req.method == "POST":
            sources = json.loads(req.content)["sources"]
            if len(sources) > 3 or sources[0]["url"].endswith("bad.pdf"):
                return httpx.Response(422, json={"detail": "rejected"})
            return httpx.Response(
                200,
                json={
                    "task_id": sources[0]["url"].rsplit("/", 1)[-1],
                    "task_type": "extract",
                    "task_status": "success",
                },
            )
        name = req.url.path.rsplit("/", 1)[-1]
        payload = ExtractDocumentResponse(
            num_converted=1,
            num_succeeded=1,
            num_failed=0,
            processing_time=0.1,
            documents=[_doc(filename=name)],
        )
        return httpx.Response(200, json=payload.model_dump(mode="json"))

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(
            transport=httpx.MockTransport(handle), **kwargs
        ),
    )
    names = ["a.pdf", "b.pdf", "bad.pdf", "c.pdf", "d.pdf"]
    with DoclingServiceClient(url="https://service.example") as client:
        results = list(
            client.extract_all(
                [f"https://example.com/{name}" for name in names],
                TARGET,
                max_concurrency=2,
            )
        )

    assert len([c for c in calls if c.method == "POST"]) == 5
    by_name = {d.input.file.name: d for d in results}
    assert sorted(by_name) == sorted(names)
    assert by_name["bad.pdf"].status == ConversionStatus.FAILURE
    assert all(
        by_name[n].status == ConversionStatus.SUCCESS for n in names if n != "bad.pdf"
    )


@pytest.mark.anyio
async def test_async_extract_and_extract_all():
    tr, _ = _result_transport([_doc(filename="b.pdf")])
    async with AsyncDoclingServiceClient(url="https://service.example") as client:
        await client._async_client.aclose()
        client._async_client = httpx.AsyncClient(transport=tr)
        document = await client.extract("https://example.com/b.pdf", TARGET)
        assert document.input.file.name == "b.pdf"
        results = [
            d async for d in client.extract_all(["https://example.com/b.pdf"], TARGET)
        ]
    assert [d.input.file.name for d in results] == ["b.pdf"]


@pytest.mark.anyio
async def test_oversized_local_file_is_skipped_before_upload(tmp_path):
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4" + b"0" * 100)
    tr, calls = _result_transport([_doc(filename="b.pdf")])
    async with AsyncDoclingServiceClient(url="https://service.example") as client:
        await client._async_client.aclose()
        client._async_client = httpx.AsyncClient(transport=tr)
        with pytest.raises(ExtractionError, match="max_file_size"):
            await client.extract(big, TARGET, max_file_size=10)
        results = [
            d
            async for d in client.extract_all(
                [big, "https://example.com/b.pdf"], TARGET, max_file_size=10
            )
        ]
    assert {d.input.file.name: d.status for d in results} == {
        "big.pdf": ConversionStatus.SKIPPED,
        "b.pdf": ConversionStatus.SUCCESS,
    }
    # Only the URL source was submitted; the big file was never uploaded.
    assert [
        json.loads(c.content)["sources"][0]["kind"] for c in calls if c.method == "POST"
    ] == ["http"]
