import pytest
from pydantic import ValidationError

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    BatchConvertSourcesRequest,
    ConvertSourcesRequest,
    HttpSourceRequest,
)
from docling.datamodel.service.responses import (
    ArtifactRef,
    DoclingTaskResult,
    DocumentArtifactItem,
    FailureCategory,
    FailurePhase,
    PresignedArtifactResult,
    PublicFailureInfo,
    TaskFailureResult,
    TaskStatusResponse,
)


def test_http_source_request_rejects_zip_urls() -> None:
    with pytest.raises(ValidationError, match="ZIP URLs are not accepted"):
        HttpSourceRequest(url="https://example.com/report.zip")


def test_any_http_source_request_allows_zip_urls() -> None:
    request = AnyHttpSourceRequest(url="https://example.com/report.zip")

    assert str(request.url) == "https://example.com/report.zip"


def test_convert_sources_request_rejects_s3_sources() -> None:
    with pytest.raises(ValidationError):
        ConvertSourcesRequest.model_validate(
            {
                "sources": [
                    {
                        "kind": "s3",
                        "endpoint": "s3.example.com",
                        "access_key": "key",
                        "secret_key": "secret",
                        "bucket": "documents",
                    }
                ]
            }
        )


def test_batch_convert_sources_request_allows_zip_http_urls() -> None:
    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [{"kind": "http", "url": "https://example.com/report.zip"}],
            "target": {"kind": "presigned_url"},
        }
    )

    assert str(request.sources[0].url) == "https://example.com/report.zip"


def test_batch_convert_sources_request_rejects_file_sources() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        BatchConvertSourcesRequest.model_validate(
            {
                "sources": [
                    {
                        "kind": "file",
                        "base64_string": "ZmFrZQ==",
                        "filename": "report.pdf",
                    }
                ],
                "target": {"kind": "presigned_url"},
            }
        )


def test_docling_task_result_accepts_presigned_artifact_results() -> None:
    result = DoclingTaskResult(
        result=PresignedArtifactResult(
            documents=[
                DocumentArtifactItem(
                    source_index=0,
                    source_uri="https://example.com/input.pdf",
                    filename="input.pdf",
                    status=ConversionStatus.SUCCESS,
                    artifacts=[
                        ArtifactRef(
                            artifact_type="markdown",
                            mime_type="text/markdown",
                            uri="s3://converted/input.md",
                        )
                    ],
                )
            ]
        ),
        processing_time=0.5,
        num_converted=1,
        num_succeeded=1,
        num_partially_succeeded=0,
        num_failed=0,
    )

    assert result.result.kind == "PresignedArtifactResult"


def test_task_failure_result_roundtrip() -> None:
    failure = PublicFailureInfo(
        category=FailureCategory.INTERNAL,
        message="Internal processing error.",
        retryable=False,
        phase=FailurePhase.ORCHESTRATION,
    )
    result = TaskFailureResult(failure=failure)

    restored = TaskFailureResult.model_validate_json(result.model_dump_json())

    assert restored.failure.category == FailureCategory.INTERNAL
    assert restored.failure.message == "Internal processing error."
    assert restored.failure.retryable is False
    assert restored.kind == "TaskFailureResult"


def test_task_status_response_accepts_structured_failure() -> None:
    response = TaskStatusResponse(
        task_id="task-1",
        task_type="convert",
        task_status="failure",
        error_message="Internal processing error.",
        failure=PublicFailureInfo(
            category=FailureCategory.INTERNAL,
            message="Internal processing error.",
            retryable=False,
            phase=FailurePhase.ORCHESTRATION,
        ),
    )

    assert response.failure is not None
    assert response.failure.phase == FailurePhase.ORCHESTRATION
