import pytest
from pydantic import ValidationError

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    AzureBlobSourceRequest,
    BatchConvertSourcesRequest,
    ConvertSourcesRequest,
    GenericSourceRequest,
    GoogleCloudStorageSourceRequest,
    GoogleDriveSourceRequest,
    HttpSourceRequest,
    S3SourceRequest,
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


def test_batch_convert_sources_request_preserves_generic_source() -> None:
    source = {
        "kind": "filenet",
        "base_url": "https://filenet.example.com/content-services-graphql",
        "username": "user",
        "api_key": "secret",
        "repository_id": "OS1",
        "folder_id": "/incoming",
    }

    request = BatchConvertSourcesRequest.model_validate(
        {"sources": [source], "target": {"kind": "presigned_url"}}
    )

    assert isinstance(request.sources[0], GenericSourceRequest)
    assert request.model_dump(mode="json")["sources"][0] == source


@pytest.mark.parametrize("source", [{}, {"kind": ""}])
def test_batch_convert_sources_request_requires_non_empty_kind(
    source: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        BatchConvertSourcesRequest.model_validate(
            {"sources": [source], "target": {"kind": "presigned_url"}}
        )


@pytest.mark.parametrize(
    ("kind", "required_field"),
    [
        ("s3", "endpoint"),
        ("azure_blob", "account_name"),
        ("google_cloud_storage", "bucket"),
        ("google_drive", "path_id"),
    ],
)
def test_batch_convert_sources_request_keeps_known_source_validation(
    kind: str,
    required_field: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        BatchConvertSourcesRequest.model_validate(
            {"sources": [{"kind": kind}], "target": {"kind": "presigned_url"}}
        )

    locations = {error["loc"] for error in exc_info.value.errors()}
    assert ("sources", 0, required_field) in locations
    assert all("GenericSourceRequest" not in location for location in locations)


@pytest.mark.parametrize(
    "source",
    [
        AnyHttpSourceRequest(url="https://example.com/report.pdf"),
        S3SourceRequest(
            endpoint="s3.example.com",
            access_key="key",
            secret_key="secret",
            bucket="documents",
        ),
        AzureBlobSourceRequest(
            account_name="account",
            container="documents",
            connection_string="connection-string",
        ),
        GoogleCloudStorageSourceRequest(bucket="documents"),
        GoogleDriveSourceRequest(
            path_id="folder-id",
            token_path="token.json",
            credentials_path="credentials.json",
        ),
    ],
)
def test_batch_convert_sources_request_preserves_known_source_type(
    source: AnyHttpSourceRequest
    | S3SourceRequest
    | AzureBlobSourceRequest
    | GoogleCloudStorageSourceRequest
    | GoogleDriveSourceRequest,
) -> None:
    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [source.model_dump(mode="json")],
            "target": {"kind": "presigned_url"},
        }
    )

    assert type(request.sources[0]) is type(source)


def test_batch_convert_sources_request_preserves_known_source_instance() -> None:
    source = HttpSourceRequest(url="https://example.com/report.pdf")

    request = BatchConvertSourcesRequest.model_validate(
        {"sources": [source], "target": {"kind": "presigned_url"}}
    )

    assert request.sources[0] is source


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
