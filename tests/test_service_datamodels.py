# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from pydantic import ValidationError

from docling.datamodel.base_models import (
    ConfidenceReport,
    ConversionStatus,
    PageConfidenceScores,
    QualityGrade,
)
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    AzureBlobSourceRequest,
    BatchConvertSourcesRequest,
    ConvertSourcesRequest,
    GenericSourceRequest,
    GenericTargetRequest,
    GoogleCloudStorageSourceRequest,
    GoogleDriveSourceRequest,
    HttpSourceRequest,
    S3SourceRequest,
)
from docling.datamodel.service.responses import (
    ArtifactRef,
    ConfidenceScores,
    DoclingTaskResult,
    DocumentArtifactItem,
    FailureCategory,
    FailurePhase,
    PresignedArtifactResult,
    PublicFailureInfo,
    TaskFailureResult,
    TaskStatusResponse,
    ZipArchiveResult,
)
from docling.datamodel.service.sources import S3Coordinates
from docling.datamodel.service.targets import (
    AzureBlobTarget,
    GoogleCloudStorageTarget,
    GoogleDriveTarget,
    PresignedUrlTarget,
    S3Target,
    ZipTarget,
)

S3_MODEL_TYPES = (S3Coordinates, S3SourceRequest, S3Target)
FAKE_S3_CREDENTIALS = {
    "access_key": "fake-access-key-for-testing-only",
    "secret_key": "fake-secret-key-for-testing-only",
}


def test_s3_source_and_target_inherit_coordinates() -> None:
    assert issubclass(S3SourceRequest, S3Coordinates)
    assert issubclass(S3Target, S3Coordinates)


@pytest.mark.parametrize("model_type", S3_MODEL_TYPES)
def test_s3_coordinates_allow_ambient_credentials(
    model_type: type[S3Coordinates],
) -> None:
    coordinates = model_type(endpoint="s3.example.com", bucket="documents")

    assert coordinates.access_key is None
    assert coordinates.secret_key is None


@pytest.mark.parametrize("model_type", S3_MODEL_TYPES)
def test_s3_coordinates_preserve_explicit_string_credentials(
    model_type: type[S3Coordinates],
) -> None:
    coordinates = model_type(
        endpoint="s3.example.com",
        bucket="documents",
        **FAKE_S3_CREDENTIALS,
    )

    assert coordinates.access_key == FAKE_S3_CREDENTIALS["access_key"]
    assert coordinates.secret_key == FAKE_S3_CREDENTIALS["secret_key"]
    assert isinstance(coordinates.access_key, str)
    assert isinstance(coordinates.secret_key, str)


@pytest.mark.parametrize("model_type", S3_MODEL_TYPES)
@pytest.mark.parametrize(
    "credentials",
    [
        {"access_key": "fake-access-key-for-testing-only"},
        {"secret_key": "fake-secret-key-for-testing-only"},
    ],
)
def test_s3_coordinates_reject_incomplete_explicit_credentials(
    model_type: type[S3Coordinates], credentials: dict[str, str]
) -> None:
    with pytest.raises(ValidationError, match="access_key and secret_key"):
        model_type(endpoint="s3.example.com", bucket="documents", **credentials)


@pytest.mark.parametrize("model_type", S3_MODEL_TYPES)
def test_s3_coordinates_credential_schema_is_optional_string(
    model_type: type[S3Coordinates],
) -> None:
    schema = model_type.model_json_schema()

    assert not {"access_key", "secret_key"} & set(schema["required"])
    for field_name in ("access_key", "secret_key"):
        field_schema = schema["properties"][field_name]
        assert {item.get("type") for item in field_schema["anyOf"]} == {
            "null",
            "string",
        }
        assert "Optional" in field_schema["description"]


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


def test_batch_convert_sources_request_preserves_generic_target() -> None:
    target = {
        "kind": "plugin_artifact_store",
        "bucket": "out",
        "api_key": "secret",
    }

    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [{"kind": "http", "url": "https://example.com/report.pdf"}],
            "target": target,
        }
    )

    assert isinstance(request.target, GenericTargetRequest)
    assert request.model_dump(mode="json")["target"] == target


@pytest.mark.parametrize("target", [{}, {"kind": ""}])
def test_batch_convert_sources_request_requires_non_empty_target_kind(
    target: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        BatchConvertSourcesRequest.model_validate(
            {
                "sources": [{"kind": "http", "url": "https://example.com/report.pdf"}],
                "target": target,
            }
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
def test_batch_convert_sources_request_keeps_known_target_validation(
    kind: str,
    required_field: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        BatchConvertSourcesRequest.model_validate(
            {
                "sources": [{"kind": "http", "url": "https://example.com/report.pdf"}],
                "target": {"kind": kind},
            }
        )

    locations = {error["loc"] for error in exc_info.value.errors()}
    assert ("target", required_field) in locations
    assert all("GenericTargetRequest" not in location for location in locations)


@pytest.mark.parametrize(
    "target",
    [
        S3Target(
            endpoint="s3.example.com",
            access_key="key",
            secret_key="secret",
            bucket="documents",
        ),
        AzureBlobTarget(
            account_name="account",
            container="documents",
            connection_string="connection-string",
        ),
        GoogleCloudStorageTarget(bucket="documents"),
        GoogleDriveTarget(
            path_id="folder-id",
            token_path="token.json",
            credentials_path="credentials.json",
        ),
        PresignedUrlTarget(),
    ],
)
def test_batch_convert_sources_request_preserves_known_target_type(
    target: S3Target
    | AzureBlobTarget
    | GoogleCloudStorageTarget
    | GoogleDriveTarget
    | PresignedUrlTarget,
) -> None:
    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [{"kind": "http", "url": "https://example.com/report.pdf"}],
            "target": target.model_dump(mode="json"),
        }
    )

    assert type(request.target) is type(target)


@pytest.mark.parametrize("kind", ["inbody", "zip", "put"])
def test_batch_convert_sources_request_rejects_non_batch_target(kind: str) -> None:
    target = {"kind": kind, "url": "https://example.com/upload"}

    with pytest.raises(ValidationError):
        BatchConvertSourcesRequest.model_validate(
            {
                "sources": [{"kind": "http", "url": "https://example.com/report.pdf"}],
                "target": target,
            }
        )


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


def test_zip_archive_result_filename_is_optional_on_the_wire() -> None:
    # Results from producers that predate `filename` must still parse.
    legacy = DoclingTaskResult.model_validate(
        {
            "result": {"kind": "ZipArchiveResult", "content": ""},
            "processing_time": 1.0,
            "num_converted": 1,
            "num_succeeded": 1,
            "num_failed": 0,
        }
    )
    assert isinstance(legacy.result, ZipArchiveResult)
    assert legacy.result.filename is None

    named = DoclingTaskResult.model_validate_json(
        legacy.model_copy(
            update={"result": ZipArchiveResult(content=b"", filename="report.zip")}
        ).model_dump_json()
    )
    assert isinstance(named.result, ZipArchiveResult)
    assert named.result.filename == "report.zip"


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


def test_confidence_scores_from_flat_scores() -> None:
    scores = PageConfidenceScores(
        parse_score=0.9, layout_score=0.8, table_score=None, ocr_score=0.95
    )
    wire = ConfidenceScores.from_scores(scores)

    assert wire.parse_score == 0.9
    assert wire.layout_score == 0.8
    assert wire.table_score is None
    assert wire.ocr_score == 0.95
    assert wire.pages == {}

    # JSON serialization must emit an empty dict for pages, not null or a list
    dump = wire.model_dump(mode="json")
    assert dump["pages"] == {}

    # Deserializing back into ConfidenceReport succeeds
    report = ConfidenceReport.model_validate(dump)
    assert report.parse_score == 0.9
    assert report.pages == {}


def test_confidence_scores_from_report_with_pages() -> None:
    report = ConfidenceReport(
        pages={
            1: PageConfidenceScores(parse_score=0.9, layout_score=0.8),
            2: PageConfidenceScores(parse_score=0.7, layout_score=0.6),
        }
    )
    wire = ConfidenceScores.from_scores(report)

    assert 1 in wire.pages
    assert 2 in wire.pages
    assert wire.pages[1].parse_score == 0.9
    assert wire.pages[2].parse_score == 0.7

    dump = wire.model_dump(mode="json")
    assert "1" in dump["pages"]
    assert "2" in dump["pages"]

    restored = ConfidenceReport.model_validate(dump)
    assert len(restored.pages) == 2
    assert restored.pages[1].parse_score == 0.9
    assert restored.pages[2].parse_score == 0.7


def test_confidence_report_tolerant_to_pages_none() -> None:
    legacy_payload = {
        "parse_score": 0.85,
        "layout_score": 0.80,
        "table_score": None,
        "ocr_score": 0.90,
        "mean_score": 0.85,
        "low_score": 0.80,
        "mean_grade": QualityGrade.GOOD,
        "low_grade": QualityGrade.GOOD,
        "pages": None,
    }

    restored = ConfidenceReport.model_validate(legacy_payload)
    assert restored.parse_score == 0.85
    assert restored.pages == {}
