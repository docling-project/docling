import pytest
from pydantic import TypeAdapter, ValidationError

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.requests import (
    AnyHttpSourceRequest,
    BatchConvertSourcesRequest,
    ConvertSourcesRequest,
    HttpSourceRequest,
    TargetRequest,
)
from docling.datamodel.service.responses import (
    ArtifactRef,
    ConvertDocumentResponse,
    DoclingTaskResult,
    DocumentArtifactItem,
    DocumentResultItem,
    ExportDocumentResponse,
    ExportResult,
    PresignedArtifactResult,
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
)
from docling.datamodel.service.targets import PresignedUrlTarget


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


def test_batch_convert_sources_request_accepts_http_and_s3_sources() -> None:
    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [
                {"kind": "http", "url": "https://example.com/report.pdf"},
                {
                    "kind": "s3",
                    "endpoint": "s3.example.com",
                    "access_key": "key",
                    "secret_key": "secret",
                    "bucket": "documents",
                },
            ],
            "target": {
                "kind": "s3",
                "endpoint": "s3.example.com",
                "access_key": "key",
                "secret_key": "secret",
                "bucket": "converted",
            },
        }
    )

    assert len(request.sources) == 2


def test_batch_convert_sources_request_allows_zip_http_urls() -> None:
    request = BatchConvertSourcesRequest.model_validate(
        {
            "sources": [{"kind": "http", "url": "https://example.com/report.zip"}],
            "target": {"kind": "presigned_url"},
        }
    )

    assert str(request.sources[0].url) == "https://example.com/report.zip"


def test_batch_convert_sources_request_rejects_file_sources() -> None:
    with pytest.raises(ValidationError):
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


def test_target_request_accepts_presigned_url_target() -> None:
    parsed = TypeAdapter(TargetRequest).validate_python({"kind": "presigned_url"})

    assert isinstance(parsed, PresignedUrlTarget)


def test_document_result_item_maps_to_existing_wire_models() -> None:
    item = DocumentResultItem(
        document=ExportDocumentResponse(filename="example.pdf", md_content="# hello"),
        status=ConversionStatus.SUCCESS,
    )

    assert item.model_dump(mode="json") == ExportResult(
        content=item.document,
        status=item.status,
        errors=item.errors,
        timings=item.timings,
    ).model_dump(mode="json")
    assert ConvertDocumentResponse(
        document=item.document,
        status=item.status,
        errors=item.errors,
        processing_time=1.25,
        timings=item.timings,
    ).model_dump(mode="json") == {
        "document": item.document.model_dump(mode="json"),
        "status": "success",
        "errors": [],
        "processing_time": 1.25,
        "timings": {},
    }


def test_document_result_item_accepts_legacy_content_field() -> None:
    item = DocumentResultItem.model_validate(
        {
            "kind": "ExportResult",
            "content": {"filename": "example.pdf", "md_content": "# hello"},
            "status": ConversionStatus.SUCCESS,
        }
    )

    assert item.document.filename == "example.pdf"
    assert item.model_dump(mode="json")["content"]["filename"] == "example.pdf"


def test_document_result_item_content_property_warns() -> None:
    item = DocumentResultItem(
        document=ExportDocumentResponse(filename="example.pdf"),
        status=ConversionStatus.SUCCESS,
    )

    with pytest.warns(DeprecationWarning, match="use \\.document instead"):
        content = item.content

    assert content.filename == "example.pdf"


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
        num_failed=0,
    )

    assert result.result.kind == "PresignedArtifactResult"


def test_presigned_url_convert_response_includes_partial_success_counts() -> None:
    item = DocumentArtifactItem(
        source_index=0,
        source_uri="https://example.com/input.pdf",
        filename="input.pdf",
        status=ConversionStatus.SUCCESS,
        artifacts=[
            ArtifactRef(
                artifact_type="markdown",
                mime_type="text/markdown",
                uri="s3://converted/000000-input/input.md",
            )
        ],
    )
    result = DoclingTaskResult(
        result=PresignedArtifactResult(documents=[item]),
        processing_time=0.5,
        num_converted=1,
        num_succeeded=1,
        num_partial_success=0,
        num_failed=0,
    )
    response = PresignedUrlConvertResponse(
        documents=result.result.documents,
        processing_time=result.processing_time,
        num_converted=result.num_converted,
        num_succeeded=result.num_succeeded,
        num_partial_success=result.num_partial_success,
        num_failed=result.num_failed,
    )

    assert result.result.kind == "PresignedArtifactResult"
    assert response.documents[0].source_index == 0


def test_presigned_url_convert_document_response_defaults_partial_success_to_zero() -> (
    None
):
    response = PresignedUrlConvertDocumentResponse(
        processing_time=0.5,
        num_converted=2,
        num_succeeded=1,
        num_failed=1,
    )

    assert response.num_partial_success == 0
