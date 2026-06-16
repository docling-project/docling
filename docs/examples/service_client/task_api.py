from __future__ import annotations

import os

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.responses import (
    PresignedUrlConvertResponse,
    TaskStatusResponse,
)
from docling.datamodel.service.targets import (
    InBodyTarget,
    PresignedUrlTarget,
    ZipTarget,
)
from docling.service_client import ConversionJob, DoclingServiceClient, RawServiceResult

SERVICE_URL_ENV = "DOCLING_SERVICE_URL"
SERVICE_API_KEY_ENV = "DOCLING_SERVICE_API_KEY"
SAMPLE_SOURCE = "https://arxiv.org/pdf/2206.01062"


def _service_url() -> str:
    service_url = os.environ.get(SERVICE_URL_ENV)
    if not service_url:
        raise SystemExit(f"Set {SERVICE_URL_ENV} to a running docling-serve URL.")
    return service_url


def create_conversion_options() -> ConvertDocumentsRequestOptions:
    return ConvertDocumentsRequestOptions(
        do_ocr=False,
        do_table_structure=False,
        include_images=False,
        to_formats=[OutputFormat.JSON],
        abort_on_error=False,
    )


def _client() -> DoclingServiceClient:
    return DoclingServiceClient(
        url=_service_url(),
        api_key=os.environ.get(SERVICE_API_KEY_ENV),
    )


def run_presigned_task_flow_with_watch() -> None:
    print("\n=== Presigned artifact task flow: submit -> watch -> result ===")
    with _client() as client:
        job: ConversionJob[PresignedUrlConvertResponse] = client.submit(
            source=SAMPLE_SOURCE,
            options=create_conversion_options(),
            output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
            target=PresignedUrlTarget(),
        )
        print("submit() returned task id:", job.task_id)
        print("job fields:", "task_id,", "status,", "queue_position,", "submitted_at")

        for update in job.watch(timeout=300.0):
            status_update: TaskStatusResponse = update
            print(
                "status update:",
                status_update.task_status,
                "queue_position=",
                status_update.task_position,
                "task_type=",
                status_update.task_type,
            )

        result = job.result(timeout=300.0)
        print("result() return type: PresignedUrlConvertResponse")
        print(
            "task counts:",
            result.num_succeeded,
            "succeeded /",
            result.num_failed,
            "failed",
        )
        for document in result.documents:
            print("document:", document.filename, "status=", document.status.value)
            for artifact in document.artifacts:
                print("artifact:", artifact.artifact_type, str(artifact.uri))


def run_json_task_flow_without_storage() -> None:
    print("\n=== Inline JSON task flow: explicit InBodyTarget ===")
    with _client() as client:
        job: ConversionJob[ConversionResult] = client.submit(
            source=SAMPLE_SOURCE,
            options=create_conversion_options(),
            target=InBodyTarget(),
        )
        result = job.result(timeout=300.0)
        print("result() return type: ConversionResult")
        print("json result status:", result.status.value)
        print("json document name:", result.document.name)


def run_zip_task_flow_without_watch() -> None:
    print("\n=== ZIP task flow: submit(...).result(...) ===")
    with _client() as client:
        job: ConversionJob[RawServiceResult] = client.submit(
            source=SAMPLE_SOURCE,
            options=create_conversion_options(),
            output_formats=[OutputFormat.MARKDOWN],
            target=ZipTarget(),
        )
        raw_result = job.result(timeout=300.0)
        print("result() return type: RawServiceResult")
        print("raw content-type:", raw_result.content_type)
        print("raw payload bytes:", len(raw_result.content))


def main() -> None:
    run_presigned_task_flow_with_watch()
    run_json_task_flow_without_storage()
    run_zip_task_flow_without_watch()


if __name__ == "__main__":
    main()
