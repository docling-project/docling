from __future__ import annotations

import os

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.responses import TaskStatusResponse
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


def run_json_task_flow_with_watch() -> None:
    print("\n=== JSON task flow: submit -> watch -> result ===")
    with _client() as client:
        job: ConversionJob[ConversionResult] = client.submit(
            source=SAMPLE_SOURCE,
            options=create_conversion_options(),
            target_format=OutputFormat.JSON,
        )
        print("submit() returns ConversionJob[ConversionResult]")
        print("job fields:", "task_id,", "status,", "queue_position,", "submitted_at")
        print("submit() returned task id:", job.task_id)
        print(
            "watch(timeout=...) blocks and yields updates until task status is terminal."
        )
        print(
            "watch() yields TaskStatusResponse(task_id, task_type, task_status, "
            "task_position, task_meta, error_message)"
        )

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

        print(
            "result(timeout=...) fetches the final payload; after terminal watch, "
            "this call should return immediately."
        )
        result: ConversionResult = job.result(timeout=300.0)
        print("result() return type (JSON target): ConversionResult")
        print("result fields:", "status,", "document,", "errors,", "timings")
        print("json result status:", result.status.value)
        print("json document name:", result.document.name)


def run_markdown_task_flow_without_watch() -> None:
    print("\n=== Raw task flow: submit(...).result(...) ===")
    with _client() as client:
        print(
            "watch() is optional. result(timeout=...) waits until success/failure "
            "or timeout."
        )
        job: ConversionJob[RawServiceResult] = client.submit(
            source=SAMPLE_SOURCE,
            options=create_conversion_options(),
            target_format=OutputFormat.MARKDOWN,
        )
        print("submit() returns ConversionJob[RawServiceResult] for MARKDOWN target")
        raw_result: RawServiceResult = job.result(timeout=300.0)
        print("result() return type (MARKDOWN target): RawServiceResult")
        print("result fields:", "content,", "content_type,", "filename")

        print("raw content-type:", raw_result.content_type)
        print("raw payload bytes:", len(raw_result.content))


def main() -> None:
    run_json_task_flow_with_watch()
    run_markdown_task_flow_without_watch()


if __name__ == "__main__":
    main()
