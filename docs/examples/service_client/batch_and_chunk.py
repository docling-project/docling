from __future__ import annotations

import os

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import AnyHttpSourceRequest
from docling.datamodel.service.targets import PresignedUrlTarget
from docling.service_client import ChunkerKind, DoclingServiceClient

SERVICE_URL_ENV = "DOCLING_SERVICE_URL"
SERVICE_API_KEY_ENV = "DOCLING_SERVICE_API_KEY"
SAMPLE_SOURCES = [
    "https://arxiv.org/pdf/2206.01062",
    "https://arxiv.org/pdf/2305.03393",
]


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


def run_submit_batch(client: DoclingServiceClient) -> None:
    print("\n=== submit_batch(): HTTP sources -> PresignedUrlTarget ===")
    job = client.submit_batch(
        sources=[AnyHttpSourceRequest(url=source) for source in SAMPLE_SOURCES],
        target=PresignedUrlTarget(),
        output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
        options=create_conversion_options(),
    )
    result = job.result(timeout=300.0)
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


def run_chunk(client: DoclingServiceClient) -> None:
    print("\n=== chunk(): single HTTP source -> hierarchical chunks ===")
    chunk_response = client.chunk(
        source=SAMPLE_SOURCES[0],
        chunker=ChunkerKind.HIERARCHICAL,
        options=create_conversion_options(),
    )
    print("num chunks:", len(chunk_response.chunks))
    print("num documents:", len(chunk_response.documents))


def main() -> None:
    with DoclingServiceClient(
        url=_service_url(),
        api_key=os.environ.get(SERVICE_API_KEY_ENV),
    ) as client:
        run_submit_batch(client)
        run_chunk(client)


if __name__ == "__main__":
    main()
