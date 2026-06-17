"""Batch conversion with submit_batch() for many or long-running sources.

The batch endpoint accepts source *requests* (HTTP/ZIP URLs or S3 objects) rather
than uploaded files, and always needs an explicit batch target: `PresignedUrlTarget`
for service-managed download URLs, or `S3Target` for caller-managed object storage.

Run from the repository root:

    python docs/examples/service_client/batch.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from docling.datamodel.base_models import OutputFormat
from docling.service_client import (
    AnyHttpSourceRequest,
    DoclingServiceClient,
    PresignedUrlTarget,
)

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SOURCES = [
    "https://arxiv.org/pdf/2206.01062",
    "https://arxiv.org/pdf/2305.03393",
]


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        job = client.submit_batch(
            sources=[AnyHttpSourceRequest(url=url) for url in SOURCES],
            target=PresignedUrlTarget(),
            output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
        )
        result = job.result(timeout=300.0)
        print(result.num_succeeded, "succeeded /", result.num_failed, "failed")
        for document in result.documents:
            print(document.filename, document.status.value)
            for artifact in document.artifacts:
                print(" ", artifact.artifact_type, str(artifact.uri))

        # S3 fan-out: read inputs from a bucket and write results back to one.
        # Requires real bucket credentials, so it is left commented out.
        #
        # from docling.service_client import S3SourceRequest, S3Target
        # job = client.submit_batch(
        #     sources=[S3SourceRequest(bucket="in", key="docs/report.pdf", ...)],
        #     target=S3Target(bucket="out", prefix="results/", ...),
        #     output_formats=[OutputFormat.MARKDOWN],
        # )
        # result = job.result(timeout=600.0)


if __name__ == "__main__":
    main()
