from __future__ import annotations

import os

from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.service_client import DoclingServiceClient

SERVICE_URL_ENV = "DOCLING_SERVICE_URL"
SERVICE_API_KEY_ENV = "DOCLING_SERVICE_API_KEY"
SAMPLE_SOURCE_ENV = "DOCLING_SAMPLE_SOURCE"
SECOND_SAMPLE_SOURCE_ENV = "DOCLING_SECOND_SAMPLE_SOURCE"

DEFAULT_SAMPLE_SOURCE = "https://arxiv.org/pdf/2206.01062"
DEFAULT_SECOND_SAMPLE_SOURCE = "https://arxiv.org/pdf/2305.03393"


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


def main() -> None:
    sources = [
        os.environ.get(SAMPLE_SOURCE_ENV, DEFAULT_SAMPLE_SOURCE),
        os.environ.get(SECOND_SAMPLE_SOURCE_ENV, DEFAULT_SECOND_SAMPLE_SOURCE),
    ]

    with DoclingServiceClient(
        url=_service_url(),
        api_key=os.environ.get(SERVICE_API_KEY_ENV),
    ) as client:
        options = create_conversion_options()
        print("batch results:", len(sources))
        chunked_count = 0
        for idx, (source, result) in enumerate(
            zip(
                sources,
                client.convert_all(
                    sources=sources,
                    options=options,
                    max_concurrency=2,
                    raises_on_error=False,
                ),
            ),
            start=1,
        ):
            print(
                f"{idx}.",
                "input=",
                result.input.file.name,
                "status=",
                result.status.value,
            )

            if result.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                print("skip chunking due to failed conversion:", source)
                continue

            chunk_response = client.chunk(
                source=source,
                chunker="hierarchical",
                options=options,
            )
            chunked_count += 1
            print("chunked source:", source)
            print("num chunks:", len(chunk_response.chunks))
            print("num documents:", len(chunk_response.documents))

        print("sources chunked:", chunked_count)


if __name__ == "__main__":
    main()
