from __future__ import annotations

import os

from docling.datamodel.base_models import OutputFormat
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


def run_convert(client: DoclingServiceClient, source: str) -> None:
    print("\n=== convert() (single source) ===")
    result = client.convert(source=source, options=create_conversion_options())
    print("status:", result.status.value)
    print("document name:", result.document.name)
    print("num pages in output:", len(result.document.pages))


def run_convert_all(client: DoclingServiceClient, sources: list[str]) -> None:
    print("\n=== convert_all() (multiple sources) ===")
    for idx, result in enumerate(
        client.convert_all(
            sources=sources,
            options=create_conversion_options(),
            max_concurrency=2,
            raises_on_error=False,
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


def main() -> None:
    sources = [
        os.environ.get(SAMPLE_SOURCE_ENV, DEFAULT_SAMPLE_SOURCE),
        os.environ.get(SECOND_SAMPLE_SOURCE_ENV, DEFAULT_SECOND_SAMPLE_SOURCE),
    ]

    with DoclingServiceClient(
        url=_service_url(),
        api_key=os.environ.get(SERVICE_API_KEY_ENV),
    ) as client:
        health = client.health()
        print("health:", health.status)

        try:
            version = client.version()
            print("version keys:", ", ".join(sorted(version.keys())[:5]), "...")
        except Exception as exc:
            print("version endpoint unavailable:", exc)

        run_convert(client, sources[0])
        run_convert_all(client, sources)


if __name__ == "__main__":
    main()
