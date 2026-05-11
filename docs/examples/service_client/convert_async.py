# NOTE: docling.service_client is experimental and may change in future releases.
"""Async client examples: convert / submit / chunk against a running docling-serve."""

from __future__ import annotations

import asyncio
import os

from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.service_client import (
    AsyncConversionJob,
    AsyncDoclingServiceClient,
    ChunkerKind,
    RawServiceResult,
)

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


async def run_convert_single(client: AsyncDoclingServiceClient) -> None:
    print("\n=== await client.convert(source) ===")
    result = await client.convert(
        source=SAMPLE_SOURCES[0],
        options=create_conversion_options(),
    )
    print("status:", result.status.value)
    print("document name:", result.document.name)
    print("num pages in output:", len(result.document.pages))


async def run_convert_all(client: AsyncDoclingServiceClient) -> None:
    print("\n=== async for r in client.convert_all(sources) ===")
    idx = 0
    async for result in client.convert_all(
        sources=SAMPLE_SOURCES,
        options=create_conversion_options(),
        max_concurrency=2,
    ):
        idx += 1
        print(
            f"{idx}.",
            "input=",
            result.input.file.name,
            "status=",
            result.status.value,
        )


async def run_task_api_json(client: AsyncDoclingServiceClient) -> None:
    print("\n=== submit -> watch -> result (JSON target) ===")
    job: AsyncConversionJob[ConversionResult] = await client.submit(
        source=SAMPLE_SOURCES[0],
        options=create_conversion_options(),
        target_format=OutputFormat.JSON,
    )
    print("submitted task id:", job.task_id)
    async for update in job.watch(timeout=300.0):
        print(
            "status update:",
            update.task_status,
            "queue_position=",
            update.task_position,
        )
    # After a terminal watch, result(...) returns immediately.
    result = await job.result(timeout=300.0)
    print("final status:", result.status.value)
    print("document name:", result.document.name)


async def run_task_api_markdown(client: AsyncDoclingServiceClient) -> None:
    print("\n=== submit(...).result() (MARKDOWN target) ===")
    job: AsyncConversionJob[RawServiceResult] = await client.submit(
        source=SAMPLE_SOURCES[0],
        options=create_conversion_options(),
        target_format=OutputFormat.MARKDOWN,
    )
    raw_result = await job.result(timeout=300.0)
    print("raw content-type:", raw_result.content_type)
    print("raw payload bytes:", len(raw_result.content))


async def run_batch_and_chunk(client: AsyncDoclingServiceClient) -> None:
    print("\n=== convert_all() + chunk() (async) ===")
    options = create_conversion_options()
    chunked_count = 0
    sources_iter = iter(SAMPLE_SOURCES)
    async for result in client.convert_all(
        sources=SAMPLE_SOURCES,
        options=options,
        max_concurrency=2,
    ):
        source = next(sources_iter)
        print(
            "input=",
            result.input.file.name,
            "status=",
            result.status.value,
        )
        if result.status not in {
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        }:
            continue
        chunk_response = await client.chunk(
            source=source,
            chunker=ChunkerKind.HIERARCHICAL,
            options=options,
        )
        chunked_count += 1
        print("num chunks:", len(chunk_response.chunks))
    print("sources chunked:", chunked_count)


async def main() -> None:
    async with AsyncDoclingServiceClient(
        url=_service_url(),
        api_key=os.environ.get(SERVICE_API_KEY_ENV, ""),
    ) as client:
        health = await client.health()
        print("health:", health.status)

        await run_convert_single(client)
        await run_convert_all(client)
        await run_task_api_json(client)
        await run_task_api_markdown(client)
        await run_batch_and_chunk(client)


if __name__ == "__main__":
    asyncio.run(main())
