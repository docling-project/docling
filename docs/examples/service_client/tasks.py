"""The submit API: raw service responses with explicit result targets.

Unlike `convert()`/`convert_all()` (which return a reconstructed
`ConversionResult`), the `submit*` family returns the raw service response and
lets you choose where results land:

- `submit()`            one job, with `watch()` -> `result()` lifecycle
- result targets        `InBodyTarget`, `PresignedUrlTarget`, `ZipTarget`
- `submit_and_retrieve_each()`  many items, one outcome each (errors inline)

Run from the repository root:

    python docs/examples/service_client/tasks.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.targets import ZipTarget
from docling.service_client import ConversionItem, DoclingServiceClient

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SOURCE = Path("tests/data/pdf/regression/2305.03393v1-pg9.pdf")
MANY = [
    Path("tests/data/pdf/regression/2305.03393v1-pg9.pdf"),
    Path("tests/data/pdf/regression/code_and_formula.pdf"),
    Path("tests/data/pdf/regression/picture_classification.pdf"),
]


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        # submit() -> watch() -> result(): track one job to completion.
        # Omitting `target` uses presigned artifacts, falling back to in-body.
        print("=== submit() -> watch() -> result() ===")
        job = client.submit(source=SOURCE, output_formats=[OutputFormat.MARKDOWN])
        print("task id:", job.task_id)
        for update in job.watch(timeout=300.0):
            print(" status:", update.task_status, "position:", update.task_position)
        result = job.result(timeout=300.0)
        print("done:", result.num_succeeded, "succeeded /", result.num_failed, "failed")

        # Explicit result targets. ZipTarget returns a raw archive of the
        # requested output formats; PresignedUrlTarget returns download URLs.
        # InBodyTarget() returns the document inline, where the service allows it
        # (some services restrict targets to storage-backed kinds).
        print("\n=== explicit targets ===")
        archive = client.submit(
            source=SOURCE,
            output_formats=[OutputFormat.MARKDOWN],
            target=ZipTarget(),
        ).result(timeout=300.0)
        print("ZipTarget:", archive.content_type, len(archive.content), "bytes")

        # submit_and_retrieve_each(): fan out many items, one outcome each.
        # Failures surface inline as the outcome instead of raising.
        print("\n=== submit_and_retrieve_each() ===")
        items = [
            ConversionItem(source=s, metadata={"id": i}) for i, s in enumerate(MANY)
        ]
        for item, outcome in client.submit_and_retrieve_each(items, max_in_flight=4):
            if isinstance(outcome, Exception):
                print(" ", item.metadata, "failed:", outcome)
            else:
                print(" ", item.metadata, "ok")


if __name__ == "__main__":
    main()
