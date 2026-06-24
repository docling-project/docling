"""High-level conversion with DoclingServiceClient.

Mirrors the local `DocumentConverter` API: `convert()` for a single source and
`convert_all()` for many, both returning a `ConversionResult`. Defaults (OCR,
table structure, Markdown output) match docling's local defaults. Pass `options=...`
only to override them.

Run from the repository root:

    python docs/examples/service_client/convert.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from docling.service_client import DoclingServiceClient

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SINGLE = Path("tests/data/pdf/regression/2305.03393v1-pg9.pdf")
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
        # One document.
        result = client.convert(source=SINGLE)
        print("convert():", result.document.name, result.status.value)
        print(result.document.export_to_markdown()[:500])

        # Many documents, converted concurrently.
        print("\nconvert_all():")
        for result in client.convert_all(source=MANY, max_concurrency=4):
            print(" ", result.input.file.name, result.status.value)


if __name__ == "__main__":
    main()
