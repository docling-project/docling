"""Structured extraction with DoclingServiceClient.

Mirrors the local `DocumentExtractor` API: `extract()` for a single source and
`extract_all()` for many, both returning `DocumentExtractionResult`s, and both
taking the contract as `target=` (an `ExtractionTarget`: what to extract).
`options` (`ExtractDocumentsOptions`) is purely operational — model preset,
decode mode, input channel, page range — and defaults to the server's defaults.

Run from the repository root:

    python docs/examples/service_client/extract.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from docling.datamodel.extraction import ExtractionTarget, ExtractionTemplate
from docling.service_client import (
    DoclingServiceClient,
    ExtractDocumentsOptions,
    ExtractionError,
)

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SINGLE = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")
MANY = [
    Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf"),
    Path("tests/data/pdf/sources/code_and_formula.pdf"),
]

# What to extract. A schema, a template, or both — here a NuExtract-native template.
TARGET = ExtractionTarget(
    template=ExtractionTemplate(
        format="nuextract",
        value={"title": "verbatim-string", "authors": ["string"]},
    )
)


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        # One document.
        try:
            document = client.extract(SINGLE, target=TARGET)
        except ExtractionError as exc:
            print("extract() failed:", exc)
        else:
            print("extract():", document.input.file.name, document.status.value)
            for item in document.items:
                print(" ", item.scope, item.extracted_data, item.errors)

        # Operational overrides live on `options`, never on the contract.
        document = client.extract(
            SINGLE,
            target=TARGET,
            options=ExtractDocumentsOptions(extraction_preset="granite_vision_4_1"),
            page_range=(1, 2),
        )
        print("\nwith preset:", document.input.file.name, document.status.value)

        # Many sources: one job per source, at most `max_concurrency` at a time,
        # yielded as each job completes (not in input order).
        print("\nextract_all():")
        for document in client.extract_all(MANY, target=TARGET, max_concurrency=4):
            print(" ", document.input.file.name, document.status.value)


if __name__ == "__main__":
    main()
