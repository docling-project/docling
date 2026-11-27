"""Structured extraction with DoclingServiceClient.

Mirrors the local `DocumentExtractor` API: `extract()` for a single source and
`extract_all()` for many, both returning `DocumentExtractionResult`s, and both
taking the contract as `spec=` (an `ExtractionSpec`: what to extract).
`options` (`ExtractDocumentsOptions`) is purely operational — model preset,
decode mode, input channel, page range — and defaults to the server's defaults.

An `ExtractionSpec` has two fields with one role each:

- `json_schema` is the contract. The server validates each answer against it
  and, by default, converts it into the model input.
- `native_template` overrides the model input only. It is in the native
  template format of the model (NuExtract) and goes to the model as is.

Run from the repository root:

    python docs/examples/service_client/extract.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from docling.datamodel.extraction import ExtractionSpec
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

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title"],
}

# 1. Schema only: the server derives the model input and validates the answer.
SCHEMA_ONLY = ExtractionSpec(json_schema=SCHEMA)

# 2. Schema and template: the template guides a NuExtract model with a native
#    type (`verbatim-string`) that a JSON Schema cannot express; the schema
#    still validates the answer.
SCHEMA_AND_TEMPLATE = ExtractionSpec(
    json_schema=SCHEMA,
    native_template={"title": "verbatim-string", "authors": ["string"]},
)

# 3. Template only: no validation (`validation_status` is `not_requested`).
TEMPLATE_ONLY = ExtractionSpec(
    native_template={"title": "verbatim-string", "authors": ["string"]}
)


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        # One document, for each of the three combinations.
        for name, spec in [
            ("schema only", SCHEMA_ONLY),
            ("schema and template", SCHEMA_AND_TEMPLATE),
            ("template only", TEMPLATE_ONLY),
        ]:
            try:
                document = client.extract(SINGLE, spec=spec)
            except ExtractionError as exc:
                print(f"extract() [{name}] failed:", exc)
                continue
            print(f"extract() [{name}]:", document.status.value)
            for item in document.items:
                print(" ", item.scope, item.extracted_data, item.validation_status)

        # Operational overrides live on `options`, never on the contract. A
        # generic chat preset has no native template format, so use a schema.
        document = client.extract(
            SINGLE,
            spec=SCHEMA_ONLY,
            options=ExtractDocumentsOptions(extraction_preset="granite_vision_4_1"),
            page_range=(1, 2),
        )
        print("\nwith preset:", document.input.file.name, document.status.value)

        # Many sources: one job per source, at most `max_concurrency` at a time,
        # yielded as each job completes (not in input order).
        print("\nextract_all():")
        for document in client.extract_all(MANY, spec=SCHEMA_ONLY, max_concurrency=4):
            print(" ", document.input.file.name, document.status.value)


if __name__ == "__main__":
    main()
