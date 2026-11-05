"""Export retrieval-ready chunks through the normal convert endpoint.

The convert endpoint writes one JSON object per line to a `.chunks.jsonl` file.
Use a zip target when the client needs to retrieve those chunks locally.

Run from the repository root:

    python docs/examples/service_client/chunk.py
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.chunking import HybridChunkerOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.targets import ZipTarget
from docling.service_client import DoclingServiceClient, RawServiceResult

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SOURCE = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        job = client.submit(
            source=SOURCE,
            options=ConvertDocumentsOptions(
                to_formats=[OutputFormat.CHUNKS],
                chunking_options=HybridChunkerOptions(
                    use_markdown_tables=True,
                ),
            ),
            target=ZipTarget(),
        )
        result = job.result()
        if not isinstance(result, RawServiceResult):
            raise TypeError("Expected the conversion result to be a zip archive.")

        with ZipFile(BytesIO(result.content)) as archive:
            chunk_files = [
                name for name in archive.namelist() if name.endswith(".chunks.jsonl")
            ]
            if len(chunk_files) != 1:
                raise ValueError("Expected one chunks.jsonl file in the archive.")
            chunks = archive.read(chunk_files[0]).decode("utf-8").splitlines()

        print(len(chunks), "chunks")
        for chunk in chunks[:3]:
            print("---")
            print(chunk[:300])


if __name__ == "__main__":
    main()
