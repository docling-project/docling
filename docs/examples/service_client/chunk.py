"""Chunk a document into retrieval-ready pieces with chunk().

`chunk()` converts a source and splits it with the requested chunker in one call,
returning the chunks plus the documents they came from.

Run from the repository root:

    python docs/examples/service_client/chunk.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from docling.service_client import ChunkerKind, DoclingServiceClient

load_dotenv()  # DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY from env or a .env

SOURCE = Path("tests/data/pdf/regression/2305.03393v1-pg9.pdf")


def main() -> None:
    with DoclingServiceClient(
        url=os.environ["DOCLING_SERVICE_URL"],
        api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
    ) as client:
        response = client.chunk(source=SOURCE, chunker=ChunkerKind.HIERARCHICAL)
        print(
            len(response.chunks), "chunks from", len(response.documents), "document(s)"
        )
        for chunk in response.chunks[:3]:
            print("---")
            print(chunk.text[:300])


if __name__ == "__main__":
    main()
