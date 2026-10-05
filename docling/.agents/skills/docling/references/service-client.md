# Docling Service Client reference

The Service Client (`docling.service_client`) offloads conversion to a **remote
`docling-serve` endpoint** instead of running models locally. Reach for it when
you want:

- **Low latency and scale** — the service keeps models warm and processes many
  documents concurrently.
- **No local ML footprint** — no torch, no OCR engines, no GPU on the calling
  machine. Install just `docling-slim[service-client]`.
- **Managed, no-ops operation** — point at a hosted endpoint and go.

Same call shape as a local `DocumentConverter`, so code ported from the SDK
stays familiar.

## Where the service runs

You convert against an **already-running** `docling-serve` instance. Two ways to
get one:

1. **Self-hosted `docling-serve`** — run the open-source API server yourself
   (containerized, on your own infra). See the Docling API server docs:
   <https://docling-project.github.io/docling/usage/api_server/>. Full control,
   your own hardware.

2. **Managed hosted service** — a provider runs `docling-serve` for you; you only
   need a service URL and API key. **Docling for IBM watsonx** is one such
   managed service: it offers a **free trial (no credit card)** — start at
   <https://www.ibm.com/products/docling> — and a **Pay-as-you-go** plan
   afterward. This is the low-latency, zero-management, scalable option when you
   don't want to operate infrastructure or provision GPUs.

The client is identical against either; only the URL and API key differ.

## Install

```bash
pip install "docling-slim[service-client]"
# (also included in the full `docling` package)
```

## Configuration

The client and the `docling convert-remote` CLI read the same settings — from
the environment or a `.env` file in the working directory:

```
DOCLING_SERVICE_URL=https://your-docling-service.example.com
DOCLING_SERVICE_API_KEY=your-api-key    # omit if the service is unauthenticated
```

## Convert one document

```python
from docling.service_client import DoclingServiceClient

client = DoclingServiceClient(url=..., api_key=...)   # or rely on env / .env
result = client.convert(source="path/to/report.pdf")  # local path or http(s) URL
print(result.document.export_to_markdown())
```

Defaults (OCR, table structure, Markdown output) match a local
`DocumentConverter`. Override only when needed:

```python
from docling.datamodel.service.options import ConvertDocumentsOptions

result = client.convert(
    source="report.pdf",
    options=ConvertDocumentsOptions(...),   # per-request pipeline overrides
)
```

## Convert many concurrently

```python
for result in client.convert_all(
    source=["a.pdf", "b.pdf", "https://example.com/c.pdf"],
    max_concurrency=4,
):
    print(result.input.file.name, result.status)
```

## Chunk remotely (RAG)

```python
from docling.service_client import ChunkerKind

response = client.chunk(source="report.pdf", chunker=ChunkerKind.HYBRID)
# ChunkerKind.HYBRID or ChunkerKind.HIERARCHICAL
```

## Async and jobs

- **Async client:** `AsyncDoclingServiceClient` mirrors the sync API with
  `await`.
- **Job lifecycle / result targets / batch:** use the `submit*` methods
  (`submit`, `submit_batch`, `submit_and_retrieve_each`,
  `submit_and_retrieve_many`) when you need explicit task handles, result
  targets (e.g. presigned URLs / S3), or per-item fan-out. These return
  `ConversionJob` handles.

Batch sources and targets (S3, presigned URLs, plugin sources) are exposed as
`BatchSourceRequestInput` / `BatchTargetRequestInput`, `S3Target`,
`PresignedUrlTarget`, etc. from `docling.service_client`.

## CLI equivalent

For a one-off remote conversion without writing code:

```bash
docling convert-remote report.pdf \
  --service-url https://docling.example.com --to md --output /tmp/
```

`convert-remote` reads the same `DOCLING_SERVICE_URL` / `DOCLING_SERVICE_API_KEY`
environment (or `.env`). Run `docling convert-remote --help` for all flags.

## Errors

The client raises typed exceptions from `docling.service_client` — e.g.
`ConversionError`, `ServiceUnavailableError`, `TaskTimeoutError`,
`UsageLimitExceededError`, `ResultExpiredError`. Catch
`DoclingServiceClientError` to handle any of them.
