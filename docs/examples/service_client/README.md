# Client SDK Examples

These scripts use the `docling.service_client` SDK against an **already running**
`docling-serve` instance. They do not start a service.

## Setup

Point the client at your service. The client and these examples read the same
variables as `docling convert-remote` — from the environment or a `.env` file in
the working directory:

```
DOCLING_SERVICE_URL=https://your-docling-service.example.com
DOCLING_SERVICE_API_KEY=your-api-key   # omit if the service is unauthenticated
```

Install docling-slim with the `service-client` extra :

```
pip install "docling-slim[service-client]"
```

Run the examples **from the repository root** — they reference sample documents
under `tests/data/pdf/` by relative path:

```
uv run python docs/examples/service_client/convert.py
```

## The basics

Convert one document — same call shape as a local `DocumentConverter`:

```python
from docling.service_client import DoclingServiceClient

client = DoclingServiceClient(url=..., api_key=...)
result = client.convert(source="path/to/report.pdf")  # or an http(s) URL
print(result.document.export_to_markdown())
```

Convert many concurrently:

```python
for result in client.convert_all(
    source=["a.pdf", "b.pdf", "https://.../c.pdf"],
    max_concurrency=4,
):
    print(result.input.file.name, result.status)
```

Defaults (OCR, table structure, Markdown output) match that of docling's `DocumentConverter`. Pass
`options=ConvertDocumentsOptions(...)` only when you need to override them.

## Examples


| Script       | What it shows                                                      |
| ------------ | ------------------------------------------------------------------ |
| `convert.py` | `convert()` and `convert_all()` — the high-level API               |
| `tasks.py`   | the `submit*` API: job lifecycle, result targets, per-item fan-out |
| `batch.py`   | `submit_batch()` for many or long-running HTTP/S3 sources          |
| `chunk.py`   | `chunk()` — split a document into retrieval-ready pieces           |

