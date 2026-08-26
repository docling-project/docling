# Docling Client

**Lightweight client SDK for converting documents via a remote [Docling Serve](https://github.com/docling-project/docling-serve) endpoint**

`docling-client` is a meta-package that installs [`docling-slim[service-client]`](https://pypi.org/project/docling-slim/), giving you `DoclingServiceClient` — a drop-in replacement for the local `DocumentConverter` that offloads conversion to a Docling Serve instance over HTTP.

For the full documentation, see the [Docling docs](https://docling-project.github.io/docling/usage/api_server/).

## Why a remote client?

| You want to… | Use |
|---|---|
| Convert documents in a Python application, **without running models locally** | `docling-client` → point at a Docling Serve endpoint |
| Run Docling directly **in-process** in a Python application | [`docling`](https://pypi.org/project/docling/) |
| Full control over which extras are installed | [`docling-slim[service-client]`](https://pypi.org/project/docling-slim/) |

Switching from local to remote conversion typically requires changing only the client class and the endpoint URL — the conversion API (sources, options, output formats) stays the same.

## Getting started

### 1. Install

```bash
pip install docling-client
```

### 2. Point at a Docling Serve endpoint

You need a running [Docling Serve](https://github.com/docling-project/docling-serve) instance — [self-hosted](https://docling-project.github.io/docling/usage/api_server/deployment/) or a [managed service](#managed-services).

Set your connection details in the environment (or a `.env` file):

```
DOCLING_SERVICE_URL=https://your-docling-service.example.com
DOCLING_SERVICE_API_KEY=your-api-key   # omit if the service is unauthenticated
```

### 3. Convert a document

```python
import os
from docling.service_client import DoclingServiceClient

with DoclingServiceClient(
    url=os.environ["DOCLING_SERVICE_URL"],
    api_key=os.environ.get("DOCLING_SERVICE_API_KEY", ""),
) as client:
    result = client.convert(source="https://arxiv.org/pdf/2501.17887")
    print(result.document.export_to_markdown())
```

Convert many documents concurrently:

```python
sources = [
    "https://arxiv.org/pdf/2501.17887",
    "path/to/report.pdf",
    "path/to/slides.pptx",
]

with DoclingServiceClient(url=os.environ["DOCLING_SERVICE_URL"]) as client:
    for result in client.convert_all(source=sources, max_concurrency=4):
        print(result.input.file.name, result.status)
        print(result.document.export_to_markdown()[:200])
```

## Switching from local to remote

If you already use the local `DocumentConverter`, the client API mirrors it closely. Only the import and instantiation change:

```python
# Before — local, runs models on this machine
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert("report.pdf")

# After — remote, offloads conversion to Docling Serve
from docling.service_client import DoclingServiceClient
converter = DoclingServiceClient(url="https://...", api_key="...")
result = converter.convert(source="report.pdf")
```

Both `result.document.export_to_markdown()` and other output methods work the same way.

## Managed services

Running [Docling Serve](https://github.com/docling-project/docling-serve) yourself means operating infrastructure. Managed services remove that overhead.

### Docling for IBM watsonx

A fully managed, hosted instance of Docling Serve — no servers, GPUs, scaling, or operational monitoring required. It exposes the same REST API, so your client code stays portable: swap the base URL, supply your API key, and go.

- [Product page](https://www.ibm.com/products/docling)
- [Free trial](https://www.ibm.com/products/docling) — no credit card needed

## More examples

Runnable examples are in [`docs/examples/service_client/`](https://github.com/docling-project/docling/tree/main/docs/examples/service_client/) in the repository:

| Script | What it shows |
|---|---|
| [`convert.py`](https://github.com/docling-project/docling/blob/main/docs/examples/service_client/convert.py) | `convert()` and `convert_all()` — the high-level API |
| [`tasks.py`](https://github.com/docling-project/docling/blob/main/docs/examples/service_client/tasks.py) | Job lifecycle: `submit()`, `watch()`, `result()`, result targets |
| [`batch.py`](https://github.com/docling-project/docling/blob/main/docs/examples/service_client/batch.py) | `submit_batch()` for plugin sources and artifact targets |
| [`chunk.py`](https://github.com/docling-project/docling/blob/main/docs/examples/service_client/chunk.py) | `chunk()` — split a document into retrieval-ready pieces |

## Documentation

- [API server overview](https://docling-project.github.io/docling/usage/api_server/)
- [Deployment guide](https://docling-project.github.io/docling/usage/api_server/deployment/)
- [Managed services](https://docling-project.github.io/docling/usage/api_server/managed/)
- [REST API reference](https://docling-project.github.io/docling/usage/api_server/rest_api/)

## License

MIT License — see [LICENSE](https://github.com/docling-project/docling/blob/main/LICENSE)
