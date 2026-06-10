<!-- Source: docling-serve@v1.21.0 — keep in sync on serve releases that touch config/deployment/usage. -->
!!! info "Synced from docling-serve v1.21.0"
    This page summarizes the [docling-serve](https://github.com/docling-project/docling-serve) documentation at **v1.21.0**. For the exhaustive reference, follow the links to the source repository.

# API server

Run Docling as an HTTP service with [docling-serve](https://github.com/docling-project/docling-serve) — a FastAPI server that exposes Docling's document conversion over a REST API.

## When to use what?

| You want to… | Use |
|---|---|
| Call Docling over HTTP from any language, or share one conversion service | the **API server** — [self-host](deployment.md) it, or use the [fully managed](managed.md) option |
| Use Docling directly inside a Python application | the [Python library](../../getting_started/quickstart.md) |
| Run large-scale or distributed batch conversions | [Jobkit](../jobkit.md) |
| Expose Docling as tools to an AI agent | the [MCP server](../mcp.md) |

!!! note "Two MCP modes"
    docling-serve can also expose Docling over MCP, bundled in the same image (streamable-http transport). That is distinct from the standalone [MCP server](../mcp.md), which remains the canonical MCP documentation. For the bundled mode see the serve [`mcp.md`](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/mcp.md).

## Getting started

Install and start the server:

```sh
pip install "docling-serve[ui]"
docling-serve run --enable-ui
```

To call the API you need the **service URL** and, if the server requires one (`DOCLING_SERVE_API_KEY`), an **API key**. For a local run the service URL is `http://localhost:5001` — interactive API docs are at `/docs` and the demo UI at `/ui`.

```sh
curl -X POST "http://localhost:5001/v1/convert/source/async" \
  -H "Content-Type: application/json" \
  -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
```

See [Deployment](deployment.md) for configuration and the other ways to run it, and [REST API](rest_api.md) for the full API.

## How it works

A request hits the docling-serve API, which runs the conversion through Docling and returns the result (synchronously, or as an async task you poll). Background jobs run on a pluggable [compute engine](deployment.md#compute-engines) — in-process by default, or a Redis-backed queue for scaling. For Docling's internals see [Architecture](../../concepts/architecture.md); for the full API see [REST API](rest_api.md).
