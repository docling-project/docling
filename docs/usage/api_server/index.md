<!-- Source: docling-serve@v1.21.0 — keep in sync on serve releases that touch config/deployment/usage. -->
!!! info "Synced from docling-serve v1.21.0"
    This page summarizes the [docling-serve](https://github.com/docling-project/docling-serve) documentation at **v1.21.0**. For the exhaustive reference, follow the links to the source repository.

# API server

Run Docling as an HTTP service with [docling-serve](https://github.com/docling-project/docling-serve) — a FastAPI server that exposes Docling's document conversion over a REST API.

## When to use what?

| You want to… | Use |
|---|---|
| Call Docling over HTTP from any language (including Python), or share one conversion service | the **API server** — [self-host](deployment.md) it, or use a [managed service](managed.md) |
| Run Docling directly (in-process) in a Python application | the [Python library](../../getting_started/quickstart.md) |
| Run large-scale or distributed batch conversions | [Jobkit](../jobkit.md) |
| Expose Docling as tools to an AI agent | the [MCP server](../mcp.md) |

!!! note "Using the API server from an MCP agent"
    The [MCP server](../mcp.md) can convert through this REST API instead of running Docling locally — set `DOCLING_CONVERSION_MODE=remote` and point it at your service URL. See [MCP server](../mcp.md).

## Getting started

To use the API server you need a running endpoint — [run your own](deployment.md) or use a [managed service](managed.md) — plus its **service URL** and, if the server requires one (`DOCLING_SERVE_API_KEY`), an **API key**. A local server defaults to `http://localhost:5001` (interactive API docs at `/docs`):

```sh
curl -X POST "http://localhost:5001/v1/convert/source/async" \
  -H "Content-Type: application/json" \
  -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
```

See [Deployment](deployment.md) to run and configure it, and [REST API](rest_api.md) for the full API.

## How it works

A request hits the docling-serve API, which runs the conversion through Docling and returns the result (synchronously, or as an async task you poll). Background jobs run on a pluggable [compute engine](deployment.md#compute-engines) — in-process by default, or a Redis-backed queue for scaling. For the full API see [REST API](rest_api.md).
