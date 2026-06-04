<!-- Source: docling-serve@v1.21.0 — keep in sync on serve releases that touch config/deployment/usage. -->
!!! info "Synced from docling-serve v1.21.0"
    This page summarizes the [docling-serve](https://github.com/docling-project/docling-serve) documentation at **v1.21.0**. For the exhaustive reference, follow the links to the source repository.

# API server

Run Docling as an HTTP service with [docling-serve](https://github.com/docling-project/docling-serve) — a FastAPI server that exposes Docling's document conversion over a REST API.

## When to use it

- **API server (self-hosted or fully managed)** — call Docling over HTTP from any language; good for multi-language clients, shared/remote conversion, and scaling out.
- **Python library** — `import docling` in a Python app; best for in-process use.
- **[Jobkit](../jobkit.md)** — large-scale / distributed batch pipelines.
- **[MCP server](../mcp.md)** — expose Docling as tools to AI agents.

!!! info "Fully managed option"
    A hosted version of this same service is available as **Docling for IBM watsonx** — see the [managed option](deployment.md#fully-managed-option) on the Deployment page.

!!! note "Two MCP modes"
    docling-serve can also expose Docling over MCP, bundled in the same image (streamable-http transport). That is distinct from the standalone [MCP server](../mcp.md), which remains the canonical MCP documentation. For the bundled mode see the serve [`mcp.md`](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/mcp.md).

## Hello world

The fastest way to make a conversion call. Both options hit the *same* REST API.

=== "Managed (no install)"

    On **Docling for IBM watsonx** there is nothing to install — point the call at your service base URL with a service-issued key:

    ```sh
    curl -X POST "https://<MANAGED_SERVICE_BASE_URL>/v1/convert/source/async" \
      -H "Content-Type: application/json" \
      -H "X-Api-Key: <YOUR_KEY>" \
      -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
    ```

=== "Self-hosted"

    ```sh
    pip install "docling-serve[ui]"
    docling-serve run --enable-ui
    ```

    Then call the local server (API at `http://localhost:5001`, interactive docs at `/docs`, demo UI at `/ui`):

    ```sh
    curl -X POST "http://localhost:5001/v1/convert/source/async" \
      -H "Content-Type: application/json" \
      -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
    ```

## How it works

A request hits the docling-serve API, which runs the conversion through Docling and returns the result (synchronously, or as an async task you poll). Background jobs run on a pluggable compute engine — in-process by default, or a Redis-backed queue for scaling. For Docling's internals see [Architecture](../../concepts/architecture.md); for the full API see [REST API](rest_api.md) and [Deployment](deployment.md).

## Configuration model

docling-serve is configured by **CLI flags or environment variables**. Precedence is **environment variable > config file > defaults**.

!!! warning "Subprocess gotcha"
    When uvicorn runs with `--reload` or `--workers > 1` it spawns subprocesses, and CLI flags (e.g. `--enable-ui`, `--artifacts-path`) are ignored. Use the `DOCLING_SERVE_*` environment variables in those deployments.

### Most common settings

| Setting (env var) | What it does | Default |
|---|---|---|
| `UVICORN_HOST` / `UVICORN_PORT` | bind address / port | `0.0.0.0` / `5001` |
| `UVICORN_WORKERS` | uvicorn worker processes | `1` |
| `DOCLING_SERVE_API_KEY` | require an `X-Api-Key` header | unset |
| `DOCLING_SERVE_ENABLE_UI` | serve the Gradio demo UI at `/ui` | `false` |
| `DOCLING_SERVE_ARTIFACTS_PATH` | local path to pre-downloaded models | unset (auto-download) |
| `DOCLING_SERVE_MAX_NUM_PAGES` / `DOCLING_SERVE_MAX_FILE_SIZE` | per-request limits | unset |
| `DOCLING_SERVE_ENG_KIND` | async engine: `local` or `rq` (also `kfp`/`ray` — see serve repo) | `local` |

See the full reference in the source repo: [`configuration.md`](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/configuration.md) and [`.env.example`](https://github.com/docling-project/docling-serve/blob/v1.21.0/.env.example).

### Docling settings (env vars)

These tune Docling itself and are read by the server too:

| Env var | What it does | Default |
|---|---|---|
| `DOCLING_DEVICE` | inference device: `cpu` / `cuda` / `mps` | auto |
| `DOCLING_NUM_THREADS` | CPU threads | runtime default |
| `DOCLING_PERF_PAGE_BATCH_SIZE` | pages per batch | runtime default |
| `DOCLING_PERF_ELEMENTS_BATCH_SIZE` | elements per batch | runtime default |
| `DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS` | log per-stage timings | `false` |

For how to *choose* device/perf values see [GPU support](../gpu.md). For offline / air-gapped model setup see the [FAQ](../../faq/index.md) and [Advanced options](../advanced_options.md); set `DOCLING_SERVE_ARTIFACTS_PATH` to a pre-populated model directory.

### Choosing a compute engine

- **Local** (`DOCLING_SERVE_ENG_KIND=local`, default) — jobs run in-process; no extra services. Single machine.
- **RQ** (`DOCLING_SERVE_ENG_KIND=rq`) — jobs go to Redis and run in separate worker processes; scales horizontally.

See [Deployment](deployment.md) for how to launch each.
