<!-- Source: docling-serve@v1.21.0 — keep in sync on serve releases that touch config/deployment/usage. -->
!!! info "Synced from docling-serve v1.21.0"
    This page summarizes the [docling-serve](https://github.com/docling-project/docling-serve) documentation at **v1.21.0**. For the exhaustive reference, follow the links to the source repository.

# REST API

The docling-serve HTTP API. Examples target a local server at `http://localhost:5001`; on **Docling for IBM watsonx** use your service base URL and key.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/convert/source` | POST | convert from URLs / base64 sources (sync) |
| `/v1/convert/file` | POST | convert uploaded files, multipart (sync) |
| `/v1/convert/source/async`, `/v1/convert/file/async` | POST | submit an async task |
| `/v1/status/poll/{task_id}` | GET | poll task status |
| `/v1/status/ws/{task_id}` | WebSocket | subscribe to status |
| `/v1/result/{task_id}` | GET | fetch a finished result |

The full, interactive schema is always available at **`/docs`** (OpenAPI / Swagger) on any running server.

## Authentication

If `DOCLING_SERVE_API_KEY` is set on the server, send it on every request:

```sh
-H "X-Api-Key: <YOUR_KEY>"
```

## Conversion options (common)

Pass these in the JSON body under `options`. This is the common subset — the **authoritative, full schema is the live OpenAPI docs at `/docs`** on any running server.

| Option | Meaning |
|---|---|
| `from_formats` / `to_formats` | input / output formats (see [Supported formats](../supported_formats.md)) |
| `image_export_mode` | how images are emitted (`placeholder` / `embedded` / `referenced`) |
| `do_ocr` / `force_ocr` | enable / force OCR |
| `ocr_engine` / `ocr_lang` | OCR engine and languages |
| `table_mode` | table-structure mode (`fast` / `accurate`) |
| `pdf_backend` | PDF parsing backend |
| `pipeline` | processing pipeline (e.g. standard / VLM) |
| enrichment flags | code, formula, picture classification/description, chart |

For VLM / picture-description model configuration see [Vision models](../vision_models.md) and [Model catalog](../model_catalog.md).

## Example: convert a URL (async)

```sh
curl -X POST "http://localhost:5001/v1/convert/source/async" \
  -H "Content-Type: application/json" \
  -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
```

For a synchronous call use `/v1/convert/source` with the same body.

## Example: upload a file (multipart)

The `/v1/convert/file` endpoint accepts one or more files as `multipart/form-data`, with options as form fields:

```sh
curl -X POST "http://localhost:5001/v1/convert/file" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@2206.01062v1.pdf;type=application/pdf" \
  -F "from_formats=pdf" \
  -F "to_formats=md" \
  -F "do_ocr=true" \
  -F "image_export_mode=embedded" \
  -F "table_mode=fast"
```

## Base64 file upload

To send a file inline instead of multipart, POST to `/v1/convert/source` with `file_sources`. For large files, write the request body to a temp file and pass it with `-d @file` to avoid the shell's "Argument list too long" error:

```sh
# 1. base64-encode the file
B64_DATA=$(base64 -w 0 /path/to/document.pdf)

# 2. build the request body
cat > /tmp/request_body.json <<EOF
{
  "file_sources": [{ "base64_string": "${B64_DATA}", "filename": "document.pdf" }]
}
EOF

# 3. POST the request
curl -X POST "http://localhost:5001/v1/convert/source" \
  -H "Content-Type: application/json" \
  -d @/tmp/request_body.json
```

## Response format

A single-file conversion returns JSON:

```jsonc
{
  "document": {
    "md_content": "",
    "json_content": {},
    "html_content": "",
    "text_content": "",
    "doctags_content": ""
  },
  "status": "success",   // success | partial_success | skipped | failure
  "processing_time": 0.0,
  "timings": {},
  "errors": []
}
```

Only the `*_content` fields you requested via `to_formats` are populated. `processing_time` is in seconds; `timings` carries per-component detail when enabled. If you request the zip `target`, or the job produces multiple files, the response is a zip archive instead of JSON.

## Asynchronous API

Both convert endpoints have `/async` variants. Submitting returns a task descriptor:

```jsonc
{
  "task_id": "<task_id>",
  "task_status": "pending",   // pending | started | success | failure
  "task_position": 1,
  "task_meta": null
}
```

**Poll** until done — `GET /v1/status/poll/{task_id}`:

```python
import time
import httpx

base_url = "http://localhost:5001/v1"
task = response.json()  # from the /async submission

while task["task_status"] not in ("success", "failure"):
    task = httpx.get(f"{base_url}/status/poll/{task['task_id']}").json()
    time.sleep(5)
```

**Subscribe** via WebSocket — `/v1/status/ws/{task_id}` — for push updates (messages are JSON with `message` = `connection | update | error` plus the task object).

**Fetch** the result when finished — `GET /v1/result/{task_id}`.

## Picture description (local vs API)

When picture-description enrichment is on, choose an execution mode:

- **local** — run the vision-language model in-process (`picture_description_local`, e.g. a Hugging Face `repo_id`).
- **api** — call an external OpenAI-compatible endpoint (`picture_description_api`, with `url` / `params`). This requires launching the server with `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true`.

See [Vision models](../vision_models.md) for model choices and configuration.
