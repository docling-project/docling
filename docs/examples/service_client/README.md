# Client SDK Examples

These examples use the `docling.service_client` SDK against an already running
`docling-serve` instance. They do not start a service process.

Set the service endpoint before running them:

```bash
export DOCLING_SERVICE_URL="https://your-docling-service.example.com"
export DOCLING_SERVICE_API_KEY="your-api-key"  # optional
```

Run from the repository root, or from any environment where `docling` is
installed:

```bash
python docs/examples/service_client/convert_compat.py
python docs/examples/service_client/task_api.py
python docs/examples/service_client/batch_and_chunk.py
python docs/examples/service_client/convert_folder.py /path/to/input --hostname localhost:5001 --api-key your-api-key
```

Notes:

- `submit()` supports `PresignedUrlTarget`, `InBodyTarget`, and `ZipTarget`.
  When you omit `target`, the client tries `PresignedUrlTarget()` first and
  falls back to `InBodyTarget()` if the service rejects presigned artifacts.
- `submit_batch()` targets `POST /v1/convert/source/batch` and always requires
  an explicit batch-compatible target: `PresignedUrlTarget()` for service-managed
  download URLs, or `S3Target(...)` for caller-managed object storage.
- `submit_batch(..., headers={...})` forwards request-level service headers such
  as `X-Tenant-Id`. Per-source fetch headers still belong on the individual
  HTTP source models.
- For batch HTTP inputs, use `AnyHttpSourceRequest`; it accepts ordinary HTTP
  document URLs and ZIP URLs on the batch endpoint.
