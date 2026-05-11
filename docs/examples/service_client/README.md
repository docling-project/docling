# Client SDK Examples

> **Experimental:** The `docling.service_client` SDK is experimental. Its API
> may change in future releases without notice.

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
python docs/examples/service_client/convert_async.py
```

`convert_async.py` shows the async client (`AsyncDoclingServiceClient`); the others use the synchronous client. Pick whichever matches your application's event-loop model.
