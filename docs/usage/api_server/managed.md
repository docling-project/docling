# Fully managed

Prefer not to run the server yourself? **Docling for IBM watsonx** is a fully managed, hosted instance of the same Docling service described in these pages.

## Why use the managed service

- **No infrastructure to run.** No servers, GPUs, scaling, upgrades, or operational monitoring to manage — the service is hosted and maintained for you.
- **Simple integration.** It exposes the same [REST API](rest_api.md) as the self-hosted server, so wiring it into applications and AI agents is just an HTTP call — point your client at the managed endpoint and go. Client code stays portable: typically you only swap the base URL and supply your API key.
- **Same Docling conversion.** The same document understanding and output formats as the open-source library and server.

## Using it

You need the **service URL** and an **API key** issued by the service, then call it exactly like the [REST API](rest_api.md):

```sh
curl -X POST "https://<MANAGED_SERVICE_BASE_URL>/v1/convert/source/async" \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <YOUR_KEY>" \
  -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
```

Account setup, API-key provisioning, quotas, SLA, and data handling are documented with the service itself.
