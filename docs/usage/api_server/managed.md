# Why use the managed service

Running the API server yourself means operating infrastructure. A managed service removes that:

- **No infrastructure to run.** No servers, GPUs, scaling, upgrades, or operational monitoring — it is hosted and maintained for you.
- **Simple integration.** It exposes the same [REST API](rest_api.md) as the self-hosted server, so wiring it into applications and AI agents is just an HTTP call — point your client at the managed endpoint and go. Client code stays portable: typically you only swap the base URL and supply your API key.
- **Same Docling conversion.** The same document understanding and output formats as the open-source library and server.

## Available managed services

### Docling for IBM watsonx

A fully managed, hosted instance of the Docling service, exposing the same [REST API](rest_api.md) described in these pages. You provide the service URL and an API key, then call it like any other endpoint.

- [Product page](https://www.ibm.com/products/docling)
<!-- - [Free trial](https://www.ibm.com/products/docling) — hidden for the moment -->
