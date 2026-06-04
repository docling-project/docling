<!-- Source: docling-serve@v1.21.0 — keep in sync on serve releases that touch config/deployment/usage. -->
!!! info "Synced from docling-serve v1.21.0"
    This page summarizes the [docling-serve](https://github.com/docling-project/docling-serve) documentation at **v1.21.0**. For the exhaustive reference, follow the links to the source repository.

# Deployment

Get docling-serve running on one machine fast. For cluster/production hardening, follow the links to the [docling-serve repo](https://github.com/docling-project/docling-serve/tree/v1.21.0/docs).

The four ways people describe "launching" docling-serve are really **two independent choices**:

- **How you start the process:** the `docling-serve` command, or **Docker Compose**.
- **Which async engine runs the jobs** (`DOCLING_SERVE_ENG_KIND`): the in-process **Local** engine (default), or the Redis-backed **RQ** engine.

| | Local engine (default) | RQ engine (Redis + workers) |
|---|---|---|
| **`docling-serve` command** | Quickstart / dev — below | Distributed — below |
| **Docker Compose** | Containerized single node (+GPU) — below | → [serve repo](https://github.com/docling-project/docling-serve/tree/v1.21.0/docs/deploy-examples) |

## Simple command (Local engine — the default quickstart)

```sh
pip install "docling-serve[ui]"
docling-serve run --enable-ui      # production-style: reload off, binds 0.0.0.0, UI off by default
# docling-serve dev                # dev: auto-reload, binds 127.0.0.1, UI on (localhost only)
```

API at `http://localhost:5001`, interactive docs at `/docs`, demo UI at `/ui`.

!!! note
    The demo UI (`--enable-ui` / `DOCLING_SERVE_ENABLE_UI`) is a Gradio app; files it produces are cleared from its cache after ~10 hours. It is a demonstrator, not durable storage.

Smoke test:

```sh
curl -X POST "http://localhost:5001/v1/convert/source/async" \
  -H "Content-Type: application/json" \
  -d '{"http_sources": [{"url": "https://arxiv.org/pdf/2501.17887"}]}'
```

!!! warning
    When uvicorn runs with `--reload` or `--workers > 1` it spawns subprocesses, and CLI flags (e.g. `--enable-ui`, `--artifacts-path`) are ignored. Use the `DOCLING_SERVE_*` environment variables in those deployments.

The **Local engine** runs jobs in a small in-process thread pool — no Redis, no extra processes; bounded by the one host. Two knobs: `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` (default `2`) and `DOCLING_SERVE_ENG_LOC_SHARE_MODELS` (default `false`). To scale past one process, use the RQ engine below.

## RQ engine (distributed: Redis + separate workers)

The API enqueues jobs to Redis; conversion runs in separate `docling-serve rq-worker` processes, so API and compute scale independently.

```sh
# 1) Redis
docker run -p 6379:6379 redis:7-alpine

# 2) API server (enqueues jobs)
DOCLING_SERVE_ENG_KIND=rq \
DOCLING_SERVE_ENG_RQ_REDIS_URL=redis://localhost:6379/0 \
docling-serve run

# 3) one or more workers (do the conversion)
DOCLING_SERVE_ENG_KIND=rq \
DOCLING_SERVE_ENG_RQ_REDIS_URL=redis://localhost:6379/0 \
docling-serve rq-worker
```

!!! warning
    The API alone *accepts* jobs but nothing runs them without at least one `rq-worker`. `DOCLING_SERVE_ENG_RQ_REDIS_URL` is required (no default) and must be identical across every API and worker process.

## Docker Compose (Local engine, including local GPU)

Same server, containerized. The shipped compose examples are all-in-one containers that don't set `ENG_KIND`, so they run the default Local engine.

```sh
# Pure CPU (no compose)
podman run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 quay.io/docling-project/docling-serve

# NVIDIA GPU
docker compose -f compose-nvidia.yaml up -d

# AMD GPU
docker compose -f compose-amd.yaml up -d
```

Compose manifests: [`compose-nvidia.yaml`](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/compose-nvidia.yaml), [`compose-amd.yaml`](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/compose-amd.yaml).

**GPU prerequisites** (host side; for the Python `AcceleratorOptions` view see [GPU support](../gpu.md) and [RTX GPU](../../getting_started/rtx.md)):

- NVIDIA — driver ≥ 550.54.14 + `nvidia-container-toolkit` + the nvidia container runtime.
- AMD — AMDGPU/ROCm ≥ 6.3; the ROCm image is **not published**, build it with `make docling-serve-rocm-image`. Detailed GID wiring: [serve repo](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deployment.md).

!!! note
    The compose files pin older image tags (`-cu126:main`, `-rocm72:main`) than the README image table; treat the [README image table](https://github.com/docling-project/docling-serve/blob/v1.21.0/README.md) as the source of truth and adjust the `image:` line if needed. There is no shipped single-CPU compose file — use the `podman` one-liner for pure CPU.

## Cluster, production & advanced variants

These live in the docling-serve repo (run-time manifests aren't vendored here):

- [OpenShift — simple deployment](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/docling-serve-simple.yaml)
- [Multi-worker RQ on Kubernetes](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/docling-serve-rq-workers.yaml) (Redis + worker pods + secret)
- [Secure deployment with oauth-proxy / TLS / Route](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/docling-serve-oauth.yaml)
- [ReplicaSets with sticky sessions](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/deploy-examples/docling-serve-replicas-w-sticky-sessions.yaml) (task state is node-local)
- [Model-cache PVC/Job](https://github.com/docling-project/docling-serve/blob/v1.21.0/docs/models.md) (pre-baking weights)
- KFP / Ray engines, OpenTelemetry, CUDA image-tagging policy → [serve repo](https://github.com/docling-project/docling-serve/tree/v1.21.0/docs)

## Fully-managed option

!!! info "Don't want to operate it yourself?"

    All of the launch modes above run **docling-serve** yourself. If you would rather not run, scale, or maintain the infrastructure, a **fully managed, hosted version** of the same service is available as **Docling for IBM watsonx**.

    It exposes the same REST API described in these pages, so client code is portable — typically you only swap the base URL and supply a service-issued API key. Hosted-only concerns (account/key provisioning, quotas, SLA, and data handling) are documented separately and are out of scope for these self-hosting pages.
