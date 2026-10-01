# Extraction API finalization — 2026-10-01

## TL;DR

The API implementation is complete and deployed on AWS stg-1 system. Granite and NuExtract3's API routes were verified locally. The remaining delivery work is to resolve PR conflicts/DCO, publish dependencies in order, and rerun the formerly timing-out SaaS case at 60 seconds (see ExtractBench).

## What a finalizer must do

| Priority / owner | Concrete next action / completion evidence |
|---|---|
| Now — PR authors | Push the updated branches. Refresh Jobkit/Serve and `docling-extractbench` locks to pick up the new branch revisions; their currently resolved revisions contain the September 24 transport fix, not the October 1 patches. Resolve Jobkit/Serve base conflicts and Docling DCO; rerun affected CI after integration. |
| Now — Docling maintainer | Settle whether [#4201](https://github.com/docling-project/docling/pull/4201) merges first or is superseded. Its head is an ancestor of #4218, so its implementation is already included. It currently conflicts and has failing CI. |
| Integration — Jobkit/client owners | Coordinate [Jobkit #253](https://github.com/docling-project/docling-jobkit/pull/253), which changes all-failed task status to failure. Verify clients still retrieve structured failed extraction envelopes and item reasons. Its helper is absent from #249; do not silently assume compatibility. |
| Deployment — SaaS operator | Record deployed revisions, debug-off configuration and the server's 60-second model timeout. Run the original complex case using the default preset; inspect document status/counts and item errors. Confirm backend detail stays private. Then run the corrected 147 single-page cases into fresh output directories using the [docling-extractbench README](https://github.ibm.com/docling-project/docling-extractbench/). |
| Release — maintainers | Publish Docling → Jobkit → Serve; replace temporary upstream Git sources with released version floors and regenerate locks. |
| Separate SaaS follow-up — billing/observability owner | Define extraction operation/work units and add callback work fields and metrics. See Serve's `docs/handoff-extraction-finalization.md`. Lifecycle callbacks alone do not establish billing parity. |

Lift live validation is optional follow-up. Qwen3.5/Gemma remain explicitly deferred. Removal of `NuExtractTransformersModel` is accepted; it was never public-facing and requires no compatibility shim.

## Current source and PR state

| PR / checkout | Pushed PR head | GitHub state checked October 1 |
|---|---|---|
| [Docling #4218](https://github.com/docling-project/docling/pull/4218) / `docling-second` | `7f237215` | Mergeable; Linux Python 3.10–3.14, docs and packaging checks green; DCO `ACTION_REQUIRED`; Windows/macOS skipped. |
| [Jobkit #249](https://github.com/docling-project/docling-jobkit/pull/249) / `docling-jobkit` | `af845006` | Conflicts with main; Python 3.10–3.14 checks and DCO green. |
| [Serve #695](https://github.com/docling-project/docling-serve/pull/695) / `docling-serve` | `b85ead7f` | Conflicts with main; current checks show DCO/Mergify, with no test-CI result. |

The original timeout/connection debug fix is on these named PR heads. The separate task-completed-callback work does not supply extraction metering. CI results above apply to the listed PR revisions; rerun CI after branch updates.

Implementation references:

| Repository | Implementation commit | Scope |
|---|---|---|
| `docling-second` | `c8bccb025f` | HTTP-body privacy, SDK item reasons, regressions and compact plans. |
| `docling-jobkit` | `0065e095cb` | Scoped callback failure reasons, regression and handoffs. |
| `docling-serve` | `7b6f5a7f53` | Portable smoke runners/configuration, corrected assertions and handoffs. |
| `docling-extractbench` | `e9d9e17364` | Standalone evaluation harness; supersedes the former ExtractBench integration. |

These identify implementation commits rather than the latest documentation revision. Smoke runners and their regression checks are committed.

## Implemented behavior and verification

The contract is implemented end to end: top-level `extraction_target` holds schema/template/instructions; `options` holds execution policy; outer `target` selects storage. Sync/async clients, bounded per-source batches, presets, source identity/expansion, durable ordered items, validation, raw output, inference metadata, artifact routing and lifecycle callbacks are covered. Serve executes extraction through Ray; Local/RQ return 501. Expandable S3 sources require direct artifact output; S3 → in-body/presigned returns 422.

| Implemented change | Verification |
|---|---|
| Docling API and KServe HTTP errors retain safe status/context publicly; backend bodies are logged and appear in errors only with debug enabled. | Debug-off/on non-2xx regressions; provider rejection still fails without fallback. |
| SDK `ExtractionError` includes page/document item reasons; Jobkit callback projection copies scoped reasons without mutating durable errors. | Failed item-only envelope and callback regressions. |
| Smoke matrix requires `ServiceError` 422 for expected rejection; validates callback event counts and artifact presence without assuming arrival order. | Regression rejects missing artifacts/wrong counts and accepts reordered delivery. |
| The adapter and resources live in `docling-extractbench`, preserve structured failure evidence, classify retryable failures, check artifact HTTP status and reject multi-page inputs. | Six adapter checks plus a subset-materialization check; subset files are copies, not symlinks. |

| Fresh test selection | Result |
|---|---|
| Docling extraction/model/template/DCLX/text/streaming/service-contract/API transport | **331 passed, 1 skipped** (optional Triton gRPC). Source checked with Jobkit's Python 3.12 environment and published Core. |
| KServe HTTP | **30 passed**, including a permitted local fake server. |
| Jobkit extraction manager / presigned results | **43 passed**. |
| Serve admission / environment parsing / smoke assertions | **64 passed**. |
| Standalone `docling-extractbench` | **7 passed**, including subset materialization; parser self-check passed. |

Docling `make validate` and all applicable commit hooks passed, including Jobkit/Serve lint/type checks and Serve's generated-doc/Vale hooks. Dependency lock hooks had no changed dependency files to check. Existing additional SDK/conversion checks passed in the initial assessment after rerunning socket-denied fixtures with permission. This was not a fresh whole-repository or live SaaS test run. Docling's own Python 3.14 environment still aborts on optional MLX import; the compatible environment avoids that unrelated collection problem.

Historical September 23 live runs covered Granite/LM Studio and NuExtract3 GGUF/llama-server, PDF/DOCX matrices, Markdown/HTML, expanded S3 prefixes, encryption, page range 2–3, forced schema failure and artifacts before document callbacks. Original temporary logs are unavailable; retain the committed evidence/ledger. Historical broad runs included unrelated parser/cv2, MinIO and async/config/OTEL failures and should not be reported as fully green.

## ExtractBench handoff

The canonical harness is the standalone [docling-extractbench repository](https://github.ibm.com/docling-project/docling-extractbench/), checked out as `docling-extractbench` beside the implementation repositories. It contains `adapter.py`, `prepare_subset.py`, `pyproject.toml`, `uv.lock`, `.env.example`, portable examples, adapter/subset checks and a README. ExtractBench is a pinned dependency; no sibling checkout or modifications to its framework are needed. The former integration directory is superseded.

Setup is `uv sync --locked` followed by `cp .env.example .env`; follow that repository's README for downloading, preparing and running the subset. Preparation copies the eligible PDF/test pairs and records a subset manifest. Generated data, credentials and runs are ignored. Historical run outputs were not copied into the standalone repository. Its Docling source follows PR #4218's `cau/extraction-api-service-models` branch; ExtractBench stays pinned to `72ad4027`. Refresh the lock with `uv lock --upgrade-package docling-slim` after upstream branch updates, then adopt the published release when available.

The historical 147-case run failed on wrong COS keys (404), not 147 model timeouts. The later one-case smoke reached the VLM and timed out at 20 seconds. Correct example-ID keys and the new 60-second server setting have **no saved successful remote rerun yet**. Client job waiting is a separate timeout. Score raw output using the baseline parser, record validation separately, and preserve complex guidance for comparable scores. The full 370-case benchmark still needs a whole-document strategy; page 1 alone is not comparable.

## Documentation and portability

Docling extraction plans/evidence were already tracked. Active plans are now compact and current; obsolete resumption/review prompts are marked completed or superseded. Historical evidence is retained with home paths removed. This assessment is the canonical finalization record; the former external report points here instead of duplicating it.

Repository assets:

- **Serve:** `docs/handoff-extraction-source-target-matrix-2026-09-21.md`, `docs/handoff-extraction-finalization.md` (consolidates the former error-leak and metering notes); `scripts/smoke_extraction_lmstudio.py`, `scripts/local_smoke/{__init__.py,_common.py,extraction_matrix.py,extraction_s3_to_s3_lmstudio.py}`, `scripts/extraction-smoke.example.yaml`, `tests/test_extraction_smoke.py`.
- **Jobkit:** `docs/plans/plan-extract-endpoint.md`, `docs/plans/extract-endpoint-review-handoff.md`, plus the callback fix/test.
- **Docling:** this assessment, updates to existing tracked extraction plans/evidence, transport/client fixes and affected regressions.
- **docling-extractbench:** a separate repository containing the harness and lock; no dataset PDFs, outputs or credentials are committed. The previous ExtractBench integration commit is historical provenance, not the current run location.

Use the current canonical handoffs; obsolete workstation notes are not required for finalization. The Serve ExtractBench adapter/handoff were superseded by `docling-extractbench`.

No home-path runtime dependency, custom Core source or sibling-checkout import is required. Published `docling-core>=2.96.0,<3` is sufficient. Extraction docs/assets contain no operator home paths; generated outputs are excluded from source control. Temporary upstream Git sources still require release replacement.
