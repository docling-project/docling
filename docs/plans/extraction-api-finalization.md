# Extraction API finalization — 2026-10-01

## TL;DR

The API implementation is complete and deployed on AWS stg-1 system. Granite and NuExtract3's API routes were verified locally. The extraction branches, smoke assets and refreshed downstream locks are published; base conflicts and Docling DCO are resolved. Remaining work is package publication, the complex-schema SaaS rerun at 60 seconds, #253 integration and separately scoped billing/metrics.

## What a finalizer must do

| Priority / owner | Concrete next action / completion evidence |
|---|---|
| Docling maintainer | Obtain the two reviewer approvals required by Mergify for test-data changes. Close or mark [#4201](https://github.com/docling-project/docling/pull/4201) superseded: #4218 now targets main directly and includes its earlier API work. No separate merge is needed for that implementation. |
| Integration — Jobkit/client owners | Coordinate [Jobkit #253](https://github.com/docling-project/docling-jobkit/pull/253), which changes all-failed task status to failure. Verify clients still retrieve structured failed extraction envelopes and item reasons. Its helper is absent from #249; do not silently assume compatibility. |
| Deployment — SaaS operator | Record deployed revisions, debug-off configuration and the server's 60-second model timeout. Run the original complex case using the default preset; inspect document status/counts and item errors. Confirm backend detail stays private. Then run the corrected 147 single-page cases into fresh output directories using the [docling-extractbench README](https://github.ibm.com/docling-project/docling-extractbench/). |
| Release — maintainers | Merge/publish Docling #4218 → Jobkit #249 → Serve #695; replace temporary upstream Git sources with released version floors and regenerate locks. |
| Separate SaaS follow-up — billing/observability owner | Define extraction operation/work units and add callback work fields and metrics. See Serve's `docs/handoff-extraction-finalization.md`. Lifecycle callbacks alone do not establish billing parity. |

Lift live validation is optional follow-up. Qwen3.5/Gemma remain explicitly deferred. Removal of `NuExtractTransformersModel` is accepted; it was never public-facing and requires no compatibility shim.

## Current source and PR state

| PR / checkout | Validated implementation revision | GitHub state checked October 1 |
|---|---|---|
| [Docling #4218](https://github.com/docling-project/docling/pull/4218) / `docling-second` | `b4d9be5d` | Targets main; mergeable; DCO green. Python 3.10–3.14 core/API, docs and packaging checks are green. Mergify awaits two approvals for test-data changes. |
| [Jobkit #249](https://github.com/docling-project/docling-jobkit/pull/249) / `docling-jobkit` | `6308522e` | Main integrated; mergeable; DCO green. Python 3.10–3.14 CI is green. |
| [Serve #695](https://github.com/docling-project/docling-serve/pull/695) / `docling-serve` | `3df7c583` | Main dependency refresh integrated; mergeable; DCO green. Package/UI/lint checks are green. |

The original timeout/connection fix and October 1 HTTP-body/item-error fixes are published on these branches. Docling DCO previously flagged 43 commits already in main because #4218 targeted an older stacked branch; targeting main excludes that inherited history without changing other authors' signoffs. The separate task-completed-callback work does not supply extraction metering.

Implementation references:

| Repository | Implementation commit | Scope |
|---|---|---|
| `docling-second` | `c8bccb025f` | HTTP-body privacy, SDK item reasons, regressions and compact plans. |
| `docling-second` | `b4d9be5d` | Safe classification of Requests-wrapped read timeouts; real HTTP fixture expectations match redacted status errors. |
| `docling-jobkit` | `0065e095cb` | Scoped callback failure reasons, regression and handoffs. |
| `docling-serve` | `7b6f5a7f53` | Portable smoke runners/configuration, corrected assertions and handoffs. |
| `docling-extractbench` | `113f403` | Published standalone harness and branch-based client lock, resolving Docling `b4d9be5d`. |

These identify implementation revisions rather than subsequent documentation-only commits. Smoke runners and regressions are committed. Jobkit and the benchmark lock Docling `b4d9be5d`; Serve locks that Docling revision and Jobkit `6308522e`. Source declarations follow branches. The three PR summaries match the published implementation. Later documentation-only commits do not change these tested lock revisions.

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
| Integrated Docling extraction/model/template/DCLX/text/streaming/service-contract/API/KServe selection | **362 passed, 1 skipped** (optional Triton gRPC). Source checked with Python 3.12 and published Core. |
| Final Docling real HTTP/API/extraction-service regressions | **149 passed, 1 skipped**; includes wrapped read-timeout classification. |
| Jobkit extraction/storage plus manager/source/export checks | **132 passed**. |
| Serve admission/settings/policy/batch/smoke checks | **147 passed**. |
| Standalone `docling-extractbench` | **7 passed**, including subset materialization; parser self-check passed. |

Required lint/type/lock/generated-doc checks passed. The Docling/main merge's added-file size hook was skipped only for main's already-committed oversized PDF fixture; subsequent scoped validation passed normally. These focused selections overlap and are not a whole-repository or fresh live SaaS run. Python 3.12 avoids the workstation's optional MLX collection abort.

Serve CI runs on extraction branch pushes. Its installed-wheel job exports dependencies from `uv.lock` and imports the wheel with Python isolated mode: released dependencies lack `ExtractSourcesRequest`, while checking out the exporter source otherwise shadows the wheel and its packaged UI. Lock regeneration uses Serve CI's pinned uv 0.12.13.

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

No home-path runtime dependency, custom Core source or sibling-checkout import is required. Docling requires published `docling-core>=2.98.0,<3`; refreshed locks resolve published Core 2.99.0. Extraction docs/assets contain no operator home paths; generated outputs are excluded from source control. Temporary upstream Git sources still require release replacement.
