# Extraction API finalization — 2026-10-01

## TL;DR

The API implementation is complete and deployed on SaaS according to the operator. Granite and NuExtract3's API routes were verified live; the GGUF run is sufficient for NuExtract3. The remaining delivery work is to push the local fixes and smoke assets, resolve PR conflicts/DCO, publish dependencies in order, and rerun the formerly timing-out SaaS case at 60 seconds. No custom Core or legacy import shim is needed.

Three confirmed error gaps are now fixed locally: raw model HTTP error bodies obey the debug setting, SDK exceptions include scoped item reasons, and callbacks retain those reasons. Focused regressions pass. These October 1 patches are **committed locally, not pushed or verified on SaaS**. The three GitHub PR summaries have already been updated.

## What a finalizer must do

| Priority / owner | Concrete next action / completion evidence |
|---|---|
| Now — PR authors | Review and push the scoped local commits below. Refresh downstream locks to the new upstream commits; their current pins contain the September 24 transport fix, not the October 1 patches. Resolve Jobkit/Serve base conflicts and Docling DCO; rerun affected CI after integration. |
| Now — Docling maintainer | Settle whether [#4201](https://github.com/docling-project/docling/pull/4201) merges first or is superseded. Its head is an ancestor of #4218, so its implementation is already included. It currently conflicts and has failing CI. |
| Integration — Jobkit/client owners | Coordinate [Jobkit #253](https://github.com/docling-project/docling-jobkit/pull/253), which changes all-failed task status to failure. Verify clients still retrieve structured failed extraction envelopes and item reasons. Its helper is absent from #249; do not silently assume compatibility. |
| Deployment — SaaS operator | Record deployed revisions, debug-off configuration and the server's 60-second model timeout. Run the original complex case using the default preset; inspect document status/counts and item errors. Confirm backend detail stays private. Then run the corrected 147 single-page cases into fresh output directories using the ExtractBench README. |
| Release — maintainers | Publish Docling → Jobkit → Serve; replace temporary upstream Git sources with released version floors and regenerate locks. |
| Separate SaaS follow-up — billing/observability owner | Define extraction operation/work units and add callback work fields and metrics. See Serve's metering handoff. Lifecycle callbacks alone do not establish billing parity. |

Lift live validation is optional follow-up. Qwen3.5/Gemma remain explicitly deferred. Removal of `NuExtractTransformersModel` is accepted; it was never public-facing and requires no compatibility shim.

## Current source and PR state

| PR / checkout | Head | GitHub state checked October 1 |
|---|---|---|
| [Docling #4218](https://github.com/docling-project/docling/pull/4218) / `docling-second` | `7f237215` | Mergeable; Linux Python 3.10–3.14, docs and packaging checks green; DCO `ACTION_REQUIRED`; Windows/macOS skipped. |
| [Jobkit #249](https://github.com/docling-project/docling-jobkit/pull/249) / `docling-jobkit` | `af845006` | Conflicts with main; Python 3.10–3.14 checks and DCO green. |
| [Serve #695](https://github.com/docling-project/docling-serve/pull/695) / `docling-serve` | `b85ead7f` | Conflicts with main; current checks show DCO/Mergify, with no test-CI result. |

The original timeout/connection debug fix is on these named PR heads, not stranded in a worktree. The separate task-completed-callback worktree does not supply extraction metering. PR checks above cover pushed heads, not the new local patches.

## Complete, fixed now, and tested

The contract is implemented end to end: top-level `extraction_target` holds schema/template/instructions; `options` holds execution policy; outer `target` selects storage. Sync/async clients, bounded per-source batches, presets, source identity/expansion, durable ordered items, validation, raw output, inference metadata, artifact routing and lifecycle callbacks are covered. Serve executes extraction through Ray; Local/RQ return 501. Expandable S3 sources require direct artifact output; S3 → in-body/presigned returns 422.

| October 1 local change | Check |
|---|---|
| Docling API and KServe HTTP errors retain safe status/context publicly; backend bodies are logged and appear in errors only with debug enabled. | Debug-off/on non-2xx regressions; provider rejection still fails without fallback. |
| SDK `ExtractionError` includes page/document item reasons; Jobkit callback projection copies scoped reasons without mutating durable errors. | Failed item-only envelope and callback regressions. |
| Smoke matrix requires `ServiceError` 422 for expected rejection; validates callback event counts and artifact presence without assuming arrival order. | Regression rejects missing artifacts/wrong counts and accepts reordered delivery. |
| ExtractBench adapter now lives with its resources, preserves structured failure evidence, classifies retryable failures, checks artifact HTTP status and rejects multi-page inputs. | Six offline integration checks, parser self-check, validated request/target examples and 147 valid relative data links. |

| Fresh test selection | Result |
|---|---|
| Docling extraction/model/template/DCLX/text/streaming/service-contract/API transport | **331 passed, 1 skipped** (optional Triton gRPC). Source checked with Jobkit's Python 3.12 environment and published Core. |
| KServe HTTP | **30 passed**, including a permitted local fake server. |
| Jobkit extraction manager / presigned results | **43 passed**. |
| Serve admission / environment parsing / smoke assertions | **64 passed**. |
| ExtractBench integration | **6 passed**, self-check and portable examples passed. |

Docling `make validate` and affected Jobkit/Serve lint/type hooks passed. Downstream lock hooks were skipped because dependency declarations/locks were unchanged; Serve's unrelated generated-doc hook was skipped. Existing additional SDK/conversion checks passed in the initial assessment after rerunning socket-denied fixtures with permission. This was not a fresh whole-repository or live SaaS test run. Docling's own Python 3.14 environment still aborts on optional MLX import; the compatible environment avoids that unrelated collection problem.

Historical September 23 live runs covered Granite/LM Studio and NuExtract3 GGUF/llama-server, PDF/DOCX matrices, Markdown/HTML, expanded S3 prefixes, encryption, page range 2–3, forced schema failure and artifacts before document callbacks. Original temporary logs are unavailable; retain the committed evidence/ledger. Historical broad runs included unrelated parser/cv2, MinIO and async/config/OTEL failures and should not be reported as fully green.

## ExtractBench handoff

The self-contained integration is `integrations/docling/` in ExtractBench: adapter, pinned client requirements, blank credential example, portable request/complex-target examples, six checks and run instructions. It needs no sibling checkout. Generated links, credentials and outputs are ignored; historical outputs were moved into `integrations/docling/runs/2026-09-24/` and preserved locally.

The historical 147-case run failed on wrong COS keys (404), not 147 model timeouts. The later one-case smoke reached the VLM and timed out at 20 seconds. Correct example-ID keys and the new 60-second server setting have **no saved successful remote rerun yet**. Client job waiting is a separate timeout. Score raw output using the baseline parser, record validation separately, and preserve complex guidance for comparable scores. The full 370-case benchmark still needs a whole-document strategy; page 1 alone is not comparable.

## Plans, portability and Git scope

Docling extraction plans/evidence were already tracked. Active plans are now compact and current; obsolete resumption/review prompts are marked completed or superseded. Historical evidence is retained with home paths removed. This assessment is the canonical finalization record; the former external report points here instead of duplicating it.

Previously untracked assets included in the local commits:

- **Serve:** `docs/handoff-extraction-source-target-matrix-2026-09-21.md`, `docs/handoff-vlm-extraction-inference-error-leak-2026-09-24.md`, `docs/handoff-extraction-metering-callbacks-2026-09-23.md`; `scripts/smoke_extraction_lmstudio.py`, `scripts/local_smoke/{__init__.py,_common.py,extraction_matrix.py,extraction_s3_to_s3_lmstudio.py}`, `scripts/extraction-smoke.example.yaml`, `tests/test_extraction_smoke.py`.
- **Jobkit:** `docs/plans/plan-extract-endpoint.md`, `docs/plans/extract-endpoint-review-handoff.md`, plus the callback fix/test.
- **Docling:** this assessment, updates to existing tracked extraction plans/evidence, transport/client fixes and affected regressions.
- **ExtractBench:** only the reusable `integrations/docling/` files above; no outputs, data links or credentials.

The redundant historical Serve LM Studio pointer and superseded plan in `docling_release` were updated locally but remain untracked; the current canonical instructions are committed instead. The old Serve ExtractBench handoff/adapter were removed after migration. Unrelated untracked Serve/Jobkit files are preserved, including the FileNet runner with its machine-specific path; do not stage entire directories blindly.

No home-path runtime dependency, custom Core source or sibling-checkout import is required. Published `docling-core>=2.96.0,<3` is sufficient. Selected extraction docs/assets contain no operator home paths; private historical logs/outputs remain local and ignored. Temporary upstream sources are HTTPS Git pins and still require release replacement.
