# Continue stage 11: close the remaining gaps and run the final verification

**Rewritten 2026-09-23. This prompt is preparation only. Run stage 11 only when the user asks for it.** It replaces the earlier stage 11 prompt, which assumed `options.target`, borrowed environments through `PYTHONPATH` and planned a new offline end-to-end harness. After stage 10, much of that harness's purpose is already covered by downstream unit tests and the live SDK matrix. Stage 11 now closes the few real gaps, runs the final verification once, and reports the release gates. Stages 6–7 (Qwen3.5, Gemma) remain deferred by the user and are not part of this stage.

## Read first

- Applicable AGENTS/CLAUDE instructions in all three repositories.
- The central ledger [extraction-additional-vlm-models-execution.md](extraction-additional-vlm-models-execution.md), stages 8–10 and the stage 11 section.
- [extraction-service-client-review-handoff.md](extraction-service-client-review-handoff.md) for the current client shape.
- `docling-serve/docs/handoff-extraction-source-target-matrix-2026-09-21.md` for the live stack and the smoke matrix. That file is untracked in Serve.
- Part II G of the external handoff `/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md`, for intent only. Its service shape predates C1; the code and this prompt win where they differ. Do not edit that checkout.

## Baseline

Recheck these before starting and record what differs. All three branches were pushed and had no tracked changes on 2026-09-23 apart from the uncommitted plan-doc edits in Docling.

| Repository | Branch | HEAD |
|---|---|---|
| `/Users/cau/Documents/Development/docling-second` | `cau/extraction-api-service-models` | `1d3f63d2761054ed676aba2c045335c386aa1335` |
| `/Users/cau/Documents/Development/docling-jobkit` | `cau/extract-endpoint` | `00c7022b596c35f398387ed96f543d00b2a1d65d` |
| `/Users/cau/Documents/Development/docling-serve` | `cau/extract-endpoint` | `0de30b7fbe4b51050239ae6187494bb0ccd0927b` |

Jobkit and Serve pin Docling (and Serve pins Jobkit) as git branch sources in `[tool.uv.sources]`. Their `uv.lock` files and `.venv`s resolve to the exact heads above. Check with `cat .venv/lib/python3.12/site-packages/docling_{slim,jobkit}-*.dist-info/direct_url.json`. If a head moved, push it and relock first; the venvs do not see sibling working trees. No `PYTHONPATH` override is needed for Jobkit or Serve. Docling tests keep the previous setup: `docling_release/.venv` (Python 3.13.5) with `PYTHONPATH=/Users/cau/Documents/Development/docling-second`, because this checkout's own `.venv` is Python 3.14 and was not used for earlier stages.

## Current contract (post stage 10)

These facts replace older wording in the ledger and handoff:

- The contract is the top-level `ExtractSourcesRequest.extraction_target`. `ExtractDocumentsOptions` holds only operational settings (preset/custom config, output mode, channels, page range), all with defaults. `schema_constrained` without `output_schema` fails request validation.
- `ExtractionItem.errors` holds `ErrorItem`s. Inference telemetry lives in `ExtractionItem.inference_metadata`. Scopes are `{"kind": "page", "page_no": n}` or `{"kind": "document"}`.
- Client: `extract()` handles one non-expandable source and returns a local `DocumentExtractionResult`. `extract_all()` runs one job per source with bounded concurrency. `submit_extract(source, extraction_target, options=, target=)` covers storage targets and returns typed presigned or storage responses. The client has no `template=`.
- Serve: extraction runs only on Ray; Local and RQ return 501. Expandable sources (S3, Google Drive, ...) require a storage target; in-body and presigned targets return 422. S3→S3 returns counts only, and each document is stored as `<stem>.extraction.json`. ZIP sources are rejected in the shared request models, which also changes conversion. Operator-defined presets, the `"default"` preset sentinel and the ServicePolicy custom-config gate are in place.

## Already covered — do not rebuild

- Docling: preparation, absolute page scopes, document scopes, validation states, failure and timeout statuses (stages 1–5 tests).
- Jobkit: target forwarding, cache isolation, durable items, frozen `SourceIdentity`, presigned processors, callback order, `page_range` forwarding (`tests/test_extraction_manager.py`, `tests/test_presigned_target_results.py`).
- Serve: admission, pairing rules, 501 on Local/RQ, tenant metadata on enqueue (`tests/test_extraction_admission.py`, `tests/test_service_policy.py`). Result access is checked per task by the generic `_assert_task_tenant` (`tests/test_tenant_isolation.py`).
- Live, 2026-09-21/23, Granite via LM Studio and NuExtract3 via llama-server: every source and target for a PDF, multi-document S3 prefixes, an encrypted-PDF failure, and a source without pages (DOCX) on every source and target. Markdown and HTML only for `file → inbody`. The native `nuextract` template ran through `scripts/smoke_extraction_lmstudio.py`.

## Gaps to close

Keep each one small and reuse the existing scripts and tests. Don't build a new harness.

1. **Page range through the service.** Add `--page-range START-END` to `scripts/local_smoke/extraction_matrix.py`. For the PDF, `2-3` must return two items with scopes `page_no` 2 and 3, in that order.
2. **Schema failure through the service.** Add `--schema-fail` to the matrix, using a schema the model's answer cannot satisfy (for example `title` as `integer`, `prompt_only`). Expect `validation_status: "failed"` with `raw_text` kept, a structured validation `ErrorItem`, and a document status other than `success`. Record the observed status (`partial_success` or `failure`) and confirm it matches the Docling status rules.
3. **Callback order.** Add `--callback-url` to the matrix, backed by a local receiver started by the script (same pattern as `start_http_server`). For one S3→S3 run with two or more documents, record the order `SET_NUM_DOCS`, then for each document an upload before `DOCUMENT_COMPLETED`, then `UPDATE_PROCESSED`, with `TASK_COMPLETED` last. Use only this local receiver; no external callbacks.
4. **Tenant access (optional).** Add an extraction task case to `tests/test_tenant_isolation.py` only if its `owned_task` fixture takes a task type cheaply. Otherwise record that access is covered by composition.
5. **Conversion after the shared-model changes.** Run the Serve and Jobkit convert and batch suites, plus the Docling `test_service_*` suites, to prove that the only conversion changes are the intended ones: ZIP rejection, the `_normalize_source` http fix and `SerializeAsAny` on `VlmModelSpec` fields.

## Final verification run

1. **Offline regressions at the exact heads**, with `CI=1 HF_HUB_OFFLINE=1`:
   - Docling: the extraction and service-client suites from the ledger's stage 8/10 commands, plus `tests/test_extraction_service_contract.py`, then `make validate`.
   - Jobkit: extraction, presigned, orchestrator and connector tests, excluding the known baseline failures in the ledger.
   - Serve: admission, policy, tenant and batch/convert tests, plus native pre-commit hooks.

   Reproduce any new failure on the untouched baseline before excluding it.
2. **Live matrix against the local stack** (start-up is in the Serve matrix handoff). **Restart docling-serve first**: the server running on 2026-09-23 started before `0de30b7`, so it still accepted `s3 → presigned`. Then run:
   - the full PDF matrix and the full `--doc-format docx` matrix for both models (36 runs; `s3 → presigned` must now be rejected);
   - `--doc-format md` and `html` for `file`/`http` → `inbody`;
   - multi-document S3 and the encrypted PDF;
   - gaps 1–3.

   Keep the full logs with their exit codes.
3. **Lift.** Ask the user whether Lift gets a live vLLM run, which needs weights, a server and authorization. Otherwise it stays documented as contract-tested only.

## Report, don't fix

These are release gates. List them in the ledger with their current state; changing them needs a separate decision:

- Replace the git branch sources in Jobkit and Serve `pyproject.toml` with the first published Docling/Jobkit versions that contain this contract, and raise the minimums.
- `NuExtractTransformersModel` was removed in `0a1e58d7`, but `origin/main` ships it. Restore a deprecated shim, or record the break in the release notes.
- NuExtract3 live evidence is a quantized GGUF (Q5_K_M) through llama-server's generic OpenAI-compatible API, not the pinned revision on Transformers or vLLM. Decide whether llama-server becomes a claimed transport. LM Studio remains unable to deliver caller templates for NuExtract3.
- Stages 6–7 stay deferred. Only NuExtract 2, Granite Vision, NuExtract3 and Lift are documented.
- The external handoff in `docling_release` still describes the pre-C1 service shape and the kept `NuExtractTransformersModel`. It needs a dated deviations note once its checkout may be edited.

## Rules

- Stop if a docling-core change is needed. Use published Core only; never edit, sync or build its checkout.
- Preserve all unrelated tracked and untracked files. Restore only proven hook-generated unrelated edits.
- Live runs use only the local stack (LM Studio, llama-server, MinIO, Redis). No model downloads, and no external or customer callbacks.
- Do not commit, push or publish. When the user asks for commits, author them as Christoph Auer with his own `Signed-off-by` and no Claude attribution lines.
- Write Markdown without hard line wraps.

When done, update the ledger's stage 11 checkpoint with scope, commands, exit codes, counts, log paths, the live-verification table and the release gates above. Offline closure plus a live smoke does not certify untested deployments or authorize production enablement.
