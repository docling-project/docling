# Continue stage 11: close cross-repository offline contracts

**This prompt is preparation only. Stage 11 is pending; execute only after the
user explicitly requests it. Parent signed the stage 10 checkpoints below.**
Stages 6–7 Qwen3.5/Gemma remain user-deferred; do not resume them automatically.

Read applicable AGENTS/instructions and
`/Users/cau/Documents/Development/docling-second/docs/plans/extraction-additional-vlm-models-execution.md`,
especially stages 8–10, then Part II G and the scope/validation contracts in the
external authoritative handoff
`/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md`.
Part II supersedes the standalone proposal. Do not edit the dirty external checkout.

Recheck branch, exact parent-signed HEADs, tracked/untracked state and environment:

- `/Users/cau/Documents/Development/docling-second`,
  `cau/extraction-api-service-models`: stage 8 production baseline
  `c2d5347b7534972b3e8c7282236b7515e09b1e4f`, stage 9 tracking
  `99e3e8d634657c818de37a8256e52ac16d4c2315`; require the signed stage 10
  pure API-preflight factoring `bb6ef8b5020ffcb3f90529ad341b7c7456367515`
  plus tracking checkpoint from ledger/current HEAD.
- `/Users/cau/Documents/Development/docling-jobkit`, `cau/extract-endpoint`:
  stage 9 signed baseline `a833735833a1299d5a6878433e99bfdcb80e1629`;
  signed stage 10 startup-resolution followup is
  `7f641bf7e701d9f1411afc8b6b89e526b57a94fb`.
- `/Users/cau/Documents/Development/docling-serve`, `cau/extract-endpoint`:
  stage 10 predecessor `e9820f6069122cd1c771d1147942a0a7a9d4b8ad`;
  signed stage 10 admission/README/tests checkpoint is
  `00cd751c682b24d2be9d25c95d9b8770d2d4c93c`.

Use existing checkouts and preserve all unrelated material. Stop if Core edits are
needed; never edit, sync, build or create editable overrides of its dirty checkout.
Published Core only. Parent alone makes separately authorized scoped signed commits;
no worker commits/push/publication. Sequential work, one GPT-5.6-sol/high worker at
a time as requested, no subagents. Apply Ponytail full with bounded progress.

Trace real end-to-end extraction flows before adding tests. Prefer existing
Jobkit extraction/presigned and Serve admission doubles and result builders over
new harness abstractions. Close deterministic integration contracts at model and
transport boundaries, without loading extraction weights or live inference:

1. Valid native NuExtract and generic example/schema-only preparation; independent
   selected absolute page scopes and unpaginated document scopes.
2. Schema failure, partial input/source outcomes, raw answer/error/usage preservation,
   original task source index retained across connector expansion.
3. In-body and durable JSON artifacts; complete frozen SourceIdentity ownership for
   different expanded URIs sharing one source index; remote/presigned destinations.
4. SET_NUM_DOCS, upload-before-DOCUMENT_COMPLETED, UPDATE_PROCESSED and post-durable
   terminal TASK_COMPLETED/billing order. Controlled doubles, no customer callbacks.
5. Tenant/task/result access, source/target operator gates and unchanged conversion
   contracts. Local/RQ extraction stays prequeue rejected.

Service `options.target` is mandatory portable ExtractionTarget; outer request
`target` is the independent destination. No bare-template aliases, grouping/chunks,
Python schema validators or schema guessing. Canonical durable results contain
source index/URI/filename/status/errors/items, without runtime input/backends.
No service pages aliases or invented document page 1. Main SDK compatibility remains.
Reuse Docling preparation/preflight; do not duplicate dialect/decoder logic downstream.
Caches contain stable configuration only.

Existing Serve Python 3.12.7 works with exact
`PYTHONPATH=/Users/cau/Documents/Development/docling-serve:/Users/cau/Documents/Development/docling-jobkit:/Users/cau/Documents/Development/docling-second`;
installed Docling/Jobkit metadata is stale (2.124.0/3.5.0), published Core 2.93.0.
Jobkit Python 3.12.7 uses Core 2.92.0. Docling affected tests use
`docling_release/.venv` Python 3.13.5, published Core 2.96.0. Print actual imports,
versions and revisions; no sync/local pins/downloads. These are offline contract
environments, not a freshly synchronized release matrix. Raise release dependency
minimums only when the first published contract versions are known.

Run focused integration checks and the stage 10 applicable regressions, with
`CI=1 HF_HUB_OFFLINE=1`. Stage 10: Serve 226 passed/12 baseline deselected;
Docling 315 passed; Jobkit focused 40 passed. The central ledger has exact commands,
source/env evidence and baseline exclusions. Avoid broad live/model integration
modules. Docling `make validate` is required; Serve has a Makefile without validate,
so run native pre-commit hooks. Reproduce new baseline failures before excluding.
Known Serve: stale batch doubles, YAML expectations, S3 region, OTEL expectation;
docs generator rewrites unrelated conversion rows, uv-lock macOS NULL-object panic.
Known Jobkit: full MyPy four untouched S3 helper errors, scoped MyPy passes.
Restore only proven hook-generated unrelated changes; never edit unrelated production
or assertions just to green. Review hook edits and rerun, whitespace/path audits.

Update central ledger with exact scope, commands/exits/counts/log paths, imports,
remaining live gates and review checkpoint. NuExtract3 local/vLLM and Lift live
verification remain separate and unrun; recorded NuExtract3 LM Studio cannot deliver
caller templates. Offline closure does not certify an untested deployment or
implicitly authorize production enablement, publication, downloads or live jobs.
