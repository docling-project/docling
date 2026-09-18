# Continue stage 10: Serve extraction admission and endpoint migration

Implement **stage 10 only** in `/Users/cau/Documents/Development/docling-serve`,
branch `cau/extract-endpoint`. Read applicable AGENTS/instructions and recheck
branch, HEAD, tracked/untracked state and existing environment before edits. The
read-only baseline on 2026-09-18 was
`e9820f6069122cd1c771d1147942a0a7a9d4b8ad`, with no tracked modifications.
Preserve unrelated material and use the existing checkout.

Read the central ledger:
`/Users/cau/Documents/Development/docling-second/docs/plans/extraction-additional-vlm-models-execution.md`,
and Part II G plus A/B/D of the external authoritative handoff:
`/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md`.
Part II supersedes its standalone proposal; do not edit that dirty external checkout.
Stages 1–5 and 8–9 are implemented. Qwen3.5/Gemma stages 6–7 are user-deferred;
stage 11 is pending and not requested. The user authorized stages 8–10, using
source/offline contracts as implementation gates; live verification is separate.
Require parent stage 9 review/signoff before work and resolve its exact signed
Jobkit HEAD `a833735833a1299d5a6878433e99bfdcb80e1629` from the ledger/current checkout. Parent alone makes authorized scoped
signed-off commits after review; no worker commits, pushes or publication.

Use exact sources, never stale installed service DTOs:

- Docling: `/Users/cau/Documents/Development/docling-second`, branch
  `cau/extraction-api-service-models`, stage 8 signed source HEAD
  `c2d5347b7534972b3e8c7282236b7515e09b1e4f`; subsequent stage 9 tracking changes
  are docs only. Verify current HEAD and provenance.
- Jobkit: `/Users/cau/Documents/Development/docling-jobkit`, branch
  `cau/extract-endpoint`; stage 9 predecessor
  `51a6339d715045ebca9a2b8e206c75e62444961d`. Require its parent's signed stage 9
  checkpoint, with the target/items migration and source identity fix.

**Stop if Docling Core changes are needed.** Never edit, sync, build or create
editable overrides of its dirty checkout. Consume published Core only. No model
weights/downloads, live extraction inference, production defaults or external
callbacks are required/authorized. Work sequentially, one agent at a time,
GPT-5.6-sol/high as requested; no subagents. Apply Ponytail full and bounded
substantive progress about every minute.

## Contract and scope

Trace all extraction policy/startup/settings/endpoint/result callers before edits.
Primary paths are `docling_serve/{app,policy,settings}.py`, affected manager wiring,
response/result/OpenAPI references and `tests/test_service_policy.py` plus relevant
endpoint/authorization tests. The current policy uses old
`ExtractDocumentsOptions(template={})` at startup and resolves the model for
requests; migrate that contract directly, without compatibility aliases for this
unreleased API. Preserve the existing asynchronous source extraction endpoint,
conversion/chunking surfaces, connector expansion and artifact destinations.
Local/RQ extraction must remain rejected before enqueue because they lack execution.

`ExtractDocumentsOptions.target` is mandatory `ExtractionTarget`, with optional
`output_schema`, explicitly tagged `ExtractionTemplate(format="nuextract" |
"example_json", value={...})` and instructions. At least schema or template is
required. No bare-template aliases, grouping/chunk inputs, Python classes/validators
or schema inference from examples. `ExtractSourcesRequest.target` remains the
independent output destination, not extraction guidance. `output_mode` is explicitly
`prompt_only` (default) or `schema_constrained` and service mode controls custom
configuration. Jobkit resolves that override through Docling option validation.
Cache only stable model/engine/channel/mode config, never task targets/validators.

Reuse actual Docling preparation/preflight helpers (currently
`docling.models.extraction.prompt_utils.prepare_target` and
`prepare_output_target`; read their signatures and callers first). Docling owns
schema conversion/dialect/decoder compatibility and response validation. Do not
copy model/template/schema logic into Serve or instantiate/load pipelines for
admission. Validate the resolved model against target format, schema support,
engine/output mode and requested channels before enqueue. Preserve remote-service
policy, custom-config/preset/engine/format allow-lists and error classification.
Default/startup admission must validate stable operator configuration without
fabricating an incompatible target for a generic model. Only implemented presets
NuExtract2/Granite/NuExtract3/Lift may be assumed; do not add deferred models.

Durable `ExtractionDocumentResult` contains required original source index,
expanded source URI, filename, status/errors and canonical `ExtractionItem`s.
Items carry absolute page/document scopes, extracted JSON/raw answer/item errors,
validation (`not_requested`, `not_run`, `passed`, `failed`) and inference metadata.
Task results/in-body responses/JSON artifacts use this envelope; never serialize
runtime `DocumentExtractionResult.input`/backends or duplicate item types. Unpaginated
text is document-scoped; no invented page 1. Old service `pages` shapes have no
aliases; main's SDK `template=`/`ExtractionResult.pages` stays compatible.

Jobkit stage 9 preserves original `task.sources` indices across connector expansion;
multiple document URIs can share one index. S3/Azure presigned artifact ownership
now keys by complete frozen SourceIdentity. Preserve that identity and storage
contracts. Keep SET_NUM_DOCS, upload-before-DOCUMENT_COMPLETED, UPDATE_PROCESSED and
post-durable-terminal TASK_COMPLETED/billing ordering. Exercise result unions,
tenant/task/result authorization, failure/partial responses and remote destinations.

## Validation and continuation

Use an existing compatible environment; verify imports/dependencies before selecting
it. No sync or persistent local path pins. Example exact source route:

```sh
CI=1 HF_HUB_OFFLINE=1 \
PYTHONPATH=/Users/cau/Documents/Development/docling-serve:/Users/cau/Documents/Development/docling-jobkit:/Users/cau/Documents/Development/docling-second \
  /path/to/existing/python -m pytest ...
```

Print actual imported Docling/Jobkit/Core paths. Jobkit stage 9 used its Python 3.12.7
venv, with published Core 2.92.0; installed Docling slim 2.124.0/Jobkit 3.5.0 metadata
were stale and source PYTHONPATH bypassed them. Borrowed Docling Python 3.13 lacked
Jobkit dependencies. Do not assume either environment is a synchronized release
matrix. Serve currently declares Docling >=2.127.0 and Jobkit >=3.7.0; record that
release minimums must move to the first published contract releases when known,
without inventing numbers or committing local pins.

Prove OpenAPI exposes explicit tagged target/output mode and canonical item/scope
responses. Test invalid/unknown target fields, incompatible native/example formats,
unsupported constrained modes/decoder schema constructs, disallowed operator settings,
remote-services-off and channel incompatibility all fail before enqueue. Test valid
native/example/schema-only routes, exact request forwarding and target isolation.
Verify startup/default policy, existing async endpoint surface and tenant/result access.
Use deterministic offline doubles at inference/transport boundaries without loading
weights. Inspect and run Serve's own configured hooks; do not assume a Makefile.
Review formatter changes, rerun after edits, run whitespace/explicit changed-path audits.

Jobkit final applicable offline subset passed 622 tests, 11 optional/live skips,
12 baseline/model-running deselections; pure orchestrator subset passed 3 with 1 CI
Ray module skip. Native Ruff/uv-lock and six changed production modules' MyPy pass.
Full MyPy retains four proven untouched S3 helper errors because source
S3Coordinates lacks `region`; baseline had those plus two removed extraction errors.
Other proven unchanged baseline failures: connector registry/backends, sandbox Metal,
Redis transaction fixtures. Do not modify unrelated Jobkit/Docling/Core to green
these. Earlier overly broad validation selections initialized cached standard
conversion models; final stage 9 selection omits such model-running tests.
NuExtract3 local/vLLM and Lift live verification remain unrun; installed NuExtract3
LM Studio is incompatible with caller template delivery.

Update the central ledger with exact changed paths, decisions/commands/results,
import/dependency evidence, baseline limitations and remaining live verification.
After stage 10, return control to parent for review and authorized signed commits;
stage 11 remains pending, not automatically dispatched. No push/publication or
production enablement. Final report must be self-contained with paths/results/blockers.
