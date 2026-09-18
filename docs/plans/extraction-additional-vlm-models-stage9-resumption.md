# Continue stage 9: Jobkit extraction target forwarding and durable items

Implement **stage 9 only** in `/Users/cau/Documents/Development/docling-jobkit`,
branch `cau/extract-endpoint`. Read its `AGENTS.md` and applicable instructions,
then recheck branch, HEAD, worktree and environment before editing. The read-only
starting baseline on 2026-09-18 was
`51a6339d715045ebca9a2b8e206c75e62444961d`, with no tracked changes and extensive
untracked material to preserve. Do not create or move a worktree unnecessarily.

Read the Docling ledger at
`/Users/cau/Documents/Development/docling-second/docs/plans/extraction-additional-vlm-models-execution.md`
and relevant Part II G plus A/D contract sections in the authoritative external
handoff at
`/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md`.
Part II supersedes the standalone proposal. The user deferred Qwen3.5/Gemma
stages 6–7 and authorized 8–10. Stages 1–5 and 8 are implemented; stages 6–7 remain deferred. Require the parent's
stage 8 review/signoff checkpoint before starting this downstream stage, and
record its exact HEAD. Documented behavior and offline contracts are completion
gates; live-model verification is separate. No live server, weights, downloads,
inference, stage 10/11 work or production-default change is required/authorized.

Use the exact updated Docling source from
`/Users/cau/Documents/Development/docling-second` on
`cau/extraction-api-service-models` (stage 5 predecessor
`6f7beeab6fdba9f1f6682512986a38df68596be1`; resolve stage 8's signed HEAD from the
ledger/current source). Trace all extraction request/options/result callers,
artifact construction, source expansion and callback paths before editing.
Read the actual source types, not an older installed Docling package.

Docling stage 8 replaced `ExtractDocumentsOptions.template` with
`target: ExtractionTarget`, containing `output_schema`, tagged
`ExtractionTemplate(format="nuextract" | "example_json", value={...})`, and
optional instructions. `output_mode` is explicitly `prompt_only` (default) or
`schema_constrained`; source page range, channel and preset/custom configuration
remain. `ExtractSourcesRequest.target` still selects output storage, independently
of `request.options.target` extraction guidance. Preserve artifact destinations.
No branch-only aliases, grouping/chunk inputs, Python classes/validators or schema
inference are accepted. Core owns preparation, dialect/schema conversion and
response validation; Jobkit only forwards the target and stable options.

New SDK target calls return `DocumentExtractionResult.input/status/errors/items`.
The input owner is runtime-only. Durable wire DTO `ExtractionDocumentResult`
replaces `ExtractionResultItem` with required `source_index` (zero-based original
source index), `source_uri` (expanded document URI), `filename`, `status`, `errors`,
and canonical `items: list[ExtractionItem]`. Items carry page/document scopes,
validation status (`not_requested`, `not_run`, `passed`, `failed`), extracted JSON,
raw answer, item errors and inference metadata. Absolute page scopes must survive
storage; unpaginated text is document-scoped. Do not serialize runtime backends
or duplicate item types. Old service `pages`/bare-template shapes have no aliases;
main's legacy SDK `template=`/`ExtractionResult.pages` compatibility is unchanged.

Primary Jobkit paths:
`docling_jobkit/convert/extraction_manager.py`,
`docling_jobkit/convert/extraction_results.py`, affected task/orchestrator/result
code and `tests/test_extraction_manager.py`. The current manager/results use
`options.template`/`ExtractionResult.pages`; migrate directly to `target=` and
`items`. Forward output mode through stable configuration, with explicit service
options governing generation mode. Cache only stable model/engine/channel/mode
configuration: never task targets, schemas, validators or chunks. Prove cached
extractor reuse isolates two different caller templates/instructions/schemas.
Only implemented presets NuExtract2/Granite/NuExtract3/Lift may be assumed;
NuExtract3 local/vLLM and Lift live verification remain unrun, installed NuExtract3
LM Studio is incompatible. Do not implement Qwen3.5/Gemma downstream.

Preserve source expansion/identity, in-body and remote artifact destinations,
partial status, storage failures and billable/callback ordering. Serialize the
same document/item envelope in task results and extraction JSON artifacts.
Check callbacks and exported task unions, not only manager tests. Update scoped
Jobkit examples/docs/tests. Stop if Docling Core edits are required; never sync,
build or create editable overrides of its dirty checkout. No Serve edits here.
Do not mutate the external dirty handoff/proposal; the Docling ledger records
that its proposal remains retired as authority.

Environment: the last Docling checks borrowed
`/Users/cau/Documents/Development/docling_release/.venv/bin/python` (Python 3.13.5)
with source `PYTHONPATH`, published Core from that environment's site-packages,
and no sync. Verify Jobkit dependencies before choosing that or its own existing
environment; no dependency sync or persistent local source pin is authorized.
Use, for example,
`PYTHONPATH=/Users/cau/Documents/Development/docling-jobkit:/Users/cau/Documents/Development/docling-second`
for exact source testing and print imported package/Core paths. Jobkit currently
declares `docling-slim[standard]>=2.128.0,<3.0.0`; record the eventual release
strategy without fabricating a release number or committing local path pins.
Native MLX initialization can require permitted escalation. Offline tests must
not download/load weights. Jobkit had no Makefile at the baseline: inspect and
run its native configured validation hooks rather than blindly invoking Docling's
`make validate`. Review formatter edits and repeat validation until clean;
run `git diff --check` and an explicit changed-path audit.

User execution constraints: one agent at a time, GPT-5.6sol/high; no subagents,
commits, pushes, publication or callbacks to external systems. Parent reviews and
makes each signed-off commit. Use Ponytail full: trace completely, reuse existing
helpers/stdlib, keep the smallest correct diff and meaningful offline checks.
Send bounded substantive progress about every minute. Update the Docling ledger
with actual paths/decisions, exact commands/results, dependency/import evidence,
remaining live verification, and a self-contained stage 10 prompt for
`/Users/cau/Documents/Development/docling-serve`, branch `cau/extract-endpoint`
(baseline `e9820f6069122cd1c771d1147942a0a7a9d4b8ad`). Preserve stages 6–7 as deferred
and stage 11 as pending. Report the stage 9 outcome and any blockers to parent.
