# Continue stage 8: Docling extraction service and client contracts

Implement stage 8 only in `/Users/cau/Documents/Development/docling-second`,
branch `cau/extraction-api-service-models`. Stage 5 is signed off at
`6f7beeab6fdba9f1f6682512986a38df68596be1`; stage 4 at
`ef5a8a9305f63e4c4eb3831505a4e71a2056d367`. Recheck branch, HEAD, worktree,
environment and all callers. The ledger and this prompt have the parent's
uncommitted routing update; preserve it.

Read `AGENTS.md`, `docs/plans/extraction-additional-vlm-models-execution.md`
and relevant A/D/G sections of the authoritative revised handoff at
`/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md`.
The standalone proposal is superseded. Latest user instructions override old
live-model completion gates: implement documented behavior with offline contract
tests; record live verification separately. Caller-provided extraction templates
are mandatory functionality. The user deferred stages 6–7 (Qwen3.5/Gemma) and
authorized stages 8–10 next; do not integrate the deferred models or stage 11.

Replace unreleased extraction options' bare `template` field directly with an
explicit `ExtractionTarget` containing tagged template, output schema and
instructions. Keep source/range/model/channel configuration and expose output
mode consistently. Distinguish the request's existing output-storage `target`
from the extraction guidance target in options; preserve artifact destinations.
No public chunk/grouping input, Python classes, validators or schema inference.
Do not retain aliases for branch-only wire shapes; preserve main conversion APIs
and legacy SDK compatibility.

Move extraction service responses to JSON-safe document envelopes with canonical
items, scopes, validation status, raw answers, errors and inference metadata.
Preserve source/document identity; do not serialize runtime InputDocument or
backend objects. Reuse existing JSON-safe item types rather than duplicate them.
Trace exports, task result unions, serialization and sync/async client paths.
Update applicable clients and tests together. Generic client submission may be
reused if it already handles extraction; do not create redundant infrastructure.

Test equivalent SDK/wire target validation, explicit templates, original-schema
failures, page/document scopes, request/result round-trips, source identity,
client payload/endpoint handling, unknown/old fields and slim/API-only imports.
No model weights or responding server are needed. Update the shipped extraction
and slim references, extraction notebook/examples and `docs/plans/extraction.md`
to source/target and result.items/item.scope usage. Only existing implemented
presets (NuExtract2, Granite, NuExtract3, Lift) may be documented as implemented;
local/vLLM runtime verification for new models remains unrun, installed NuExtract3
LM Studio is incompatible. Do not rewrite external dirty handoff/proposal files;
record any external retirement needed, with the ledger making authority explicit.

Preserve unrelated files. Stop if Core needs edits; no dirty Core checkout access
for builds/sync/overrides. No Jobkit/Serve edits in this stage, dependency sync,
large downloads, live inference, runtime or production-default changes, push,
publication, commits or subagents. Parent reviews and makes each signed-off commit.
Use Ponytail full: existing helpers/stdlib, minimal correct diff, no registries or
speculative abstractions.

Use the verified borrowed Python 3.13.5 environment in docling_release with this
source on PYTHONPATH and published Core from site-packages. Native MLX imports
require permitted escalation outside the sandbox. Run relevant service/client
tests and applicable extraction regressions (last 366 passed, four existing weight
skips, two known conversion exclusions). Preserve goldens. Run
`UV_NO_SYNC=1 make validate`, inspect edits and repeat until clean; audit scope and
`git diff --check`. Do not let hooks rewrite unrelated files.

Update the execution ledger with actual decisions, changed paths, exact commands
and results, separate live-verification status, and current routing. Mark stage 8
implementation done only when contract gates pass. Generate a self-contained
stage 9 prompt for `/Users/cau/Documents/Development/docling-jobkit`, branch
`cau/extract-endpoint`, using exact updated Docling source without permanent local
path pins. Stages 6–7 remain deferred. Notify parent of progress about every minute.
