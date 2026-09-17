# Additional extraction VLMs: conversation-sized execution ledger

Updated 2026-09-17. Stages 0 and 1 are complete. The user subsequently
authorized the signed-off stage 1 commit, titled
`feat: add extraction target contracts and preparation`. Next stage: **2**.

## Authority and boundaries

The design and acceptance criteria live in
[the handoff in docling_release](/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md),
including its 2026-09-17 automatic-chunk revision. Part II supersedes the older
standalone implementation proposal. This ledger sequences that work; it does
not introduce a competing design or relax any acceptance gate.

- Docling first, then Jobkit, then Serve. Do not implement model behavior downstream.
- **Stop immediately if a change to docling-core is needed.** Do not switch,
  edit, synchronize, build, or use editable overrides of its dirty checkout.
  Published Core APIs may be consumed without modifying that checkout.
- Preserve unrelated tracked and untracked files. No cleanup of existing artifacts.
- No commits, merges/cherry-picks creating commits, pushes, publication, production
  enablement, or external callbacks without separate authorization.
- Preserve main's SDK boundaries only; replace unreleased interfaces directly.
- No public grouping, chunk inputs, joint-page inference, scope registry,
  model subclasses per checkpoint, or schema inference from examples.
- Each stage leaves existing supported calls working. Do not expose `target=`
  on the SDK until the execution/result/chunk path is functional in stage 3.

## Verified starting state

All three implementation worktrees have no tracked modifications. Existing
untracked files remain in place. Per the user's clarification, use `docling-second`
for Docling rather than moving its branch to `docling_release`:

| Worktree under `/Users/cau/Documents/Development` | Implementation branch | Local HEAD | Remote main |
|---|---|---|---|
| `docling-second` | `cau/extraction-api-service-models` | `477448cef49fe4a613292eb4595646d7f86cec59` | `629440ee795fda854a9f9667ca8c54b0013a5ad6` |
| `docling-jobkit` | `cau/extract-endpoint` | `51a6339d715045ebca9a2b8e206c75e62444961d` | `a1f828dfa30ddd6b609016d427d8b57ab02c7922` |
| `docling-serve` | `cau/extract-endpoint` | `e9820f6069122cd1c771d1147942a0a7a9d4b8ad` | `a9a64c58b57f3c7673ee53dff074312b69b8b974` |

Remote heads were verified with `git ls-remote`; no fetch, pull, rebase or merge
was performed. Jobkit is 10 commits and Serve 9 commits ahead of their remote
extraction branches; these local changes are part of the starting baseline.
The current remote main still has the legacy SDK/result/constructor surfaces
listed in Part II. Its HEAD is newer than the handoff's original main baseline.

The chosen Docling branch already contains the extraction implementation at
`9f4a726549e4bb0107d0137eae6c27f05c2dac51` plus its two service-model commits.
No branch relocation or sibling-delta port is needed. Stage 8 updates those
existing service models directly. `docling_release` was switched from main to
`cau/extraction-with-api` during initial setup and remains there; it holds the
canonical handoff, but is not the production implementation worktree.

Downstream dependency declarations currently point to released Docling, not
this checkout. Jobkit requires `docling-slim[standard]>=2.128.0,<3.0.0`; Serve
requires `docling-slim[...]>=2.127.0,<3.0.0` and `docling-jobkit[...]>=3.7.0,<4`.
Do not mistake installed released packages for the updated source contract.

Baseline, from `docling_release`, using its existing Python 3.13.5 environment:

```sh
.venv/bin/python -m pytest -m 'not ml_vlm' \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py
```

Result: **34 passed**, exit 0, 5.94 seconds, outside the restricted sandbox.
The first sandboxed run aborted during eager `mlx_whisper` hardware imports
before collection, exit 134. Marker exclusion does not prevent those imports.
Do not patch extraction to hide this environment issue. No weights were loaded
by these baseline tests. The full extraction suite and downstream suites have
not been run in this planning stage.

After selecting `docling-second`, reran the same four files plus
`tests/test_service_datamodels.py` from that worktree against its source:

```sh
PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest \
  -m 'not ml_vlm' tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_service_datamodels.py
```

Result: **71 passed**, exit 0, 4.96 seconds, with 27 existing deprecation warnings,
outside the sandbox. This borrowed the Python 3.13.5 environment without syncing
it; source paths in the test output are under `docling-second`. Its own existing
`.venv` points to Python 3.14 and was not used for these tests. Prepare/verify the
stage 1 environment before dependency work rather than assuming either environment
matches the eventual updated lockfile.

## Delivery stages

One stage is the default budget for one conversation. The first three divide
the handoff's A–E by executable boundaries rather than exposing an incomplete
public API after slice A. Stage 3 intentionally keeps chunk ownership, failure
scope, result status and SDK projection together: they must agree end to end.
Stages 4–7 separate real model verification from shared plumbing.

| Stage | Repository | Deliverable | Handoff coverage | State |
|---|---|---|---|---|
| 0 | All | Branch setup, current baseline, this ledger | Baseline | Done |
| 1 | Docling | Target/result types and call-local target preparation | A types, B, D types | Done |
| 2 | Docling | One local/API ordered-content execution contract | C, A model compatibility | Next |
| 3 | Docling | Complete source-to-item SDK path and automatic streaming chunks | Remaining A, D, E | Pending |
| 4 | Docling | NuExtract3 integration and native/conversion smoke evidence | F: NuExtract | Pending |
| 5 | Docling | Lift single-page integration and explicit vLLM constraints | F: Lift | Pending |
| 6 | Docling | Shared Qwen3.5 profile, 4B functional and 9B capacity checks | F: Qwen | Pending |
| 7 | Docling | Gemma processor/response integration and smoke evidence | F: Gemma | Pending |
| 8 | Docling | Explicit service/client contract and complete Docling docs | G: Docling | Pending |
| 9 | Jobkit | Target forwarding, cached config, durable items and callbacks | G: Jobkit | Pending |
| 10 | Serve | Admission/OpenAPI/endpoint migration and authorization | G: Serve | Pending |
| 11 | All | Exact-source end-to-end verification and final audit | G: closure | Pending |

### 1. Define the contract and prepare targets

Read Part II A, B and D's type declarations. Primary files: `extraction.py`,
`extraction_options.py`, `prompt_utils.py`, new `template_utils.py`, packaging
and focused template tests. Define the public target/tagged template, strict
page/document scopes, item and envelope models; keep legacy DTOs unchanged.

Implement schema normalization, deep-copy ownership, Draft 2020-12 preflight,
object-only output contract and validation with no external reference retrieval.
Add `jsonschema` to `extract-core` while preserving slim/API-only imports.
Evaluate the official NuMind converter against the required fixtures and its
license/footprint/loss reporting; keep only the chosen implementation. Do not
defer that decision behind an abstraction or silently drop schema branches.
Prepare native/generic guidance through one private prepared-target record,
with distinct chat and processor options. Legacy sample serialization remains
explicitly separate; preserve `polyfactory` for main's class-template behavior.

Exit: supported nested/array/enum/nullable/local-reference schemas prepare
deterministically; unsupported paths/dialects/drafts fail before inference;
examples never become schemas; caller mappings are unchanged; scope/item JSON
round-trips reject malformed/unknown kinds and contain no image payload.
Existing extraction regressions remain green. New SDK execution is not exposed yet.

### 2. Unify both inference adapters

Read Part II C and A's model compatibility requirements. Trace every caller of
`process()`/`process_images()` before changing their contracts. Primary files:
`base_model.py`, extraction adapters and prompt helpers, extraction options,
and the extraction-specific API request helper.

Both engines consume ordered content and the prepared target; neither receives
chunks/scopes. Remove NuExtract-only execution guards. Route options at the
correct rendering/preprocessing boundary; merge fresh call-local mappings and
reject request-owned field collisions. Introduce explicit prompt-only versus
documented vLLM constrained output, with subset preflight and no fallback.
Preserve conversion transports, shared generation metadata/context checks and
main's local constructors/imports/image wrappers. Update PR-only process callers
atomically, feeding legacy preparation until stage 3 wires the new SDK path;
do not retain an old/new branch-only signature shim.

Exit: boundary tests capture correct text/image/mixed payloads and local
processor arguments; templates/instructions cannot leak across cached calls;
dynamic constraints and collision rejection work; local/named API variants
reject unverified constrained mode; provider rejection never retries prompt-only.
Main image wrappers and remote authorization/error regressions still pass.

### 3. Wire the complete SDK, result and chunk path

Read all remaining A, D and E. Primary files: `document_extractor.py`, base/VLM
extraction pipelines, extraction DTOs/options and existing SDK/text/DCLX/streaming
tests. Restore main-shaped inline constants and retain the legacy boundary.

Add keyword-only `target=` without shifting positional arguments; normalize
once per public call and warn once for legacy calls. Canonical pipeline results
are document envelopes; only the outer legacy entry projects page items.
Use overloads for the SDK distinction. New and legacy calls share inference.

One pipeline-local generator determines reliable pagination before channels:
one request per absolute selected page, or one text-only unpaginated document.
Establish scope before content loading. Consume predictions before advancing;
release owned images/backend resources in `finally`, close on early exits,
and preserve borrowed DCLX ownership. Failed selected pages cannot disappear.
Apply unchanged-schema validation once, retain raw answers and metadata, and
derive truthful validation/partial/failure status, including length/timeout cases.

Exit: exercise source-to-item behavior for all channels, non-1 page ranges,
unpaginated rejection rules, unattributed text, image-only pages, load/inference/
tokenization/parse/schema failures, timeout and context overflow. Prove one-live-
page resource bounds during lazy prediction consumption and no call-state on
cached instances. Positional SDK calls, legacy DTO construction/serialization,
pipeline `execute(template=...)`, inline overrides and slim imports still work.
Run the full applicable extraction regression set, not only new tests. Exercise
existing NuExtract2/Granite through the completed path before new presets.

### 4–7. Verify one model behavior per conversation

For each stage, read its Part I prerequisites and Part II F. First recheck the
official model/server contracts and available deployment; then add preset data
and only a helper justified by observed behavior. Do not add pipeline branches.

- **4, NuExtract3:** verify native and supported converted templates, instructions,
  thinking control, text/image/mixed content and two independent page requests.
- **5, Lift:** verify useful independent single-page results, termination settings,
  local execution where advertised, request-specific vLLM constraints and original-
  schema validation. Unsupported constraints fail; prompt-only is intentional.
- **6, Qwen:** one preparation profile for both sizes; verify clean non-thinking
  output and advertised channels on 4B; separately prove 9B loading/capacity.
- **7, Gemma:** verify access/license, visual-token settings, image-before-text
  ordering, clean response interpretation, and each advertised channel/engine.
  Add response parsing only if real output requires it.

Exit for each: real inference evidence for every advertised engine/capability,
not only payload tests or auto-loader mappings. Record exact model revision,
library/server versions, device, input/target, settings, raw output, extracted-
value accuracy and validation outcome. Choose capacity limits from evidence.
Perform the handoff's schema-versus-example comparison with fixed independent
inputs; JSON validity alone is not extraction quality.

Access, license, hardware or deployment blockers must be recorded per model/
engine. Keep unverified presets unavailable; do not weaken chunk policy or call
them supported. Unblocked stages may proceed, but blocked gates remain open
and final completion cannot be claimed for those capabilities. Request missing
deployment choices/authority rather than downloading large weights by assumption.

### 8. Finish Docling's service/client boundary

Read G and inspect the existing service models on this branch. Replace their
unreleased fields directly with the new target and durable
document/item/scope/validation wire DTOs. Preserve input document identity while
using JSON wire DTOs, not runtime `InputDocument` backends. Update requests,
responses, exports and applicable sync/async client paths together. The old
wire shape receives no compatibility aliases or versioning.

Exit: SDK-derived and wire schemas have identical validation outcomes; client
payload/result round-trips and service DTO tests work without Jobkit/Serve changes;
no public chunk/grouping input exists; no obsolete extraction fields survive.
Update examples, notebook, shipped extraction/slim references and `extraction.md`;
retire the sibling proposal as an alternative authority. Docling is ready for
downstream consumers with exact test/import evidence and capability limitations.

### 9. Migrate Jobkit

Primary files: `convert/extraction_manager.py`, `convert/extraction_results.py`,
affected orchestrator/task/result code and `tests/test_extraction_manager.py`.
Trace consumers of the old service result, including artifact construction.
Run against the exact updated Docling source before editing dependency declarations.

Forward `target=`, persist JSON document envelopes/items and preserve source
identity/expansion, in-body/remote outputs and callback/billable semantics.
Cache stable configuration only; no targets, validators or chunks in cache keys
or cached extractor attributes. No model/schema preparation in Jobkit.

Exit: cache reuse with distinct task targets is isolated; page/document/schema-
failure artifacts round-trip; source expansion, partial status, storage failures
and callback ordering regressions pass. Run affected source/result/callback tests,
not just the manager tests. Record exact source import paths and dependency strategy.

### 10. Migrate Serve

Primary files: `app.py`, `policy.py`, `settings.py`, manager configuration wiring
and `tests/test_service_policy.py`. Preserve the existing async source extraction
endpoint surface and conversion APIs. Run with exact updated Docling/Jobkit sources.

Exit: explicit target/scope/output-mode schemas appear correctly in OpenAPI;
malformed targets, incompatible native formats, unsupported constrained settings,
disallowed presets/custom configs/engines/formats and disabled remote services
are rejected before enqueue. Reuse Docling preflight rather than duplicate its
converter. Exercise default/startup policy checks as well as requests. Verify
tenant/result authorization and endpoint/result serialization regressions.

### 11. Close the cross-repository gates

Run real end-to-end service jobs for each enabled preparation behavior, including
independent selected pages, unpaginated text, schema failure and partial input.
Use controlled callback receivers; no real external customer callbacks.
Verify durable artifacts, source identity, terminal status, callback ordering,
tenant boundaries and unchanged conversion behavior. Repeat regression/hooks
against the exact source revisions; retain real logs/exit codes for smoke and
integration evidence. Report outstanding deployment gates honestly.

## Resume protocol and evidence

Start each conversation by reading this ledger, repo guidance and only the
relevant handoff sections. Recheck branch/HEAD/worktree state and existing code;
do not reset a changed baseline or carry forward unverified package assumptions.
Trace affected callers, implement only the current stage and run its behavioral
gate plus relevant existing regressions. A failed gate means the stage is not done.

For Docling changes run `make validate`, review formatter changes and repeat
until clean. The checkout has many unrelated untracked artifacts: hooks must
not rewrite them. If necessary run full validation in a clean isolated checkout
containing the stage delta and run scoped hooks on actual changed paths; record
the validation location and any unrelated/full-repo blockers. Use each downstream
repo's own configured hooks and focused tests rather than assuming it has the
Docling Makefile. Never update generated reference data to hide a regression.

Before ending, update the stage state and the small checkpoint below. Record
actual changed paths, exact commands/exit codes, decisions and outstanding gates.
Keep source/tests authoritative; link to actual logs rather than copying code or
the design into handoff prose. Do not create empty evidence trees or new tracking
frameworks. Without commit authorization, identify the working-tree delta and
do not mistake HEAD for an immutable implementation checkpoint.

### Current checkpoint

- Completed: stages 0 and 1. Stage 1's behavioral gates and applicable existing
  extraction regressions pass. Branch remains `cau/extraction-api-service-models`,
  implementation base `477448cef49fe4a613292eb4595646d7f86cec59`. The stage 1
  commit includes this ledger; resolve its ID with `git log -1` after committing.
  The user authorized signoff after the implementation gates passed; no push
  was requested.
- Starting state rechecked: no tracked changes; the existing untracked ledger
  was preserved and updated. `status.showUntrackedFiles=no` hides untracked files
  in default status, so the final audit used `--untracked-files=all` and
  `git ls-files --others --exclude-standard`.
- Changed paths:
  - `docling/datamodel/extraction.py`
  - `docling/datamodel/extraction_options.py`
  - `docling/models/extraction/prompt_utils.py`
  - `docling/models/extraction/template_utils.py` (new)
  - `tests/test_extraction_templates.py` (new)
  - `tests/data/extraction/template_conversion.json` (new)
  - `pyproject.toml`
  - `uv.lock`
  - `docs/plans/extraction-additional-vlm-models-execution.md` (existing untracked ledger)
- Delivered: tagged example/native target and Pydantic schema shorthand; strict
  discriminated page/document scopes; durable JSON extraction items and document
  envelope. Legacy DTOs, SDK signatures, execution and inference adapters are
  unchanged. Items reuse prediction token/stop types and accept existing typed
  API usage while retaining provider usage as JSON mappings for stable round-trips.
- Preparation: one call-owned private record carries owned target mappings,
  validator, prompt, distinct chat/processor options and the future optional
  constraint schema. The constraint schema remains unset until stage 2.
  New preparation uses the internal helper selector; legacy preparation keeps
  sample serialization, old prompts and no inferred validation contract.
- Validation: `jsonschema>=4.18.0,<5.0.0` is declared in `extract-core`.
  Draft 2020-12 schema preflight requires object output, leaves format assertions
  disabled and fails closed on external retrieval. Non-recursive document-local
  JSON Pointer references are supported, including escaped keys and root `$id`;
  nested resources, anchors, dynamic references, other drafts and vocabularies
  fail with paths. Referenced annotations are checked as schemas when referenced.
- Conversion: primitives, fixed nested objects, homogeneous arrays, multi-value
  string enums, optional/nullable fields and local references prepare deterministically.
  Original schemas/descriptions/constraints remain unchanged in the validator and
  guidance. Semantic native types are never inferred from schema descriptions.
  Unsupported structural branches, singleton/numeric enums, unions, tuples,
  dynamic keys, assertion siblings of references and untyped required properties
  fail with paths. Explicit native guidance can retain the same validation schema
  when automatic conversion is unsupported.
- Boundaries preserved: no SDK `target=` execution, public chunk input, output
  constraints execution, model enablement, weights, Core modifications/overrides,
  Jobkit/Serve edits, pushes or publication. The subsequent user request
  authorized only the signed-off stage 1 commit.
- Outstanding stage 1 blockers: none. Sandbox MLX-import and uv-cache restrictions
  were handled by authorized execution outside the sandbox, without code workarounds.
- Next: stage 2, unify the inference adapters around the prepared target, route
  options and add explicit backend constraints. Both adapters start consuming
  this record in stage 2; source-to-item SDK execution remains stage 3.

### Stage 1 converter decision and validation evidence

Evaluated NuMind's official `numind==0.4.0` wheel, linked by its
[official repository](https://github.com/numindai/nuextract-platform-sdk) and
[PyPI metadata](https://pypi.org/pypi/numind/0.4.0/json). Wheel SHA-256:
`528121ad700c460b80cceb572efabe5f4af0e0dc64fe4800ab75656a2c9d07ed`;
932,731 bytes. Its packaged LICENSE is MIT (the file retains a placeholder
copyright year). No licensing blocker was found for evaluation. The wheel was
installed only in `/tmp/docling-numind-eval-venv`, not a repository environment.

All three checked-in supported fixtures matched expected templates without
mutating input. All six unsupported fixtures produced path-bearing loss reports.
The current API returns `schema_status` and `incompatibilities` with schema paths,
not the README's older `template, dropped_branches, descriptions` tuple. With
`omit_unsupported_branches=True`, reports were `partially_converted`; those
results would need rejection. No instance-guided union selection or output
repair was used. The isolated SDK install resolved 25 distributions; direct
runtime dependencies include aiohttp/aiohttp-retry, json-repair, orjson,
python-dateutil and urllib3 alongside Pydantic/jsonschema. **Decision: use one
bounded local converter**, avoiding the platform-client and repair footprint.
No official converter or second implementation is shipped.

Exact commands/results (all from `docling-second` unless an absolute script is
shown):

```sh
# Existing baseline, outside sandbox: 71 passed, 27 existing warnings, exit 0.
PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest \
  -m 'not ml_vlm' tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_service_datamodels.py

# Official converter evaluation, exit 0; fixture results linked below.
uv venv --python /Users/cau/Documents/Development/docling_release/.venv/bin/python \
  /tmp/docling-numind-eval-venv
uv pip install --python /tmp/docling-numind-eval-venv/bin/python \
  /tmp/numind-0.4.0-py3-none-any.whl
/tmp/docling-numind-eval-venv/bin/python /tmp/docling-evaluate-numind.py \
  > /tmp/docling-numind-evaluation.log

# Dependency resolution, exit 0; no environment sync.
uv lock
uvx --from uv==0.8.3 uv lock
uv lock --locked

# Final gates + regressions, outside sandbox: 144 passed, 4 skipped,
# 27 existing warnings, exit 0, 12.82 seconds.
CI=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_transformers_model.py tests/test_service_datamodels.py \
  > /tmp/docling-stage1-regressions.log 2>&1

# Repository validation, outside sandbox: all applicable hooks passed, exit 0.
UV_NO_SYNC=1 make validate > /tmp/docling-stage1-validate.log 2>&1

# Diff audit, exit 0.
git diff --check
```

The four skips are existing real weight-loading extraction tests guarded by
`CI=1`; all local adapter boundary tests ran without weights. The 59 new target
and scope cases passed, including portable SDK/wire validation, no coercion/default
filling, native dialect rejection, empty example arrays, nonfinite JSON rejection,
non-mutation and call isolation. These gates exercise preparation and result
contracts, not later-stage inference or pipeline status mapping.

The test interpreter is Python 3.13.5 from `docling_release/.venv`; source imports
are from `docling-second`. Installed Pydantic is 2.13.5, jsonschema 4.26.0,
pytest 9.0.3, polyfactory 3.3.0 and published docling-core 2.96.0. Core's import
path is under that environment's `site-packages`; no dirty Core checkout was used.
The local Python 3.14 environment was inspected and used for repository tools
without syncing it. Initial sandbox baseline and template runs aborted on eager
`mlx_whisper` imports (exit 134); the baseline was then verified outside sandbox.
An initial tool invocation could not access uv's cache (exit 2).

Both resolver versions proposed unrelated lock marker normalization (and pinned
uv proposed a revision downgrade). Kept only the resolved `docling-slim` dependency
metadata delta, preserving all other lock entries and revision 3. The resulting
four-addition lock passes `uv lock --locked` and the repository's pinned uv-lock
hook. No dependency versions changed. Initial new tests found a usage-union
round-trip failure (50 passed, 2 failed); fixed the DTO mapping rather than relaxing
that test. Formatter edits were limited to changed Python paths. Hook-cache
metadata warnings were non-fatal; no unrelated files were rewritten.

Actual logs (temporary local evidence):
[converter results](/tmp/docling-numind-evaluation.log),
[final behavioral/regression results](/tmp/docling-stage1-regressions.log),
[repository validation](/tmp/docling-stage1-validate.log).

Resume prompt (only this file path needs to be carried into a new conversation):

> Continue the next incomplete stage in
> `/Users/cau/Documents/Development/docling-second/docs/plans/extraction-additional-vlm-models-execution.md`.
> Implement that stage, verify its gates and update the checkpoint. Preserve
> unrelated work; stop if docling-core needs changes. Do not commit or publish.
