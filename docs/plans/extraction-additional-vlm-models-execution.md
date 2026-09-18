# Additional extraction VLMs: conversation-sized execution ledger

Updated 2026-09-18. Stages 0–5 and 8–9 are implementation-complete. Stage 1 is committed
at `0e53ddc943ad36e30c424573c37f5ce4c8ebc20d`; stage 2 at
`b0e888f846d82c24b5ba64e587f1dd64bd568c5f`; stage 3 at
`cd550410f9715f730f7ef5bb88ac533580706a19`; stage 4 at
`ef5a8a9305f63e4c4eb3831505a4e71a2056d367`; stage 5 at
`6f7beeab6fdba9f1f6682512986a38df68596be1`; stage 8 at
`c2d5347b7534972b3e8c7282236b7515e09b1e4f`. Stage 9 is signed off in Jobkit at
`a833735833a1299d5a6878433e99bfdcb80e1629`. The user deferred stages 6–7
(Qwen3.5 and Gemma) and authorized proceeding through stages 8–10 instead.
**Next: stage 10**, using
[the stage 10 prompt](extraction-additional-vlm-models-stage10-resumption.md).
NuExtract3 Transformers/vLLM integration is contract-tested, not live verified;
the recorded LM Studio deployment is incompatible with caller template delivery.
No push, publication or production default change is authorized.

## Authority and boundaries

The design and acceptance criteria live in
[the handoff in docling_release](/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md),
including its 2026-09-17 automatic-chunk revision. Part II supersedes the older
standalone implementation proposal. This ledger sequences that work; it preserves the
design/schema/ownership/interfaces. The user's 2026-09-18 instruction supersedes
the old per-stage live-model completion gates: implement documented behavior and
prove offline contracts even when models/endpoints are unavailable or unresponsive.
Live smoke, capacity and quality evidence gate deployment verification/production
enablement separately; they do not block source stages 4–11.

Caller-provided **extraction templates** are mandatory for every model: native
typed `nuextract` guidance for NuExtract; explicit `example_json` guidance for
generic models, alongside any documented schema-only route. An extraction
template specifies the user's output structure; the checkpoint's **chat template**
serializes messages (often with Jinja). No chat-template replacement is required.
Each stage must prove exact caller fields/values reach rendering/API transport and
remain isolated across different targets on a cached extractor. Static targets,
silently dropped fields and schema-only support without explicit templates fail
the implementation gate; formats must remain honestly tagged.

- Docling first, then Jobkit, then Serve. Do not implement model behavior downstream.
- **Stop immediately if a change to docling-core is needed.** Do not switch,
  edit, synchronize, build, or use editable overrides of its dirty checkout.
  Published Core APIs may be consumed without modifying that checkout.
- Preserve unrelated tracked and untracked files. No cleanup of existing artifacts.
- The user authorized parent signed-off commits for stages 8–10 after review,
  including scoped Docling tracking commits. Stage workers do not commit.
  No merges/cherry-picks, pushes, publication, production enablement or external
  callbacks are authorized.
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
Stages 4–7 implement each documented model contract; live verification is separate.

| Stage | Repository | Deliverable | Handoff coverage | State |
|---|---|---|---|---|
| 0 | All | Branch setup, current baseline, this ledger | Baseline | Done |
| 1 | Docling | Target/result types and call-local target preparation | A types, B, D types | Done |
| 2 | Docling | One local/API ordered-content execution contract | C, A model compatibility | Done |
| 3 | Docling | Complete source-to-item SDK path and automatic streaming chunks | Remaining A, D, E | Done |
| 4 | Docling | NuExtract3 integration and offline template/transport contracts | F: NuExtract | Done (implementation); live verification separate |
| 5 | Docling | Lift template integration and documented vLLM constraints | F: Lift | Done (implementation); live verification separate |
| 6 | Docling | Shared Qwen3.5 4B/9B template profile and offline contracts | F: Qwen | Deferred by user |
| 7 | Docling | Gemma template/processor/response integration and offline contracts | F: Gemma | Deferred by user |
| 8 | Docling | Explicit service/client contract and complete Docling docs | G: Docling | Done; signed parent checkpoint recorded |
| 9 | Jobkit | Target forwarding, cached config, durable items and callbacks | G: Jobkit | Done (implementation); parent review/signoff pending |
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

### 4–7. Implement one documented model contract per conversation

Read its Part I prerequisites and Part II F, then verify pinned primary model,
processor and server documentation. Add explicit opt-in presets and only helpers
needed by those contracts; no pipeline branches or verification framework.

- **4, NuExtract3:** native and supported schema/Pydantic-converted templates,
  instructions/thinking/mode controls, documented processor and vLLM transport.
- **5, Lift:** explicit `example_json` templates plus schema-only guidance,
  independent single-page requests, documented EOS/stop settings, local path where
  documented, request-specific vLLM constraints and original-schema validation.
- **6, Qwen:** explicit `example_json` templates plus schema-only guidance;
  one non-thinking profile for 4B/9B with documented channel/processor transport.
- **7, Gemma:** explicit `example_json` templates plus schema-only guidance,
  access/license requirements, visual-token settings, image-before-text order and
  documented response interpretation. Add native parsing only if required.

Implementation exit for each: offline tests capture exact caller fields/values at
real rendering/API boundaries for its documented engines and all claimed channels;
prove two independent absolute selected pages, changed cached-call templates,
original-schema validation and explicit rejection of unsupported modes/constraints.
Existing applicable extraction regressions and required validation must pass.
Opt-in presets describe implemented contracts, not certified deployments. Reject
known incompatible transports (NuExtract3 LM Studio); do not generalize one engine's
results to another. Stages 8–11 consume the exact source implementations regardless
of outstanding live-model verification.

Separately record per-model/engine live verification: not run, incompatible or
verified with exact revision, versions, device/settings, inputs/targets, raw answers,
quality and validation. Weights/downloads/inference require their own authorization.
Fixed-input schema-versus-example quality comparisons remain unrun until authorized
models respond; JSON validity is not extraction accuracy. Published context/output
settings are theoretical bounds or documented examples, not tested capacity. Use
measured deployment limits only when measured; leave blocked live verification open.

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

Run exact-source end-to-end service jobs with deterministic model boundaries, including
independent selected pages, unpaginated text, schema failure and partial input.
Use controlled callback receivers; no real external customer callbacks.
Verify durable artifacts, source identity, terminal status, callback ordering,
tenant boundaries and unchanged conversion behavior. Repeat regression/hooks
against the exact source revisions; retain actual logs/exit codes. Live-model jobs
are a separate authorized verification pass. Report outstanding deployment gates
honestly; source delivery completion does not certify unrun deployments.

## Resume protocol and evidence

Start each conversation by reading this ledger, repo guidance and only the
relevant handoff sections. Recheck branch/HEAD/worktree state and existing code;
do not reset a changed baseline or carry forward unverified package assumptions.
Trace affected callers, implement only the current stage and run its behavioral
implementation gate plus relevant existing regressions. A failed offline contract
gate means the stage is not done; unavailable live models do not block it.

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

### Completed stages 1–3 evidence

The unchanged checkpoint decisions, exact commands and results are retained in
[the stages 1–3 evidence file](extraction-additional-vlm-models-stages1-3-evidence.md).
The parent split this history when stage 9 tracking exceeded the 1,500-line
repository limit; no evidence or golden results were rewritten.

### Current checkpoint: stage 4 — implementation complete

The revised gates above apply to all remaining stages. `nuextract_3` is an explicit
opt-in preset pinned to `c99dc8f5641b866aa0192b6ea78f84bf9f3535f1`, for local
Transformers and generic API configured for vLLM. It preserves native types
(including `verbatim-string`), reuses bounded schema/Pydantic conversion and merges
caller template/instructions with thinking false and structured mode. The shared
loader remains; no custom SDK, Jinja override, additional weight download, dependency
or runtime change.
NuExtract2/legacy retain tokenizer/Qwen vision preparation; NuExtract3 uses the
checkpoint processor renderer and raw images, letting its patch-size-16 processor
own image sizing. Caller image ownership and one-request-per-page policy are unchanged.

Offline coverage extends existing tests: local processor rendering receives exact
native values on text/image/mixed input; source SDK DCLX → actual HTTP JSON carries
native/schema/Pydantic templates on all channels, independently for pages 2–3.
One cached extractor changes templates/instructions without leakage; a valid JSON
answer violating the unchanged original schema fails validation. Named incompatible
API variants are rejected. These are contract tests, not model-generated answers.

| Model/engine | Source implementation | Live verification |
|---|---|---|
| NuExtract3 / Transformers | Contract tested; explicit opt-in | Not run; no weights loaded |
| NuExtract3 / vLLM API | Contract tested; caller configures endpoint | Not run; no server substituted |
| NuExtract3 / installed LM Studio | Not offered by preset; rejected | Incompatible: recorded extraction controls dropped |
| Lift, Qwen3.5, Gemma 4 | Pending stages 5–7 | Not run |

Primary sources rechecked: [pinned model card](https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/README.md),
[pinned processor config](https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/processor_config.json),
[chat template](https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/chat_template.jinja),
[model config](https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/config.json),
[generation config](https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/generation_config.json),
[Transformers processor contract](https://huggingface.co/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template)
and [vLLM chat request protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py).
Config records Transformers 5.5.4; installed 5.16.1 maps `qwen3_5` through the
existing auto-loader. This does not establish a minimum supported version or live
loading success. Pinned native processor is `Qwen3VLProcessor`; remote code is not
needed by these built-in mappings (the model-card example passes trust true).

The 4096 output budget follows the official non-thinking Transformers example;
258048 input tokens is config's 262144 context minus that output budget, a static
upper bound, not measured capacity. API server context limits are deployment-owned;
the client does not configure them. Earlier 8192 LM Studio/256–512 output probes
remain probe settings. Capacity and schema-versus-example quality comparisons are
unrun for Transformers/vLLM and make no claims about extraction accuracy.

Validation results and commands are recorded after the historical investigation.
Stage 4 was signed off at `ef5a8a9305f63e4c4eb3831505a4e71a2056d367`.
Its next Lift implementation checkpoint is recorded below.

### Historical stage 4 live investigation — superseded completion gate

The following records describe the earlier live-only gate. Their raw evidence is
preserved; the revised delivery policy above supersedes their stage-blocking
conclusion. They establish installed LM Studio incompatibility, not a source
implementation blocker.

Verification on 2026-09-17–18 used stage 3 HEAD
`cd550410f9715f730f7ef5bb88ac533580706a19` on
`cau/extraction-api-service-models`, initially clean. The user's deployment
choice was: “we can only do it through lmstudio, that is installed. Take a GGUF
with 4 bit”. Official Q4 weights and the required vision projector were downloaded
and loaded successfully. No Transformers/vLLM inference was substituted.

**At this historical checkpoint, stage 4 was not accepted under the old live gate.** No preset, loader,
transport helper, numeric capacity limit, dependency, production default, Core,
Jobkit or Serve change was made. No stage commit, push or publication was made.
The only changed paths are this ledger and these two bounded artifacts:

- [Exact requests, raw answers and rendered-input evidence](extraction-nuextract3-lmstudio-smoke.json).
- The task-owned stage 5 continuation, now replaced by [stage 6](extraction-additional-vlm-models-stage6-resumption.md).

#### Model, contract and environment evidence

- Source: [numind/NuExtract3](https://huggingface.co/numind/NuExtract3/tree/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1),
  revision `c99dc8f5641b866aa0192b6ea78f84bf9f3535f1`, public/ungated,
  Apache-2.0. Official config names `Qwen3_5ForConditionalGeneration`,
  `qwen3_5`, Transformers 5.5.4. The installed Transformers 5.16.1 auto mapping
  contains this architecture; that is inspection evidence, not local inference
  certification. No remote-code loading or loader replacement was needed/proven.
- Weights: [official NuExtract3-GGUF revision](https://huggingface.co/numind/NuExtract3-GGUF/tree/28a1aae6288c5d50bdf999ca2e572240966d669b)
  `28a1aae6288c5d50bdf999ca2e572240966d669b`.
  `NuExtract3-Q4_K_M.gguf`: 2,783,445,984 bytes, SHA256
  `7ee3c0ee9e5699a4391624ae758487f583f73b3242aa4e73dc2bb33e508d703e`.
  Required `mmproj-NuExtract3-BF16.gguf`: 675,569,184 bytes, SHA256
  `9a506ce43544691f4f56f1533768f43e1887ef69b8cd74b2e7a46c4d9a155a01`.
  The language model is Q4_K_M; the vision projector is BF16, not Q4.
- The embedded GGUF native Jinja template and the SDK-returned native template
  both equal the pinned source byte-for-byte: 6,752 UTF-8 bytes, SHA256
  `31e44d28615d268efdc3dcf59cb59bd2d51714d517455fbad97518d351b84119`.
  The template is referenced by revision/hash rather than copied into this repo.
  A nonempty native `template` sets structured mode itself. Its default is content
  mode and thinking false; the observed LM Studio prompt opens thinking instead.
- Native `verbatim-string` means exact copying; it is not interchangeable with
  paraphrasing `string`. Reused the existing bounded schema/Pydantic converter
  and call-local target preparation during tracing; no new schema semantics were
  inferred. Existing native/conversion/template-ownership tests remain green.
- Runtime: LM Studio `0.4.21+2`, `lms` commit `71bd99c`,
  `llama.cpp-mac-arm64-apple-metal-advsimd@2.40.0`; llama-server reports
  `0.4.1-dev`, build 1, commit `7ceed87`, AppleClang 15.0.0.15000309,
  Darwin ARM64. Device is Apple Metal on macOS ARM64, 64 GiB system memory;
  torch CUDA unavailable/MPS available. The HTTP smoke loaded identifier
  `docling-stage4-nuextract3-q4`, context 8192, parallel 1, GPU max:
  1.57 seconds, reported 3.22 GiB. SDK follow-up used the subsequently loaded
  official identifier `numind/nuextract3`, context 8192, parallel 4, GPU ratio 1.0,
  flash attention true. Both point to the same verified official weights.
- Metadata context length 262144 is not a measured working capacity. The 8192
  context and 256/512 output caps are probe settings, not support/preset limits.
- Tests borrowed Python 3.13.5 from `docling_release/.venv` with this worktree
  on `PYTHONPATH`; published Core 2.96.0 resolves from site-packages. Versions:
  torch 2.14.0, Transformers 5.16.1, Pydantic 2.13.5, jsonschema 4.26.0,
  pytest 9.0.3, polyfactory 3.3.0, huggingface-hub 1.31.0,
  qwen-vl-utils 0.0.14. No environment sync/Core override occurred. Official
  SDK `lmstudio==1.5.0` was installed only into `/tmp/docling-stage4-sdk` for
  bounded feasibility probes; no permanent dependency was added.

#### Real inference and the blocker

Four independent requests to `http://127.0.0.1:1234/v1/chat/completions`
held source text, temperature 0, output cap 256 and streaming false fixed.
They supplied (1) native invoice/total template with instructions/thinking false,
(2) a different buyer-only native template/instructions/thinking false,
(3) total-only template/thinking true, and (4) no template kwargs.
All returned HTTP 200 and identical content/reasoning: an HTML table truncated
at `finish_reason=length`, with 70 prompt, 256 completion and 188 reasoning
tokens. Actual model input logs are identical: `【task】content`, no template or
instruction sections, and an open `<think>`. HTTP acceptance did not apply
`chat_template_kwargs` to the native template.

The official [HTTP chat contract](https://lmstudio.ai/docs/developer/openai-compat/chat-completions)
does not document those kwargs. The [native REST chat contract](https://lmstudio.ai/docs/developer/rest/chat)
exposes reasoning controls but no per-call extraction template/Jinja override.
The [legacy completion endpoint](https://lmstudio.ai/docs/developer/openai-compat/completions)
skips chat formatting but does not establish a multimodal native-template route.
The [GUI template override](https://lmstudio.ai/docs/app/advanced/prompt-template)
is a model default, which does not establish per-call ownership/isolation.
Internal application forwarding of `chatTemplateKwargs` alone was insufficient
evidence; the rendered inputs resolve the actual HTTP behavior.

The official SDK was then tested independently, without changing Docling's API
path. Its [prediction config mapping](https://github.com/lmstudio-ai/lmstudio-python/blob/main/src/lmstudio/_kv_config.py)
supports `config.promptTemplate`/`llm.prediction.promptTemplate`. A one-token
control returned the exact native checkpoint template in the prediction config.
The next probe prepended Jinja assignments for the original native template,
instructions and thinking false and supplied that template per prediction,
temperature 0/output cap 512. Actual input still used content mode/open thinking;
the answer contained reasoning delimiters and the complete HTML table, 70 prompt
and 373 predicted tokens, stopping at EOS. An independent eight-token sentinel
override (`{{ 'DOCLING_STAGE4_SENTINEL' }}`) was echoed unchanged in the result's
prediction config but ignored by the actual prompt, which retained the original
content task, 42 prompt tokens and open thinking. A small SDK transport wrapper
would therefore still fail the required native contract on this installed runtime.
No custom WebSocket stack, engine-guard bypass or generic prompting imitation was
introduced. Load-time/custom runtime configuration remains an unverified future
option, not a demonstrated dynamic transport.

Factual quality and output validity are separate: the truncated HTTP answer
contains the correct invoice/seller but omits the requested final total/buyer
before its length stop. The full SDK table includes the correct source fields,
including final total 32.43, while following the wrong task/output form. None
is a JSON object. These native-template transport probes supplied no standards
`output_schema`; unchanged-schema validation was **not requested**, not passed.
No extraction accuracy score or schema-versus-example advantage is claimed.

Remaining real-model gates were not run: Docling SDK native/converted/Pydantic
targets, image and mixed channels, two independent absolute selected pages/non-1
ranges, honored instruction/thinking variants, cached-call isolation, original
schema validation and fixed-input schema-versus-example comparison without
copying evaluation answers into examples. No NuExtract3 engine/channel is
advertised as supported. NuExtract2/Granite/main behavior remains unchanged.

#### Commands and regression results

Commands were run from `docling-second`; network/model/MLX-dependent operations
used permitted escalation. The first `lms get` proxy attempt timed out around
8 MB; pinned direct downloads succeeded and the explicit proxy retry eventually
completed. Native `lms import` and opening the installed app refreshed the catalog;
the first pre-refresh loads returned model-not-found. No unrelated model or
defaults were changed by these probes. Cached verified artifacts are now under
`/Users/cau/.lmstudio/models/numind/NuExtract3-GGUF`; temporary hard links under
`docling-stage4/NuExtract3-GGUF` do not duplicate the weight bytes. No cleanup of
unrelated local artifacts was performed.

```sh
git branch --show-current
git rev-parse HEAD
git status --short

/Users/cau/.lmstudio/bin/lms get \
  'https://huggingface.co/numind/NuExtract3-GGUF@Q4_K_M' --gguf -y
# Direct fallback, repeated for mmproj-NuExtract3-BF16.gguf:
curl -fL --retry 3 --connect-timeout 20 -C - \
  -o /tmp/NuExtract3-Q4_K_M.gguf \
  https://huggingface.co/numind/NuExtract3-GGUF/resolve/28a1aae6288c5d50bdf999ca2e572240966d669b/NuExtract3-Q4_K_M.gguf
/Users/cau/.lmstudio/bin/lms import /tmp/NuExtract3-Q4_K_M.gguf \
  --user-repo numind/NuExtract3-GGUF -y
open -a 'LM Studio'
/Users/cau/.lmstudio/bin/lms load nuextract3 \
  --identifier docling-stage4-nuextract3-q4 --context-length 8192 \
  --parallel 1 --gpu max -y
# Exact HTTP bodies and actual inputs are preserved in the linked JSON.
/Users/cau/.lmstudio/bin/lms log stream --source model --filter input --json
/Users/cau/Documents/Development/docling_release/.venv/bin/python \
  /tmp/docling-stage4-lmstudio-probe.py

uv pip install --python /Users/cau/Documents/Development/docling_release/.venv/bin/python \
  --target /tmp/docling-stage4-sdk lmstudio
# SDK records preserve config, exact variable prefix, native-template hash,
# raw content/stats and actual inputs; no copied checkpoint template is needed.

CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_stage3.py tests/test_extraction_transformers_model.py \
  tests/test_service_datamodels.py tests/test_api_image_request.py \
  tests/test_build_generation_config.py tests/test_interfaces.py \
  tests/test_input_doc.py tests/test_invalid_input.py \
  -k 'not test_convert_path and not test_convert_stream' \
  > /tmp/docling-stage4-regressions.log 2>&1

UV_NO_SYNC=1 make validate > /tmp/docling-stage4-validate.log 2>&1
git diff --check
git status --short --untracked-files=all
```

Regressions: **331 passed, 4 existing weight skips, 2 existing conversion tests
deselected, 31 warnings**, exit 0, 21.64 seconds. The two conversion/table golden
mismatches were already reproduced on unchanged stage 1 source in stage 2; no
golden rewrite was made. Final `UV_NO_SYNC=1 make validate` passes, exit 0;
applicable conflict/large-file/max-lines hooks pass and source hooks have no
changed files. Existing installed-hook cache metadata warnings are non-fatal.
`git diff --check` passes; the final scope is exactly the three documentation
paths listed above, with no production/test changes.

The old live gate paused at stage 4 here. Under the revised policy this deployment
failure stays recorded while source implementation and stage 5 may proceed.

#### User-confirmed live endpoint follow-up — 2026-09-18

The user confirmed `localhost:1234`, model ID `numind/nuextract3`. Rechecked the
live deployment outside the localhost-restricted sandbox: the exact identifier
is already loaded, official Q4_K_M/qwen35 with vision, context 8192, parallel 4,
flash attention true, native MTP speculative decoding enabled. The model catalog
reports reasoning options off/on with default on. App version remains 0.4.21+2;
`lms runtime ls` confirms selected GGUF Metal runtime 2.40.0. No model download,
reload, switch, defaults change or other-model operation was performed.

One bounded fresh request to `http://localhost:1234/v1/chat/completions` used
`model=numind/nuextract3` and added explicit `mode=structured` to the original
native template/instructions/`enable_thinking=false` controls. It returned HTTP
200 with the exact requested model identifier, the same length-stopped HTML
answer and 70/256/188 prompt/completion/reasoning tokens. The actual rendered
input is byte-for-byte equal to the original content-task/open-thinking input;
native template and instructions remain absent. The catalog's default thinking
setting explains the default behavior; it does not establish application of the
per-call controls. The explicit target mode itself was also ignored.

The exact request/raw response, selected model's current catalog/load config and
rendered-input event are appended under `user_endpoint_follow_up` in the existing
smoke JSON. Prior records are preserved. Rechecked documented native REST/SDK
controls and existing shared converter/adapters: REST reasoning alone cannot
supply the native extraction template, and the recorded SDK sentinel already
proves its per-prediction template override is ineffective in actual input on
this runtime. Endpoint clarification resolves availability, while the dynamic
native-template transport blocker remains. No native schema validation was
requested by this transport probe; no additional model gate is marked passed.

```sh
/Users/cau/.lmstudio/bin/lms ps --json
curl -fsS --connect-timeout 5 --max-time 10 http://localhost:1234/api/v1/models
/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  '/Applications/LM Studio.app/Contents/Info.plist'
/Users/cau/.lmstudio/bin/lms runtime ls
# Replayable HTTP body and actual model-input event are in user_endpoint_follow_up.
# Used urllib.request with timeout120 and an owned lms model/input logger;
# terminated only that logger after the single request.
UV_NO_SYNC=1 make validate > /tmp/docling-stage4-validate.log 2>&1
git diff --check
git status --short --untracked-files=all
```

The documentation-only follow-up retains the 331-pass regression checkpoint;
tests were not repeated without production changes. Required `make validate` and
whitespace/scope checks pass again. This follow-up remained incomplete under the old live gate. The later revised
policy and current implementation checkpoint above supersede that gate.

### Stage 4 implementation validation — revised delivery policy

The current source pass loaded no weights, ran no live inference and made no
additional model downloads/runtime changes. The historical smoke JSON is unchanged
(SHA256 `ca79ae727b165058f41419a4f138faebddb03bd96d9c22e20b43d38cad7ba2f0`).
The task-owned obsolete stage 4 prompt was replaced by the next stage 5 prompt.

```sh
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_stage3.py tests/test_extraction_transformers_model.py \
  tests/test_extraction_api.py -k 'nuextract3 or ordered_local_content' \
  > /tmp/docling-stage4-contracts.log 2>&1

CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_stage3.py tests/test_extraction_transformers_model.py \
  tests/test_service_datamodels.py tests/test_api_image_request.py \
  tests/test_build_generation_config.py tests/test_interfaces.py \
  tests/test_input_doc.py tests/test_invalid_input.py \
  -k 'not test_convert_path and not test_convert_stream' \
  > /tmp/docling-stage4-regressions.log 2>&1

UV_NO_SYNC=1 make validate > /tmp/docling-stage4-validate.log 2>&1
git diff --check
git status --short --untracked-files=all
```

Focused contracts: **10 passed, 140 deselected**, exit 0. Applicable regressions:
**339 passed, 4 existing weight skips, 2 known conversion tests deselected,
31 warnings**, exit 0, 15.89 seconds. No golden changes. Tests used the recorded
borrowed Python/site-packages environment with source `PYTHONPATH` and permitted
escalation for native MLX import initialization, without syncing anything.
`make validate` required escalation for existing UV/hook cache access; Ruff first
fixed import order and reformatted one test. Reviewed those edits; final repeated
validation passes (including ty/tach/max-lines), exit 0. Whitespace/scope checks
pass. Existing installed-hook cache metadata warnings remain non-fatal.

Changed source/tests: `extraction_options.py`, `prompt_utils.py`,
`test_extraction_api.py`, `test_extraction_stage3.py`,
`test_extraction_transformers_model.py`; docs: this ledger, `extraction.md` and the
next stage 5 prompt, plus preserved historical smoke evidence. No Core, Jobkit,
Serve, environment, loader or production-default change. Stage 4 implementation
is complete; the live verification matrix stays open. Parent review/signoff commit
is the next authorized delivery step, then stage 5.


## Stage 5 checkpoint — Lift source implementation

Started from clean `cau/extraction-api-service-models` at signed stage 4 commit
`ef5a8a9305f63e4c4eb3831505a4e71a2056d367`. Stage 5 is implementation-complete
under the revised offline-contract delivery policy, pending parent review/signoff.
No weights were downloaded/loaded, no live inference was run, and no environment,
runtime, dependency, loader, production default, Core, Jobkit or Serve was changed.

Added opt-in `lift`, pinned to model revision
`3129597900eb6f84fb4f2c0b240f9a7cfddae595`, using the existing `generic_chat`
preparation and Transformers/vLLM API adapters. Explicit caller `example_json`
fields/values and instructions reach processor rendering/API messages intact;
schema-only guidance works as well. Lift requires an explicit output schema in
both output modes. Shared preset capability `requires_output_schema` rejects
example-only targets and legacy/final-prompt image wrappers before inference;
existing models keep their previous behavior. Native `nuextract` remains an
unsupported dialect for generic chat. No schema is inferred from an example.

The published processor is `Qwen3VLProcessor` with `Qwen2VLImageProcessorFast`,
and the model is `Qwen3_5ForConditionalGeneration`. Lift's documented local loader
uses exactly the existing `AutoModelForImageTextToText`/`AutoProcessor` path; no
custom loader/preprocessor is justified. The reference HF dependency is
Transformers >=5.2.0; the borrowed installed 5.16.1 maps the architecture. This
is compatibility inspection, not a load test. The checkpoint's existing chat
renderer supports `enable_thinking=False`, carried in both documented transports.

Pinned tokenizer IDs 248044/248046 correspond to `<|endoftext|>`/`<|im_end|>`.
Both reach local `GenerationConfig`; the served preset sends both stop strings
through `ApiModelConfig`, verified in the actual HTTP payload. Local stop metadata
recognizes either EOS. Context 262144 and official output setting 12384 justify
249760 local input tokens as a **static unmeasured upper bound**. No measured
memory/capacity/device eligibility follows from those values; server context is
operator-owned and may be smaller.

Constrained-vLLM mode sends a fresh request-specific schema through the shared
`response_format` path. The original schema remains unchanged for guidance and
validation: Lift's reference nullable-leaf rewrite and silent schema-compilation
fallback are deliberately not copied. Shared bounded subset preflight rejects
unsupported assertions before HTTP; explicit prompt-only generation can use the
full original schema. Lift's recommendation to avoid enum/union/ref/
`additionalProperties` complexity is quality/simplicity advice, not evidence that
every such construct is unsupported by vLLM. Existing enum/nullable/local-ref/
boolean-additionalProperties behavior remains, with backend compilation still to
verify live; provider rejection cannot trigger a prompt-only retry.

Extended existing behavioral tests rather than adding a separate model framework:
source DCLX → SDK → actual HTTP JSON covers every channel, independent absolute
pages 2–3, schema-only and exact explicit examples, changed schemas/templates/
instructions on one cached extractor, preserved caller mappings and valid JSON
that fails original-schema validation. Shared local rendering exercises text,
image and mixed ordered content with cached-call isolation. Both local EOS values,
legacy-wrapper/schema/dialect rejection, dynamic constraints and provider rejection
are covered. Unpaginated Lift text remains one document-scoped request; existing
ownership/finally, result-status, context and legacy regressions remain green.

| Model/engine | Source implementation | Live verification |
|---|---|---|
| Lift / Transformers | Contract tested; explicit opt-in | Not run; no weights loaded |
| Lift / vLLM API | Contract tested; explicit constrained mode available | Not run; no server substituted |
| Lift / named LM Studio, Ollama, OpenAI | Not offered by preset; rejected | Not run; no contract claimed |

Independent-page extraction usefulness, schema-versus-example field accuracy,
actual loading, per-device memory/capacity and backend-specific schema compilation
remain **not run**, separately from this completed source stage. The reference
joint-page workflow and its benchmark do not certify Docling's independent-page
behavior. NuExtract3 live verification stays unchanged: local/vLLM not run;
installed LM Studio incompatible. Historical raw smoke evidence is unchanged,
SHA256 `ca79ae727b165058f41419a4f138faebddb03bd96d9c22e20b43d38cad7ba2f0`.

Primary sources were browsed and exact revision metadata/configs rechecked:

- [Pinned Lift model card](https://huggingface.co/datalab-to/lift/blob/3129597900eb6f84fb4f2c0b240f9a7cfddae595/README.md),
  [model config](https://huggingface.co/datalab-to/lift/blob/3129597900eb6f84fb4f2c0b240f9a7cfddae595/config.json),
  [processor config](https://huggingface.co/datalab-to/lift/blob/3129597900eb6f84fb4f2c0b240f9a7cfddae595/processor_config.json),
  [tokenizer/chat template](https://huggingface.co/datalab-to/lift/blob/3129597900eb6f84fb4f2c0b240f9a7cfddae595/tokenizer_config.json)
  and [generation config](https://huggingface.co/datalab-to/lift/blob/3129597900eb6f84fb4f2c0b240f9a7cfddae595/generation_config.json).
- Reference source revision `4ff031b8c83b44bb123d7eda42907b22ec1e1e56`:
  [HF loader/EOS](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/lift/model/hf.py),
  [vLLM schema transport](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/lift/model/vllm.py),
  [prompt](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/lift/prompts.py),
  [output setting](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/lift/settings.py),
  [dependencies](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/pyproject.toml)
  and [vLLM launcher](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/lift/scripts/vllm_launcher.py)
  (reference container v0.22.0, not a measured deployment or client minimum).
- [Transformers Qwen3.5 contract](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
  and [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
  confirm processor multimodal input and JSON Schema `response_format` transport.
- [Modified OpenRAIL-M weights license](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/MODEL_LICENSE)
  differs from Apache-2.0 source code: funding/revenue and competing-service use
  restrictions, attribution and share-alike provisions require operator review and
  deployment allow-listing. This source stage grants no deployment authorization.

### Stage 5 validation and next continuation

Borrowed Python 3.13.5/Transformers 5.16.1 from
`/Users/cau/Documents/Development/docling_release/.venv/bin/python`, with this
checkout's `PYTHONPATH`. Published Core resolves from that environment's
site-packages, not its dirty checkout. No sync/editable override was used.

```sh
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_stage3.py tests/test_extraction_transformers_model.py \
  tests/test_extraction_api.py \
  -k 'model_templates or lift or ordered_local_content or dynamic_vllm or constrained_subset or unpaginated_source' \
  > /tmp/docling-stage5-contracts.log 2>&1

CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_stage3.py tests/test_extraction_transformers_model.py \
  tests/test_service_datamodels.py tests/test_api_image_request.py \
  tests/test_build_generation_config.py tests/test_interfaces.py \
  tests/test_input_doc.py tests/test_invalid_input.py \
  -k 'not test_convert_path and not test_convert_stream' \
  > /tmp/docling-stage5-regressions.log 2>&1

UV_NO_SYNC=1 make validate > /tmp/docling-stage5-validate.log 2>&1
git diff --check
git status --short --untracked-files=all
shasum -a 256 docs/plans/extraction-nuextract3-lmstudio-smoke.json
```

Focused contracts: **43 passed, 134 deselected**, exit 0. Applicable regressions:
**366 passed, 4 existing weight skips, 2 known conversion tests deselected,
31 warnings**, exit 0, 15.05 seconds. No golden changes. Tests used permitted
escalation for existing native MLX initialization; validation used existing
UV/hook cache access. Ruff adjusted test imports/formatting; reviewed those edits,
corrected the wrapper test's captured boundary and removed an unnecessary shared-
preset mutation from its EOS test. Final focused contracts pass again (43 passed,
134 deselected); repeated validation until all hooks (including ty/tach/max-lines)
passed. Installed-hook cache metadata warnings
remain non-fatal. Whitespace/scope checks pass.

Changed paths: `docling/datamodel/extraction_options.py`,
`docling/models/extraction/prompt_utils.py`, `tests/test_extraction_api.py`,
`tests/test_extraction_stage3.py`, `tests/test_extraction_transformers_model.py`,
`docs/plans/extraction.md`, this ledger, and the task-owned stage 5 continuation
renamed/replaced by [the self-contained stage 6 prompt](extraction-additional-vlm-models-stage6-resumption.md).
That stage 6 continuation is historical and deferred; current routing is stage 9
after stage 8 signoff. Stage 6 must preserve explicit caller examples and the separate live-verification
policy for both Qwen sizes. Parent review and the authorized signoff commit come
before dispatching the next sequential stage worker.


## Stage 8 checkpoint — explicit service/client contract

Started from `cau/extraction-api-service-models` at signed stage 5 HEAD
`6f7beeab6fdba9f1f6682512986a38df68596be1`, preserving the parent's uncommitted
stage 8 routing/prompt. Parent review/signoff completed at signed HEAD
`c2d5347b7534972b3e8c7282236b7515e09b1e4f` (Signed-off-by verified), with a clean
source checkout before stage 9. Stages 6–7 remain deferred; stage 9 is implemented,
stage 10 is next after parent stage 9 review/signoff, stage 11 stays pending.
No Core, Jobkit, Serve, dependency/environment/runtime/default change, live
inference, model downloads, commits or pushes occurred.

Replaced unreleased service `template` directly with `target: ExtractionTarget`,
sharing SDK tagged templates/output schemas/instructions. Strict options reject
old and unknown fields; output mode is explicit `prompt_only` (default) or
`schema_constrained`. Kept model/channel/page-range settings and the request's
independent output-storage `target`. Downstream admission/preparation remains
responsible for model/engine/schema compatibility before queueing.

Replaced branch-only `ExtractionResultItem.pages` with JSON-safe
`ExtractionDocumentResult`: required original source index, expanded source URI,
filename, status/errors and canonical `ExtractionItem`s. No runtime input/backend
serialization or duplicated item type. Existing task unions and inline responses
use those envelopes; conversion DTOs and released SDK compatibility are unchanged.
Page/document scopes, raw answers, validation failure/not-run records, stop reasons,
token/usage metadata and source identity survive JSON task/result round-trips.

There was no generic extraction client submission path. Added only
`submit_extract(ExtractSourcesRequest)` to sync/async clients, reusing current
transport, credential restoration, job polling/watch/wait and result handling.
In-body results are typed `ExtractDocumentResponse`; storage responses stay
`RawServiceResult`, preserving artifact contracts/destinations. Different caller
schema/template/instructions/output-mode payloads reach the actual HTTP JSON on
one client without stale values. No new SDK remote conversion infrastructure.

Updated shipped extraction/slim/service-client references, the extraction notebook
and `extraction.md` to source/target, `items`/`scope`, original-schema validation,
automatic absolute-page/document requests and capability limits. Notebook code
was statically parsed (13 code cells); stale outputs were cleared, no notebook
inference was run. Only NuExtract2/Granite/NuExtract3/Lift are documented as
implemented. NuExtract3/Lift local/vLLM live verification stays unrun; installed
NuExtract3 LM Studio stays incompatible. Qwen3.5/Gemma remain deferred.

External handoff/proposal files were not rewritten: Part II and its automatic
chunk revision remain authoritative; the standalone proposal is retired as an
alternative authority. An eventual external pointer/retirement edit is still
needed in its dirty checkout under separate scope, not this stage.

### Stage 8 validation

Borrowed Python 3.13.5 from `docling_release/.venv`, using this source `PYTHONPATH`
and published Core at that environment's site-packages. No sync/build/override.
First sandboxed collection aborted during known eager native MLX initialization
(exit 134); permitted escalation ran the offline tests without loading weights.
An incomplete GoogleDrive fixture was corrected to provide its required refresh
token; production was not changed to accommodate the fixture.

```sh
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_service_contract.py tests/test_service_datamodels.py \
  > /tmp/docling-stage8-contracts.log 2>&1

CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_service_contract.py tests/test_extraction_templates.py \
  tests/test_extraction.py tests/test_extraction_api.py \
  tests/test_extraction_text_channel.py tests/test_extraction_dclx.py \
  tests/test_extraction_vlm_streaming.py tests/test_extraction_stage3.py \
  tests/test_extraction_transformers_model.py tests/test_service_datamodels.py \
  tests/test_api_image_request.py tests/test_build_generation_config.py \
  tests/test_interfaces.py tests/test_input_doc.py tests/test_invalid_input.py \
  tests/test_service_callbacks.py tests/test_service_client_sdk_unit.py \
  tests/test_service_client_payload_fidelity.py tests/test_service_client_fake_service.py \
  -k 'not test_convert_path and not test_convert_stream and not test_polymorphic_option_fields_are_serialized_as_any' \
  > /tmp/docling-stage8-regressions.log 2>&1

mkdir -p /tmp/docling-stage8-baseline
git archive HEAD docling | tar -x -C /tmp/docling-stage8-baseline
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/tmp/docling-stage8-baseline \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_service_client_payload_fidelity.py \
  -k test_polymorphic_option_fields_are_serialized_as_any \
  > /tmp/docling-stage8-baseline.log 2>&1

UV_NO_SYNC=1 make validate > /tmp/docling-stage8-validate.log 2>&1
git diff --check
git status --short --untracked-files=all
```

Focused contracts: **61 passed**, exit 0. Applicable broad regressions:
**580 passed, 4 existing weight skips, 3 deselected, 3559 warnings**, exit 0,
31.33 seconds. The third exclusion is conversion-only
`test_polymorphic_option_fields_are_serialized_as_any`: identical failure on
untouched signed stage 5 source (1 failed, 7 deselected), naming `model_spec` on
CodeFormulaVlmOptions/PictureDescriptionVlmEngineOptions/VlmConvertOptions as
existing SerializeAsAny debt. Its neighboring payload tests pass. Other two
exclusions and weight skips are the prior known baseline; no golden changes.
Repeated required `make validate` after reviewing Ruff import/formatter edits;
all applicable hooks including ty/tach/coverage/max-lines pass. Existing installed
hook-cache metadata warnings remain non-fatal. Whitespace/scope checks pass.

Changed source: `docling/datamodel/service/{options,requests,responses,__init__}.py`,
`docling/service_client/{client,_async_client}.py`. Tests:
`tests/test_service_datamodels.py`, new `tests/test_extraction_service_contract.py`.
Docs: shipped references `{extraction,slim-packaging,service-client}.md`,
`docs/examples/extraction.ipynb`, `docs/plans/extraction.md`, this ledger,
preserved parent stage 8 prompt and new stage 9 prompt. Stage 9 must use the exact
parent-signed stage 8 source; downstream source paths/dependency strategy and
artifact/callback contracts are explicit in its self-contained continuation.


## Stage 9 checkpoint — Jobkit target forwarding and durable items

Started from clean tracked Jobkit `cau/extract-endpoint` HEAD
`51a6339d715045ebca9a2b8e206c75e62444961d`, preserving all existing untracked
material. Parent stage 8 signoff was required and verified at Docling
`c2d5347b7534972b3e8c7282236b7515e09b1e4f`. Implementation is complete; parent
review passed and Jobkit is signed off at `a833735833a1299d5a6878433e99bfdcb80e1629`.
The parent commit ran Ruff successfully and skipped only the full MyPy hook
(`SKIP=system`) after reproducing its untouched S3 errors; scoped MyPy passed.
The Docling tracking commit precedes stage 10.
No worker commits, pushes, dependency sync/build/pins, Core edits/overrides, Serve
edits, downloads, extraction-model inference or external callbacks occurred.

The manager forwards `target=` per call and returns `DocumentExtractionResult`s.
Explicit service `output_mode` overrides preset/custom mode through Docling option
validation without mutating the caller config. Existing cache keys remain stable
pipeline configuration/formats only; target schemas/templates/instructions are
absent from those keys. Two distinct tagged templates/schemas/instructions reuse
one extractor; channel or mode changes produce separate cached configurations.
No model-specific preparation or schema logic was added downstream.

The shared result builder now serializes `ExtractionDocumentResult` with original
source index, expanded URI and canonical items. Task unions and stored JSON retain
absolute page/document scopes, validation passed/failed/not-run records, raw answers,
item errors and inference usage. Runtime input/backend owners are omitted.
Source expansion previously used expanded-document ordinals; it now retains the
original `task.sources` index across connector expansion. Only extraction uses this
identity-aware helper; ordinary conversion expansion is unchanged. S3/Azure
presigned processors key artifact lists by the existing frozen `SourceIdentity`,
preventing cross-document leakage when different expanded URIs share an original
index. Extraction temporary JSON filenames use the iteration ordinal.

In-body/remote/presigned destinations and source-hashed keys remain. Partial counts,
upload failures and callback ordering survive: uploads precede each authoritative
DOCUMENT_COMPLETED, followed by UPDATE_PROCESSED; existing Ray SET_NUM_DOCS and
durable terminal callback paths are preserved. Jobkit's result shim exports the
canonical extraction envelope/task types. README contains a tagged target example,
result/item migration and the eventual dependency strategy. Local/RQ extraction
remains unsupported; no new execution surface was added.

### Stage 9 environment and dependency evidence

Used Jobkit's existing `.venv/bin/python` (Python 3.12.7), because the borrowed
Docling 3.13 environment lacked Ray/RQ/Redis/MyPy/pre-commit. Exact imports:

- Docling: `/Users/cau/Documents/Development/docling-second/docling/__init__.py`.
- Jobkit: `/Users/cau/Documents/Development/docling-jobkit/docling_jobkit/__init__.py`.
- Published Core: `/Users/cau/Documents/Development/docling-jobkit/.venv/lib/python3.12/site-packages/docling_core/__init__.py`.

Installed metadata: Core 2.92.0, Docling slim 2.124.0, Jobkit 3.5.0, Ray 2.55.1,
pytest 9.0.3, MyPy 1.20.2. Installed Docling/Jobkit distributions are stale and
are bypassed by source `PYTHONPATH`; this is existing-environment contract evidence,
not a freshly synchronized supported release matrix. Core is consumed from
site-packages only. No dirty Core checkout access/change, sync or editable override.
Jobkit's declared `docling-slim[standard]>=2.128.0,<3.0.0` stays unchanged until the
first published release containing this contract is known; raise that minimum
before downstream release without inventing a version or committing local pins.

### Stage 9 validation

All commands below ran from `/Users/cau/Documents/Development/docling-jobkit` with
`CI=1 HF_HUB_OFFLINE=1` and
`PYTHONPATH=/Users/cau/Documents/Development/docling-jobkit:/Users/cau/Documents/Development/docling-second`.

```sh
.venv/bin/python -m pytest -q tests/test_extraction_manager.py tests/test_presigned_target_results.py

.venv/bin/python -m pytest -q --maxfail=0 tests \
  --ignore=tests/test_local_orchestrator.py --ignore=tests/test_rq_orchestrator.py \
  --ignore=tests/test_ray_orchestrator.py \
  -k 'not test_threaded_request_reaches_conversion and not test_builtin_source_connectors_registered and not test_builtin_target_connectors_registered and not test_options_validator and not test_backend_mapping_standard_and_vlm and not test_options_cache_key and not test_image_pipeline_uses_vlm_pipeline_when_requested and not test_get_s3_connection and not test_resync_retries_after_concurrent_finalize and not test_resync_defers_after_three_conflicts'

.venv/bin/python -m pytest -q tests/test_local_orchestrator.py \
  tests/test_rq_orchestrator.py tests/test_ray_orchestrator.py \
  -k 'test_on_result_fetched_local or test_prepare_convert_sources_threads_max_file_size or test_on_result_fetched_rq or test_metadata_field_backward_compatibility or test_expire_result or test_on_result_fetched_ray or test_enqueue_rejects_s3_source_without_remote_target or test_create_deployment'

.venv/bin/mypy docling_jobkit/convert/extraction_manager.py \
  docling_jobkit/convert/extraction_results.py docling_jobkit/convert/source_expansion.py \
  docling_jobkit/connectors/s3/presigned_target_processor.py \
  docling_jobkit/connectors/azure_blob/presigned_target_processor.py \
  docling_jobkit/datamodel/result.py

UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/jobkit-stage9-uv-cache \
  .venv/bin/pre-commit run --files README.md \
  docling_jobkit/convert/extraction_manager.py docling_jobkit/convert/extraction_results.py \
  docling_jobkit/convert/source_expansion.py docling_jobkit/datamodel/result.py \
  docling_jobkit/connectors/s3/presigned_target_processor.py \
  docling_jobkit/connectors/azure_blob/presigned_target_processor.py \
  tests/test_extraction_manager.py tests/test_presigned_target_results.py
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/jobkit-stage9-uv-cache \
  .venv/bin/pre-commit run uv-lock --files pyproject.toml uv.lock

git diff --check
git diff --name-only
git status --short --untracked-files=all
```

Focused: **35 passed**, 402 warnings, exit 0 (before the final equivalent Lift
fixture/callback-order strengthening; final broad run includes both files).
Final broad applicable offline subset: **622 passed, 11 existing optional/live
skips, 12 explicit deselections, 1590 warnings**, exit 0, 7.33 seconds.
Additional pure orchestrator subset: **3 passed, 1 existing CI Ray module skip,
26 deselected**, exit 0. Scoped MyPy: **6 production files pass**.
Native Ruff formatting/lint and uv-lock hooks pass. The native full MyPy hook
checks 126 files and fails with **4 errors only in untouched
`connectors/s3/helper.py`**: updated source `S3Coordinates` has no `region`, producing
2 attribute and 2 boto overload errors. Untouched archived Jobkit baseline has
those same 4 plus 2 old extraction-contract errors (6 total); stage 9 removes the
latter. No helper changes were made solely to green unrelated debt. First hook
attempt failed uv-cache permissions; the temporary offline cache resolved it.

Baseline proof used `git archive HEAD docling_jobkit` extracted under
`/tmp/jobkit-stage9-baseline`, with only the first `PYTHONPATH` entry replaced by
that archive. Exact baseline test selections:

```sh
.venv/bin/python -m pytest -q tests/test_connector_factory.py tests/test_options.py \
  -k 'test_builtin_source_connectors_registered or test_builtin_target_connectors_registered or test_options_validator or test_backend_mapping_standard_and_vlm'
.venv/bin/python -m pytest -q --maxfail=0 tests/test_options.py tests/test_s3_helper.py \
  tests/test_ray_lifecycle_counters.py \
  -k 'test_options_cache_key or test_options_cache_key_with_presets or test_image_pipeline_uses_vlm_pipeline_when_requested or test_get_s3_connection or test_resync_retries_after_concurrent_finalize or test_resync_defers_after_three_conflicts'
.venv/bin/mypy /tmp/jobkit-stage9-baseline/docling_jobkit
```

First baseline selection: 4 identical failures, 42 deselected, exit 1 (connector
registry expectations and conversion backend expectations). Second: 5 identical
failures, 2 identical Redis fixture errors, 18 deselected, exit 1 (3 sandbox Metal
cache checks, 2 S3 region failures, 2 Redis transaction fixtures). Broader initial
run with live orchestrator modules and threaded conversion excluded:
622 passed, 11 skipped, 1 deselected, 9 failed, 2 errors, exit 1. The final selected
run excludes those proven unrelated failures and the model-executing conversion.

Validation scope correction: the earlier broad command was
`.venv/bin/python -m pytest -q tests --ignore=tests/test_local_orchestrator.py --ignore=tests/test_rq_orchestrator.py --ignore=tests/test_ray_orchestrator.py`.
It was interrupted at `test_threaded_request_reaches_conversion` after standard
conversion initialization: **348 passed, 8 skipped, 4 failed**, exit 2. An earlier
additional orchestrator `-k` also included
`test_clear_converters_clears_caches`, `test_chunker_manager_shared_across_workers`,
`test_worker_cms_tracking`, and `test_clear_converters_clears_worker_cache` alongside
the final pure selection. It completed **6 passed, 1 failed (Redis), 1 CI skip,
22 deselected**, exit 1; those three Local tests performed cached standard PDF
conversion. Thus no zero-model-initialization claim is made. No extraction-model
inference or new weight download was performed; final validation selection omits
these model-running tests. Initial focused fixture selected unsupported Granite
text; it was corrected to a compatible model, and the existing Ray test now
supplies the required `tenant_id` argument.

Logs: `/tmp/jobkit-stage9-{focused,offline-final,unit-orchestrators-final,hooks-final,lock-hook,scoped-mypy,baseline-tests,baseline-additional,baseline-mypy}.log`.
Whitespace and explicit changed-path audits pass. Changed Jobkit paths:
`README.md`; `docling_jobkit/convert/{extraction_manager,extraction_results,source_expansion}.py`;
`docling_jobkit/connectors/{s3,azure_blob}/presigned_target_processor.py`;
`docling_jobkit/datamodel/result.py`;
`tests/{test_extraction_manager,test_presigned_target_results}.py`.
Docling tracking paths: this ledger and the self-contained stage 10 continuation.
No unrelated tracked/untracked changes or dependency files were modified.

Next: parent review/signoff, then stage 10 only using
[the stage 10 prompt](extraction-additional-vlm-models-stage10-resumption.md).
Stage 6–7 Qwen3.5/Gemma remain deferred, stage 11 remains pending. NuExtract3
Transformers/vLLM and Lift live verification remain unrun; installed NuExtract3
LM Studio remains incompatible. Production enablement is separate.
