# Extraction stages 1–3: completed checkpoint evidence

Extracted unchanged from the execution ledger after signed stage 8 checkpoint
`c2d5347b7534972b3e8c7282236b7515e09b1e4f` to keep the active ledger within
the repository line limit. These are historical decisions and validation results;
[current routing and authority](extraction-additional-vlm-models-execution.md)
remain in the active ledger.

### Stage 1 checkpoint (historical)

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

### Current checkpoint: stage 2

- Location: `/Users/cau/Documents/Development/docling-second` only. Branch
  `cau/extraction-api-service-models`, stage 2 implementation base
  `0e53ddc943ad36e30c424573c37f5ce4c8ebc20d`. The subsequent user request
  authorized a signed-off stage 2 commit; resolve its ID with `git log -1`.
  Before edits, explicit
  `git status --short --untracked-files=all` and
  `git ls-files --others --exclude-standard` were empty. No unrelated files
  needed isolation from mutating hooks. Only stage 2 paths are authorized for
  staging/commit; no push or publication.
- Authority: reread this worktree's `AGENTS.md`, stage 2 above, Part I's target,
  conversion and option-ownership constraints, and revised handoff Part II C/A.
  The superseded standalone proposal was not used. Traced both adapters,
  `SupportsContentExtraction`, all production `process()`/`process_images()`
  callers, legacy prompt preparation, wrappers and their branch tests before edits.
- Delivered: both inference adapters consume ordered `ContentItem` requests plus
  `_PreparedTarget`. Removed NuExtract-only content guards. Every extraction
  pipeline channel now calls `process()` with fresh legacy preparation; SDK
  signatures, pipeline/result/page semantics and chunk production remain unchanged.
  No old/new branch-only process-signature shim survives.
- Local compatibility: restored the main inline constructor's optional positional
  `prompt_style`, and the `NuExtractTransformersModel` import/constructor through
  thin forwarding. Image wrappers still accept PIL/numpy images, RGB conversion,
  shared/per-image final prompts and mismatched-count rejection; final prompts are
  never wrapped twice. The optional pipeline no-template prompt retains its prior
  meaning. Main-shaped inline constants remain the explicitly sequenced stage 3
  boundary restoration; this stage does not change `vlm_model_specs.py`.
- Routing/ownership: generic local rendering uses the processor; NuExtract retains
  the verified tokenizer/Qwen vision path. Chat options are supplied during rendering;
  visual options during preprocessing. Granite's existing `do_pad=True` is preset
  data. API chat options merge model defaults, model API defaults and engine params
  shallowly into owned call-local mappings, then insert dynamic template/instructions.
  Ordinary model/temperature/token overrides keep engine precedence. Static request-
  owned messages/templates/instructions/output controls fail explicitly, including
  values concealed by an engine override. Processor input collisions fail before
  rendering/preprocessing. No target, validator or constraint is stored on an engine.
- Output mode: `ExtractionVlmOptions.output_mode` defaults to `prompt_only`.
  `schema_constrained` deliberately opts `engine_type=API` into the documented
  vLLM `response_format={type: json_schema, json_schema: {name, schema}}` contract.
  It requires an explicit output schema; local and named API variants reject it.
  Named variants also reject unverified vLLM chat kwargs instead of assuming option
  compatibility from an OpenAI envelope. Shared conversion engine options and
  `api_image_request()` are unchanged. Provider rejection propagates through the
  existing error mapping; no retry removes constraints or changes the output mode.
- Decoder subset: typed nested objects, homogeneous arrays, primitive/enumerated
  values, single-type/null type lists, nullable `anyOf`, required declared properties,
  boolean schemas/additionalProperties, annotations and nonrecursive local references.
  References are inlined into a fresh derived schema; schema resource metadata and
  definitions are omitted after inlining. Scalar bounds, patterns/formats, tuple/
  unique-item assertions, arbitrary unions/composition, dynamic-key schemas and
  assertion siblings of references fail with schema paths. Unsupported schema
  assertions are never silently dropped. Prompt-only may use valid schemas outside
  this decoder subset; original guidance and validator schemas stay unchanged.
- Preset validation: the inherited factory assigns overrides after construction.
  Extraction revalidates its final options so output-mode/engine combinations cannot
  bypass preflight. The shared conversion preset factory was not changed.
- Environment: Python 3.13.5 from `docling_release/.venv` with imports from
  `docling-second`; torch 2.14.0, Transformers 5.16.1, Pydantic 2.13.5,
  jsonschema 4.26.0, pytest 9.0.3 and polyfactory 3.3.0. Published Core 2.96.0
  resolves under that environment's `site-packages`. Local Python 3.14 tools used
  without syncing dependencies. No Core checkout/override, Jobkit or Serve edits,
  dependency/lock changes, model presets/enablement or weight downloads.
- Behavioral gates: captured actual shared HTTP payloads and local processor
  arguments for text/images/mixed ordering, templates/instructions and two independent
  cached calls; proved fresh nested mappings, dynamic constraints, collision preflight,
  explicit mode/engine/schema rejection and no fallback after provider compilation
  rejection. Main image constructors/wrappers, slim imports, remote authorization,
  empty/filtered/failed API behavior, generation overrides/token counts/stop reasons,
  context checks, DCLX and resource-streaming regressions pass.
- Stage 2 blockers: none. Real served vLLM/model inference remains a stage 4–7
  smoke gate, not a support claim from these payload tests. Two unrelated conversion
  golden failures reproduced against unchanged stage 1 source; no goldens or
  conversion implementation were changed.
- Next: stage 3 only, complete SDK target forwarding, envelopes/items/validation,
  main-shaped inline constants and automatic streaming chunks. Do not expose SDK
  `target=` until that complete source-to-result path is functional.

Changed paths for stage 2:

- `docling/datamodel/extraction_options.py`
- `docling/models/base_model.py`
- `docling/models/extraction/api_extraction_model.py`
- `docling/models/extraction/transformers_extraction_model.py`
- `docling/models/extraction/nuextract_transformers_model.py` (restored main import)
- `docling/models/extraction/prompt_utils.py`
- `docling/models/extraction/template_utils.py`
- `docling/utils/api_extraction_request.py` (replaces deleted `api_nuextract_request.py`)
- `docling/pipeline/extraction_vlm_pipeline.py` (legacy preparation/call routing only)
- `tests/test_extraction_api.py`
- `tests/test_extraction_text_channel.py`
- `tests/test_extraction_transformers_model.py`
- `tests/test_extraction_dclx.py`
- `tests/test_extraction_vlm_streaming.py`
- `docs/plans/extraction-additional-vlm-models-execution.md`

### Stage 2 commands and evidence

Commands run from `docling-second` unless a different working directory is stated:

```sh
# Start-state verification: branch/HEAD above; clean tracked/untracked state, exit 0.
pwd
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
git ls-files --others --exclude-standard

# Environment check: versions/path above, exit 0.
/Users/cau/Documents/Development/docling_release/.venv/bin/python -c 'import sys,importlib.metadata as m,docling_core; print(sys.version); print(docling_core.__path__); print({p:m.version(p) for p in ["torch","transformers","pydantic","jsonschema","pytest","polyfactory"]})'

# Baseline, outside sandbox: 144 passed, 4 skipped, 27 warnings; exit 0, 12.97s.
CI=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_transformers_model.py tests/test_service_datamodels.py \
  > /tmp/docling-stage2-baseline.log 2>&1

# Final gates/regressions, outside sandbox: 225 passed, 4 skipped, 2 deselected;
# 27 existing warnings, exit 0. HF_HUB_OFFLINE prevents downloads.
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_transformers_model.py tests/test_service_datamodels.py \
  tests/test_api_image_request.py tests/test_build_generation_config.py \
  tests/test_interfaces.py -k 'not test_convert_path and not test_convert_stream' \
  > /tmp/docling-stage2-regressions.log 2>&1

# Isolated baseline for unrelated conversion failures; no worktree/commit created.
mkdir -p /tmp/docling-stage2-baseline-source
git archive HEAD | tar -x -C /tmp/docling-stage2-baseline-source
# Working directory: /tmp/docling-stage2-baseline-source.
# 2 failed, 4 deselected, 4 warnings; exit 1, 8.11s.
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/tmp/docling-stage2-baseline-source \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_interfaces.py -k 'test_convert_path or test_convert_stream' \
  > /tmp/docling-stage2-interface-baseline.log 2>&1

# Required repository validation, outside sandbox: applicable hooks pass, exit 0.
# Rerun after final source/ledger changes; no unrelated rewrites or dependency sync.
UV_NO_SYNC=1 make validate > /tmp/docling-stage2-validate.log 2>&1

# Final diff audit: exit 0.
git diff --check
```

The sandbox baseline aborted on eager MLX import, exit 134, before collection;
rerunning outside the sandbox passed. `UV_NO_SYNC=1 uv run ruff ...` initially
failed to access uv's cache, exit 2; the existing `.venv/bin/ruff` tools then
formatted/linted only changed paths, exit 0. No code workaround or environment
sync was used. The first migrated regression run had three streaming failures
because the test image fake did not meet the existing PIL `ImageContentItem`
contract; updated the fake's base class while retaining all resource assertions.
Later boundary runs caught the preset-validation bypass and corrected test-only
assumptions about raw usage mappings, the existing RuntimeError provider mapping,
required inline option fields and the `content_filter` enum value. Final gates pass.

An expanded run also collected `test_interfaces.py`'s two real conversion cases.
Both fail with the same Markdown table golden mismatch (TEDs headings lack the
simple/complex/all qualifiers) on this delta and unchanged stage 1 source.
They are the two deselections in final stage 2 validation. The four skips are
existing heavy/weight-loading extraction tests guarded by `CI=1`; all local
adapter boundary tests run without model weights.

Backend contract checked against the official
[vLLM v0.19.1 structured-output documentation](https://docs.vllm.ai/en/v0.19.1/features/structured_outputs/).
This verifies the transport contract and backend choice; it does not replace
real deployment/model smoke evidence. The implementation intentionally accepts
only its documented bounded subset rather than claiming every decoder's schema
support. No backend capability is inferred from a URL.

Temporary local evidence:
[baseline](/tmp/docling-stage2-baseline.log),
[final stage 2 gates and regressions](/tmp/docling-stage2-regressions.log),
[unchanged-source conversion failures](/tmp/docling-stage2-interface-baseline.log),
[make validate](/tmp/docling-stage2-validate.log).

### Current checkpoint: stage 3 — done

- Worktree: `/Users/cau/Documents/Development/docling-second`, branch
  `cau/extraction-api-service-models`, implementation base HEAD
  `b0e888f846d82c24b5ba64e587f1dd64bd568c5f`. Initial `git status --short`
  was empty. Final changes are only the paths below. The subsequent user request authorized
  a signed-off commit of these paths only; no push. No unrelated tracked/untracked
  work was present or rewritten.
- Authority: read `AGENTS.md`, this stage's instructions, Part I's target,
  pagination, scope, validation and ownership constraints, and revised Part II
  A, D and E of the canonical handoff. The standalone proposal was not used.
  Traced SDK dispatch/cache, base/VLM execution, legacy DTOs, both ordered-content
  adapters/image wrappers, PDF sequential/random page access, declarative serializers,
  DCLX borrowing and InputDocument rejection/ownership paths before their edits.
- SDK: keyword-only `target=` and overloads preserve all positional arguments.
  Exactly one target/template entry is required. Each public SDK call immediately
  owns/revalidates its target or serializes its legacy template, including lazy
  `extract_all()` calls; one legacy warning is emitted per call. Class templates
  preserve main's example/default/sample semantics without inferring a schema.
  The existing style-dependent preparation helpers remain behind that boundary.
- Results: internal execution accumulates only `DocumentExtractionResult.items`.
  SDK legacy calls and public pipeline `execute(template=...)` project page items
  once at their respective outer boundaries. Legacy `pages=` DTO construction and
  serialization are unchanged. The main optional pipeline default prompt still
  works. Legacy unpaginated input fails with directions to use `target=`.
- Chunks: one pipeline-local generator determines native/converted pagination
  before resolving channels. Every selected absolute page receives an independent
  ordered-content request; an unpaginated source receives one text-only document
  request. Explicit image channels/non-default ranges on unpaginated input fail.
  Native page text uses backend text cells; converted page text uses Core's existing
  Markdown serializer. Unattributed, unknown-page and multi-page element provenance
  fail explicitly instead of dropping or repeating text across independent pages.
  Image-only pages remain represented. Converted documents also honor the existing
  `max_num_pages` input limit.
- Ownership: lazy prediction iterators are consumed before advancing the producer.
  Page backends and owned rendered/resized images release in `finally`; the consumer
  closes the chunk generator on early exit. Borrowed DCLX images remain open.
  Invalid/policy-rejected inputs and pipeline initialization/unavailability exits
  release constructed backends, accounting explicitly for rejected inputs whose
  backend was never constructed. Missing/invalid/load-failed selected pages retain
  their known scope; no empty-content chunk is fabricated.
- Validation/status: exactly one unchanged-schema validation per available final
  object; no repair, coercion or default filling. Invalid/nonfinite/non-object JSON,
  schema failures and provider filtering retain raw output/metadata but no successful
  extracted data. Inference/parse/prevented validation reports `not_run`; template-only
  calls report `not_requested`. Standards validation reports `passed`/`failed` with
  JSON paths. A lazy prediction iterator that yields then raises retains its raw
  answer and metadata as a failed item. Independent valid items yield partial success;
  length/stop-sequence completion and document timeout cannot yield full success.
  SDK raised errors now include scoped item errors.
- Timeout: check the document budget before loading/requesting each chunk, carry
  the remaining budget in call-local preparation into API transport (bounded by
  its configured timeout), and represent unprocessed selected pages. Completed
  items remain inspectable. Local generation is not claimed to be hard-preempted.
- Compatibility: restored main-shaped inline NuExtract2/Granite constants in
  `vlm_model_specs.py`; modern callers/defaults use `extraction_options.py`.
  Inline repository/render/generation/processor overrides, serialized options,
  constructors/imports, PIL/numpy/RGB image wrappers and slim imports pass. Serialized
  string prompt styles are normalized to the enum in the shared inline adapter,
  correcting the enum-identity bug exposed by the gate.
- Environment: Python 3.13.5 from `docling_release/.venv`, importing this worktree.
  Published Core 2.96.0 resolves from that environment's `site-packages`; torch
  2.14.0, Transformers 5.16.1, Pydantic 2.13.5, jsonschema 4.26.0, pytest 9.0.3
  and polyfactory 3.3.0. Existing local Python 3.14 tools run without environment
  synchronization. No Core checkout/override, Jobkit/Serve edit, dependency/lock
  change, model enablement or weight download occurred.
- Gates: all channels and absolute ranges, paginated native/converted text,
  unpaginated text/rejections, unattributed text, image-only/missing/invalid pages,
  source load, tokenization, inference, lazy inference, parse, schema-preflight,
  validation-runtime and supplied-schema failures, context overflow, filtering,
  length stop, API budget and timeout cleanup, one-live-page bounds, borrowed/resized
  DCLX ownership, early exits and cached-call isolation pass. Both existing local
  NuExtract2/Granite paths ran through the SDK with synthetic processors/generation;
  both existing API paths ran with captured transport arguments. This is plumbing
  evidence, not a new real-model smoke/support claim.
- Blockers: none for stage 3. The four existing weight-loading skips remain.
  The two known conversion golden mismatches remain deselected, with unchanged-source
  evidence in stage 2 above; no golden was changed. Initial gate failures exposed
  fixture assumptions (DCLX page renumbering, source provenance restoration, compact
  versus pretty JSON, synthetic EOS config and error-message case). Fixtures now
  use actual 1–3 pagination before selecting 2–3, and synthetic converted image-only/
  unsupported-attribution documents retain explicit page metadata. No Core fix was
  necessary. Required `make validate` passes without unrelated rewrites.
- Next: stage 4 only under separate authorization: NuExtract3 model/deployment and
  native/conversion smoke evidence. Stages 4–8 were not implemented in this task.

Changed paths for stage 3:

- `docling/datamodel/extraction.py`
- `docling/datamodel/extraction_options.py`
- `docling/datamodel/pipeline_options.py`
- `docling/datamodel/vlm_model_specs.py`
- `docling/document_extractor.py`
- `docling/models/extraction/api_extraction_model.py`
- `docling/models/extraction/prompt_utils.py`
- `docling/pipeline/base_extraction_pipeline.py`
- `docling/pipeline/extraction_vlm_pipeline.py`
- `tests/test_extraction_api.py`
- `tests/test_extraction_dclx.py`
- `tests/test_extraction_text_channel.py`
- `tests/test_extraction_transformers_model.py`
- `tests/test_extraction_vlm_streaming.py`
- `tests/test_extraction_stage3.py` (new source-to-item gates)
- `docs/plans/extraction-additional-vlm-models-execution.md`

### Stage 3 commands and evidence

Commands run from `docling-second`; no dependency synchronization:

```sh
# Start state: correct path/branch, stage 2 HEAD, clean worktree; exit 0.
pwd
git status --short
git branch --show-current
git rev-parse HEAD

# Environment: published Core/site-packages and versions above; exit 0.
/Users/cau/Documents/Development/docling_release/.venv/bin/python -c 'import sys,importlib.metadata as m,docling_core; print(sys.version); print(docling_core.__path__); print({p:m.version(p) for p in ["torch","transformers","pydantic","jsonschema","pytest","polyfactory"]})'

# Baseline outside sandbox: 225 passed, 4 skipped, 2 deselected, 27 warnings;
# exit 0, 14.99s. Initial sandbox attempt aborted on existing MLX imports, exit 134.
CI=1 HF_HUB_OFFLINE=1 PYTHONPATH=/Users/cau/Documents/Development/docling-second \
  /Users/cau/Documents/Development/docling_release/.venv/bin/python -m pytest -q \
  tests/test_extraction_templates.py tests/test_extraction.py \
  tests/test_extraction_api.py tests/test_extraction_text_channel.py \
  tests/test_extraction_dclx.py tests/test_extraction_vlm_streaming.py \
  tests/test_extraction_transformers_model.py tests/test_service_datamodels.py \
  tests/test_api_image_request.py tests/test_build_generation_config.py \
  tests/test_interfaces.py -k 'not test_convert_path and not test_convert_stream' \
  > /tmp/docling-stage3-baseline.log 2>&1

# Final acceptance gates + regressions outside sandbox: 331 passed, 4 skipped,
# 2 deselected, 31 warnings; exit 0, 16.23s.
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
  > /tmp/docling-stage3-regressions.log 2>&1

# Required repository validation outside sandbox: all applicable hooks pass,
# exit 0; rerun after final source/test/ledger edits. No unrelated rewrites.
UV_NO_SYNC=1 make validate > /tmp/docling-stage3-validate.log 2>&1

# Final branch/HEAD/scope and whitespace audits; exit 0.
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
git diff --check
```

Local `.venv/bin/ruff check --fix` and `.venv/bin/ruff format` were restricted to
changed Python paths; final required hook validation checks those same paths.
Non-fatal installed-hook cache metadata warnings remain. The extra four warnings
in the final regression run are the intentional legacy pipeline warnings in existing
DCLX tests; the SDK warning-count gates explicitly capture/assert one per call.

Temporary evidence:
[baseline](/tmp/docling-stage3-baseline.log),
[final acceptance and regression results](/tmp/docling-stage3-regressions.log),
[required repository validation](/tmp/docling-stage3-validate.log).

Resume prompt (only this file path needs to be carried into a new conversation):

> Continue the next incomplete stage in
> `/Users/cau/Documents/Development/docling-second/docs/plans/extraction-additional-vlm-models-execution.md`.
> Implement that stage, verify its gates and update the checkpoint. Preserve
> unrelated work; stop if docling-core needs changes. Do not commit or publish.
