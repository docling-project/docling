# Continue stage 6: Qwen3.5 4B/9B documented extraction contract

Implement **stage 6 only** in `/Users/cau/Documents/Development/docling-second`
on `cau/extraction-api-service-models`. Read repo `AGENTS.md`,
[the execution ledger](extraction-additional-vlm-models-execution.md) and Qwen's
Part I prerequisites / Part II F in
[the authoritative handoff](/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md).
Recheck branch, HEAD, worktree and environment. Stage 4 is signed off at
`ef5a8a9305f63e4c4eb3831505a4e71a2056d367`; stage 5 may now have the parent's
authorized signoff commit. Do not reset/overwrite its Lift implementation.
Preserve unrelated files and historical raw smoke evidence. Ponytail full: trace
all affected callers and reuse existing preparation, transports and ownership.
The parent coordinates one sequential worker per stage and reviews/signs its
commit; this worker creates no agents or commits.

No other model stages, downstream code, Core edits, dependency sync, weight
loading/download, live inference, runtime changes, production defaults, commit,
push or publication is authorized here. Stop if Core edits are needed.

The user revised delivery on 2026-09-18: source implementation follows documented
behavior and offline contract tests, independent of models/endpoints responding.
This supersedes the old live-only Part II F completion gates while preserving
design/schema/ownership/interfaces. Live smoke/capacity/quality stays separately
unverified per model/engine and does not block source stages 6–11. Do not add
verification registries/flags or present supported-engine lists as certification.

**Caller-provided extraction templates are mandatory for every model.** Implement
explicit `ExtractionTemplate(format="example_json", value=...)` guidance alongside
schema-only guidance for both Qwen sizes. Prove exact caller fields/values reach
local processor rendering and actual API payloads, and prove different templates/
instructions on one cached extractor remain isolated. Typed `nuextract` is a
different tagged dialect and must be rejected honestly on generic models. An
extraction template describes caller output structure; a checkpoint chat template
serializes messages. Do not replace Jinja, infer schemas from examples or silently
drop explicit templates. Carry this mandatory template gate into stage 7 Gemma's
next continuation prompt.

Recheck pinned primary Qwen3.5 4B/9B model cards, configs, processor/chat templates,
Transformers and served API contracts, and weight license/access terms before
editing. Add only two opt-in presets using the same generic multimodal profile,
with actual checkpoint differences in preset data. Preserve the existing loader
without concrete contrary evidence. Supply `enable_thinking=False` at documented
local/vLLM chat-template boundaries; do not add a speculative final-answer parser.
Only advertise implemented transports. Other providers' thinking options differ
(e.g. Alibaba top-level); do not reuse vLLM kwargs without documentation/coverage.
No provider SDK workaround, new dependency or runtime change is expected.

Reuse generic target preparation, ordered `ContentItem` conversion, original-schema
validation and explicit constrained-vLLM subset preflight. Constrained output
requires a schema; unsupported keywords/engines fail before inference and provider
rejection never falls back to prompt-only. Qwen's example-only prompt mode does not
inherit Lift's `requires_output_schema` requirement. Context/output settings from
official configs/examples are static unmeasured bounds, not memory/capacity claims;
4B and 9B live capacity must remain separately open.

Exercise text/image/mixed input through offline local rendering and actual HTTP
payload capture. Reliable pagination means independent requests for selected
absolute pages 2–3, with separate page-scoped results. Unpaginated text remains
one document request. Prove cached-call template/instruction changes, schema-only
and explicit examples, negative original-schema validation, thinking controls,
unsupported dialect/mode/constraint rejection and documented termination behavior.
No joint-page inference, public grouping/chunk input, pipeline model branch or
copied image payload in durable results. Preserve image/finally ownership, SDK
legacy projection and NuExtract2/NuExtract3/Granite/Lift behavior. Extend existing
behavioral tests instead of copying constant/preset assertions.

Use borrowed `/Users/cau/Documents/Development/docling_release/.venv/bin/python`
(Python 3.13.5, Transformers 5.16.1) with
`PYTHONPATH=/Users/cau/Documents/Development/docling-second`. Published Core
resolves from site-packages; never use its dirty checkout/editable override.
Do not sync environments. Existing native MLX imports require permitted sandbox
escalation. Run the ledger's applicable extraction regression command (HF offline,
four existing weight skips, two known conversion tests deselected); preserve
goldens. Run `UV_NO_SYNC=1 make validate`, review hook edits and rerun until clean,
then audit exact changed paths and `git diff --check`.

Update the ledger with stage 5's exact signed commit, stage 6 source decisions,
primary references, exact commands/results and separate per-size/per-engine live
verification. Mark source implementation done only after offline gates pass,
then create a self-contained stage 7 prompt with the revised policy and mandatory
explicit-template gate. NuExtract3 local/vLLM live verification is still not run;
installed LM Studio is incompatible, raw evidence unchanged. Lift local/vLLM live
verification, measured capacity and schema-versus-example extraction quality are
also not run. Do not claim accuracy, measured capacity, live loading or quality
comparison without separately authorized runs and actual evidence.
