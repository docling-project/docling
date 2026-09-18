# Continue stage 5: Lift documented extraction contract

Implement **stage 5 only** in `/Users/cau/Documents/Development/docling-second`
on `cau/extraction-api-service-models`. Read repo `AGENTS.md`,
[the execution ledger](extraction-additional-vlm-models-execution.md) and Lift's
Part I prerequisites / Part II F in
[the authoritative handoff](/Users/cau/Documents/Development/docling_release/docs/plans/extraction-additional-vlm-models-handoff.md).
Recheck branch, HEAD, worktree and environment; stage 4's implementation delta may
now have the parent's authorized signoff commit. Do not reset or overwrite it.
Preserve unrelated files and historical raw smoke evidence. Ponytail full: trace
all affected callers and reuse existing preparation, transports and ownership.
No other model stages, downstream code, Core edits, dependency sync, weight
loading/download, live inference, runtime changes, production defaults, commit,
push or publication is authorized by this prompt. Stop if Core edits are needed.

The user revised delivery on 2026-09-18: source implementation follows documented
behavior and offline contract tests, independent of models/endpoints responding.
This supersedes the old live-only Part II F completion gates, preserving design,
schema validation, ownership and interfaces. Live smoke/capacity/quality remains
separately unverified per model/engine; it does not block stages 5–11. Do not
introduce verification registries/flags, or mark engine lists as certified.

**Caller-provided extraction templates are mandatory for every model.** For Lift,
implement explicit `ExtractionTemplate(format="example_json", value=...)` guidance
plus its documented schema-only path. Prove exact caller fields/values reach its
renderer/API messages, and prove different templates/instructions on one cached
extractor remain isolated. Native NuExtract typed `nuextract` is a different tagged
dialect; reject it honestly on generic models. Do not offer schema-only support
while silently dropping explicit templates. Extraction templates describe caller
output structure; checkpoint chat templates serialize messages. Do not replace
Jinja or add SDK workarounds. Stage 6 Qwen and stage 7 Gemma must carry this same
explicit-example-template gate into their own continuation prompts.

Recheck pinned primary model/processor/Transformers/vLLM contracts and license
requirements before editing. Add only Lift opt-in preset data and helpers required
by those contracts. Preserve existing loader absent concrete failure evidence;
implement documented local and served API paths without executing them. Reuse
shared `generic_chat` preparation, ordered `ContentItem` content, explicit
`schema_constrained` vLLM transport, subset preflight and unchanged-schema
validation. Configure documented EOS/stop behavior. Unsupported constraints fail
before inference; provider rejection never silently falls back to prompt-only.
Use static context/output bounds only if justified by official contracts and label
them unmeasured; earlier LM Studio probe settings are unrelated to Lift capacity.

Exercise supported text/image/mixed channels through offline renderer and actual
HTTP payload capture. Reliable pagination means one independent request per
selected absolute page; include non-1 pages 2–3, separate page-scoped outputs and
cached-call target changes. Unpaginated text remains one document request. No
joint-page inference, public grouping/chunk input, model-specific pipeline branch
or copied image payload in durable results. Include original-schema negative
validation, unsupported mode/constraint rejection and documented generation stops.
Retain image ownership/finally cleanup, legacy SDK projection and NuExtract2,
NuExtract3 and Granite compatibility. Reuse existing behavioral tests rather than
copying constant/preset assertions.

Use borrowed `/Users/cau/Documents/Development/docling_release/.venv/bin/python`
(Python 3.13.5) with `PYTHONPATH=/Users/cau/Documents/Development/docling-second`;
published Core resolves from site-packages, not its dirty checkout. No environment
sync/editable overrides. Known MLX native import abort needs permitted escalation
outside the sandbox. Run the applicable extraction regression command in the
current ledger (HF offline, existing four weight skips, two known conversion tests
deselected); preserve goldens. Run `UV_NO_SYNC=1 make validate`, review any hook
edits, rerun until clean, and audit exact changed paths/`git diff --check`.

Update the ledger with source changes, exact commands/results/primary sources and
separate Lift model/engine live verification states. Mark implementation done only
when offline gates pass, then create the next self-contained stage 6 prompt with
the revised policy and mandatory template gate. Do not claim model accuracy,
measured capacity, live loading or schema-versus-example quality comparison unless
separately authorized and actually run. NuExtract3 local/vLLM live verification is
still not run; installed LM Studio is incompatible, raw evidence unchanged.
