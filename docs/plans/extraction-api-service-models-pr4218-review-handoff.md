# Handoff: resolve PR #4218 self-review comments

Status: drafted 2026-09-18, against `cau/extraction-api-service-models` HEAD
`1117924ad2374a0856b33c06d92813cde6d897f5`. One item (6) already applied at that
HEAD; the rest are proposals for the implementer to carry out and re-verify
against the live diff before committing, since the branch continues to move.

## Source

Six inline self-review comments on
[PR #4218](https://github.com/docling-project/docling/pull/4218#pullrequestreview-5249406266),
all on `docling/datamodel/extraction.py` and
`docling/models/extraction/nuextract_transformers_model.py`. Each item below
gives the comment, the resolution, and the evidence backing it — including
primary-source corrections made during review (item 2 revises a first-pass
answer that turned out to be wrong).

## 1. Document the three template/schema representations

**Comment:** "Explain what distinguishes 'nuextract' and 'example_json'
formats" (`ExtractionTemplate.format`, `docling/datamodel/extraction.py:53`).

**Resolution:** Add a docstring/comment on `ExtractionTemplate.format`
distinguishing all three representations Docling handles for an extraction
target, not just the two `format` values:

| Representation | Example | Meaning |
|---|---|---|
| JSON Schema (`output_schema`) | `{"type":"object","properties":{"total":{"type":"number"}},"required":["total"]}` | A validation constraint, interpreted by a JSON Schema validator |
| `template.format == "nuextract"` | `{"total": "number"}` | NuExtract's native semantic type vocabulary (`verbatim-string` vs `string`, `date-time`, `currency`, enum-as-array, multi-enum-as-nested-array); shape-identical to the output, no schema meaning |
| `template.format == "example_json"` | `{"total": 123.45}` | An illustrative example value, explicitly not a constraint |

**Evidence:** NuExtract3's model card
([huggingface.co/numind/NuExtract3](https://huggingface.co/numind/NuExtract3),
"Using NuExtract" section): "NuExtract uses an input JSON template whose
structure is identical to the output JSON. Its leaf values specify the types
of the output JSON leaves," with `verbatim-string` documented as "extract
text exactly as it appears," distinct from generic `string`. This is a real,
benchmarked distinction (NuExtract's own eval scores `verbatim-string`/
`string` leaves with Levenshtein distance vs. exact-match for everything
else) — not something either JSON Schema or a plain example can express.

## 2. Correct where `output_schema` guidance actually goes, and stop dumping the full schema into NuExtract's `instructions`

**Comment:** "Justify why we need the complexity of both `output_schema` and
`template`... Which models actually expect or support both being defined"
(`docling/datamodel/extraction.py:71-72`).

**Resolution — keep both fields**, they carry non-overlapping meaning:
`output_schema` is always the validation contract; `template` is model
guidance carrying semantics `output_schema` can't (verbatim-string) or a
plain example. Document the three cases in `ExtractionTarget`'s docstring:

- Schema only → the preparation helper derives its own guidance (native
  template via `_schema_to_nuextract`, or a prose "Output contract" block).
- Template only → parsed as JSON, no validation (`validation_status` stays
  `not_requested`).
- Both → template is model guidance, schema is the validation contract. This
  is the required combination whenever a native semantic type
  (`verbatim-string`) matters, since the schema alone can't express it.

**Resolution — fix a real defect found while answering this.** For the
NuExtract branch (`model_spec.preparation == "nuextract"`,
`docling/models/extraction/prompt_utils.py:92-107`), the full guidance text —
including a raw Draft-2020-12 JSON Schema dump with `$schema`/`$defs`/`title`
— is stuffed into the `instructions` chat-template kwarg
(`prompt_utils.py:86-90, 106`). `instructions` is a real, official NuExtract
input (confirmed in both the model card and the pinned
`chat_template.jinja`), but it's documented and demonstrated as a short,
targeted sentence (README example: `"Specify the time for the date entry
only if it is present, otherwise only output the date component."`), not a
receptacle for an entire schema. Replace the schema dump in the NuExtract
branch with a short, targeted note that only says what the native template
can't: which fields are required and which allow null. The `_MISSING_INSTRUCTIONS`
constant (`prompt_utils.py:55-59`) already covers the general rule; drop the
`json.dumps(schema, indent=2)` block for this branch specifically. Leave the
generic-chat branch (`prompt = "\n\n".join(guidance)`, `prompt_utils.py:118`)
as is — there, the full schema in a text prompt is normal and expected.

**Evidence:** pinned `chat_template.jinja` at
`numind/NuExtract3@c99dc8f5641b866aa0192b6ea78f84bf9f3535f1`: for
`mode == "structured"`, `template` and `instructions` render as separate
`【template_start】`/`【instructions_start】` blocks — both are chat-template
kwargs, never folded into a user-message prompt string
(`prompt_utils.py:107` sets `prompt = ""` for this branch, confirmed
correct). `_schema_to_nuextract`'s own docstring
(`docling/models/extraction/template_utils.py`) already states
"Requiredness, nullability and scalar bounds remain authoritative in the
validator" — i.e. the native template never carries them, which is exactly
why some text channel is genuinely needed; it just shouldn't be the whole
schema.

## 3. Split VLM inference telemetry out of `ExtractionItem`

**Comment:** "These are very VLM specific fields... It would probably need a
better home or bundled into a single submodel?"
(`docling/datamodel/extraction.py:124-128`).

**Resolution:** introduce one submodel and replace the five flat fields with
a single optional reference:

```python
class VlmInferenceMetadata(BaseModel):
    """Backend inference telemetry; absent for non-VLM extraction backends."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    generated_tokens: list[VlmPredictionToken] = Field(default_factory=list)
    generation_time: float = -1
    num_tokens: int | None = None
    usage: dict[str, JsonValue] | None = None
    stop_reason: VlmStopReason = VlmStopReason.UNSPECIFIED

    @field_validator("usage", mode="before")
    @classmethod
    def _usage_values(cls, value: Any) -> Any:
        if isinstance(value, OpenAiResponseUsage):
            return value.model_dump(mode="json")
        return value


class ExtractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope: ExtractionScope
    extracted_data: dict[str, JsonValue] | None = None
    raw_text: str | None = None
    errors: list[str] = Field(default_factory=list)  # see item 5: retype to list[ErrorItem]
    validation_status: ExtractionValidationStatus = "not_requested"
    inference: VlmInferenceMetadata | None = None
```

**Follow-through required:** every write site in
`docling/pipeline/extraction_vlm_pipeline.py` that currently sets
`item.generated_tokens = ...`, `item.generation_time = ...`, etc.
individually needs to construct one `VlmInferenceMetadata(...)` instead —
these fields are already populated together at the same point in the
pipeline, so this is a mechanical rename, not a control-flow change. Update
the service wire mirror in `docling/datamodel/service/responses.py` and any
client/service consumer that reads these fields flat
(`docling/service_client/`, downstream `docling-jobkit`/`docling-serve`
result handling) to go through `.inference.*` instead. Grep for
`generated_tokens|generation_time|num_tokens\b|\.usage\b|stop_reason` across
`docling/`, `tests/`, and the sibling service repos before editing to find
every touch point — do not assume the pipeline is the only writer.

**Scope note:** do not build a base/subtype split for VLM-vs-non-VLM
`ExtractionItem` — add that only when a second, non-VLM extraction backend
actually exists.

## 4. `ExtractionValidationStatus` meanings — confirmed, just needs a docstring

**Comment:** "Explain what these mean, except 'passed' and 'failed', which is
clear" (`docling/datamodel/extraction.py:111`).

**Resolution:** add a docstring/comment on the `Literal`:

- `not_requested` — no `output_schema` was supplied; validation was never in
  scope for this item.
- `not_run` — an `output_schema` was supplied, but validation could not
  execute because inference or JSON parsing failed first (no parsed object
  ever existed to check against the schema).
- `passed` / `failed` — schema was supplied, parsing succeeded, and the
  parsed object did or didn't validate.

No code change needed beyond documentation; this already matches
`docling/pipeline/extraction_vlm_pipeline.py:460-471`.

## 5. Unify item-level and document-level errors on `ErrorItem`

**Comment:** "we have 'errors' here (as `ErrorItem`) and we have errors in
`ExtractionItem` (`str`) — which error is going where, and why is the type
different? Can we go with one of them only?"
(`docling/datamodel/extraction.py:145-146`).

**Resolution — same type, two scopes, not one merged list.**

*Type-wise:* retype `ExtractionItem.errors` from `list[str]` to
`list[ErrorItem]`. `ErrorItem` (`docling/datamodel/base_models.py:328`) is a
strict superset of a bare string — it has `error_message: str` plus
`component_type`, `category`, and `page_no` for filtering — and it's already
Docling's canonical error DTO everywhere else. `FailureCategory`'s own
docstring (`base_models.py:304-315`) already documents `BACKEND_FAILURE` and
`INFERENCE_FAILURE` as "document/page-scope" categories, i.e. the vocabulary
for item-level extraction failures already exists and simply isn't being
reused here.

*Scope-wise:* keep two separate lists, don't collapse them into one.
`DocumentExtractionResult.errors` covers failures where a chunk never got
built at all (source/page load failed before an `ExtractionItem` could
exist — there is no `scope` to attribute the error to). `ExtractionItem.errors`
covers failures after a chunk was successfully scoped, during
inference/decode/validation of that specific item — and because `ErrorItem`
already has `page_no`, that scope is free to carry once both lists share a
type: `page_no=(item.scope.page_no if isinstance(item.scope, PageScope) else
None)`.

**Follow-through required:** update every append site in
`extraction_vlm_pipeline.py` (`item.errors.append("Model returned invalid
JSON: ...")`, the schema-validation error loop, etc.,
`extraction_vlm_pipeline.py:450-476`) to construct `ErrorItem(component_type=
DoclingComponentType.MODEL, category=FailureCategory.INFERENCE_FAILURE,
module_name=..., error_message=..., page_no=...)` instead of appending a bare
string. Update the service wire response model
(`docling/datamodel/service/responses.py`) and any client/consumer that
currently expects `item.errors: list[str]`.

## 6. `NuExtractTransformersModel` deprecation warning — applied

**Comment:** "Why do we have a derived class for this which does absolutely
nothing but forward the constructor args?"
(`docling/models/extraction/nuextract_transformers_model.py:15`), and the
follow-up: "there is something missing: A dedicated deprecation warning must
be logged somewhere."

**Status: done**, at HEAD `1117924a`. The class stays — it existed on `main`
as a full standalone implementation before this PR merged NuExtract and
Granite behind one `TransformersExtractionModel` (which already defaults to
`prompt_style=ExtractionPromptStyle.NUEXTRACT`), so it's kept only as a
compatibility shim for `main`'s import path/constructor signature, matching
the pattern already used for the `template=` SDK path
(`normalize_extraction_call`, `docling/models/extraction/prompt_utils.py:123-138`,
which already emits `DeprecationWarning`). Applied the same pattern here:

```python
class NuExtractTransformersModel(TransformersExtractionModel):
    """Deprecated: construct TransformersExtractionModel directly.

    It already defaults to the NuExtract prompt style; this subclass exists
    only to keep main's import path and constructor signature importable.
    """

    def __init__(self, enabled, artifacts_path, accelerator_options, vlm_options):
        warnings.warn(
            "NuExtractTransformersModel is deprecated; construct "
            "TransformersExtractionModel directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(enabled, artifacts_path, accelerator_options, vlm_options)
```

**Open follow-up, not yet done:** the class is currently referenced by
nothing except `tests/test_extraction_transformers_model.py:313,354`. Before
treating the compatibility shim as load-bearing, confirm there's an actual
external caller (docling-jobkit, docling-serve, or a known third-party
integration) that imports `NuExtractTransformersModel` directly — if not,
the deprecation warning is the right interim step, but the class could be
removed outright in a later cleanup once that's confirmed.

## Sequencing

Items 4 and 6 are complete or documentation-only — no dependency on the
others. Items 1 and 2 touch the same file/section
(`docling/datamodel/extraction.py`, `prompt_utils.py`) and should land
together. Items 3 and 5 both change `ExtractionItem`'s shape and have the
same category of follow-through (pipeline write sites, service wire
mirrors, downstream consumers in `docling-jobkit`/`docling-serve`) — do them
in one pass so the field-renaming grep and test updates aren't duplicated.
Re-run `tests/test_extraction.py`, `tests/test_extraction_templates.py`, and
`tests/test_extraction_vlm_streaming.py` after 3 and 5, and `make validate`
before considering the PR ready for a non-self review.
