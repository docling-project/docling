# Plan: generalize `DocumentExtractor` to multiple engines (esp. OpenAI-conformant API)

> Status: **shipped**, and since updated by R1 of
> [`extraction-text-payload.md`](extraction-text-payload.md). This doc describes
> the remote-engine work as it now stands *after* R1 moved extraction onto the
> modern `VlmModelSpec` / preset style. Where the original design differed (an
> `isinstance`-on-options dispatch, an `Inline/ApiExtractionVlmOptions` union as
> the public type), the current shape is described here and the difference noted.

## Goal

Give the extraction pipeline the same *engine selection* the VLM *convert*
pipeline has. Previously extraction ran **only** local HuggingFace transformers
(NuExtract / Granite-Vision); it now also runs against an **OpenAI-conformant
remote API**. Local **mlx / vllm are not a cheap follow-on** — see
[Local mlx / vllm](#local-mlx--vllm-not-free) — because the shared convert engine
layer cannot carry extraction's payloads.

## The model contract

`ExtractionVlmPipeline` drives its model through two calls on `BaseVlmModel`
(`docling/models/base_model.py`):

- `process_images(image_batch, prompt) -> Iterable[VlmPrediction]` — the
  image-only path (today's PDF/IMAGE behavior), shared with convert-side engines.
- `process(requests, template) -> Iterable[VlmPrediction]` — the content-array
  path (text-only or image+text; the `SupportsContentExtraction` protocol added
  by the text-payload work). NuExtract carries its schema out-of-band via
  `template`, which is why this path exists and why the shared convert engines
  cannot serve it (their input is a single image + prompt string).

The prompt flows straight from the pipeline: it builds the prompt text from the
template (on the spec, below) and passes it into the model; the model uses that
passed-in prompt, not `vlm_options.prompt`.

## Options surface (current, post-R1)

Extraction uses the modern spec/preset style, symmetric with convert's
`VlmConvertOptions`:

- **`ExtractionVlmOptions(StagePresetMixin, VlmEngineOptionsMixin, BaseModel)`**
  (`datamodel/extraction_options.py`) is the user-facing type of
  `VlmExtractionPipelineOptions.vlm_options`. It pairs a `model_spec` with an
  `engine_options` (local transformers or a remote API endpoint). Build one with
  `ExtractionVlmOptions.from_preset("nuextract_2b")` /
  `from_preset("granite_vision_4_1")`, or use a named spec.
- **`ExtractionVlmModelSpec(VlmModelSpec)`** carries the per-*model* traits:
  `prompt_style`, channel capability (`accepts_image` / `accepts_text`), the
  transformers load settings, and the prompt logic (`serialize_template` +
  `build_extraction_prompt`). The style travels with the spec, so the pipeline
  never interprets it and an illegal model/style/channel pairing cannot be
  constructed.
- The engine (transformers vs. API endpoint, incl. `url` / `params` / `headers`
  / `timeout`) lives on `engine_options`, not the spec.

> Original design (superseded): the public type was an
> `InlineExtractionVlmOptions | ApiExtractionVlmOptions` union whose mixin
> carried `extraction_prompt_style`, and the pipeline dispatched on that options
> subclass. After R1 those two classes are **internal model-input DTOs** only,
> derived from the spec via `ExtractionVlmOptions.to_inline_input()` /
> `to_api_input()`; the execution models are unchanged and still consume them.

### Back-compat

The released `main` surface — a plain `InlineVlmOptions` as `vlm_options` plus a
pipeline-level `extraction_prompt_style` field — still works: a
`model_validator` on `VlmExtractionPipelineOptions` wraps it into
`ExtractionVlmOptions` and emits a `DeprecationWarning`. Both are slated for
removal in a future release.

## Prompt shaping lives on the spec

Both the template **serialization** and the prompt **embedding** are decided on
`ExtractionVlmModelSpec`, keyed on `prompt_style`, so transformers and API
engines run one shared path:

- **NuExtract** (`NU_EXTRACT_2B_TRANSFORMERS` local, `NU_EXTRACT_API` remote):
  the template is fed through the model's *own* chat template via a special
  `template=` kwarg (out-of-band, not in the message content), so
  `build_extraction_prompt` returns it unwrapped. A Pydantic class serializes to
  a **sample instance** (via polyfactory). It accepts a **text** payload, so it
  drives the text-only formats.
- **Granite schema-instruction** (`GRANITE_VISION_4_1_TRANSFORMERS`,
  `GRANITE_VISION_4_1_API`): the serialized template is wrapped in a plain-text
  instruction ("Extract structured data… Return a JSON object matching this
  schema… Return ONLY valid JSON") and fed through a standard chat conversation.
  A Pydantic class serializes to a real **JSON Schema** with field descriptions
  (`model_json_schema()`), the key-value extraction format from the Granite
  Vision model card. Image-only (`accepts_text=False`).

Net: prompt style — not engine — decides how the template becomes text; the
engine only decides how that text + image are executed.

`ExtractionPromptStyle.GRANITE_VISION` is really a generic JSON-schema
extraction style and would work for any instruction-tuned VLM over the API, not
only Granite. A neutral alias such as `SCHEMA` is possible later but not part of
this change.

## Dispatch — engine type, not options subclass

`ExtractionVlmPipeline.__init__` (`pipeline/extraction_vlm_pipeline.py`) selects
the execution model from `engine_options.engine_type`:

```python
vlm_options = pipeline_options.vlm_options  # ExtractionVlmOptions
engine_type = vlm_options.engine_options.engine_type
if VlmEngineType.is_api_variant(engine_type):
    api_input = vlm_options.to_api_input()
    if vlm_options.extraction_prompt_style == ExtractionPromptStyle.NUEXTRACT:
        # NuExtract carries its template out-of-band -> content-array request path
        self.vlm_model = ApiExtractionVlmModel(..., vlm_options=api_input)
    else:
        # Granite -> plain image-request path (shared convert-side ApiVlmModel)
        self.vlm_model = ApiVlmModel(..., vlm_options=api_input)
else:  # inline transformers
    self.vlm_model = TransformersExtractionModel(
        ..., vlm_options=vlm_options.to_inline_input()
    )
```

The NuExtract-vs-Granite branch is a *request-shape* detail internal to the API
path (out-of-band template vs. in-message instruction), not options-subclass
routing. `_extract_data` / `_determine_status` are unchanged — every engine
returns real `stop_reason`s, which the existing LENGTH/STOP_SEQUENCE →
PARTIAL_SUCCESS logic handles.

> `TransformersExtractionModel` is the *unified* local model (added in #3398 with
> Granite Vision 4.1); it dispatches internally on `prompt_style` and serves both
> NuExtract and Granite. The older NuExtract-only
> `models/extraction/nuextract_transformers_model.py`, orphaned by that
> unification, has since been removed.

## Engine × prompt-style matrix

| Prompt style | transformers (local) | API / vllm |
|--------------|----------------------|------------|
| NuExtract    | ✅ (special `template=`) | ✅ (`ApiExtractionVlmModel`, out-of-band template) |
| Granite schema-instruction | ✅ | ✅ (`ApiVlmModel`, image-request) |

## Local mlx / vllm (not free)

Not a cheap follow-on. The shared convert-side engines
(`HuggingFaceMlxModel` / `VllmVlmModel`, reached via `create_vlm_engine`) accept
a single required image + a prompt string (`VlmEngineInput`); they cannot carry
extraction's text-only payloads, image+text content arrays, or NuExtract's
out-of-band template. So each local engine would need its **own** extraction
model implementing `process(requests, template)`, or a later generalization of
`VlmEngineInput` to a content array + template. Deferred until asked; R1's option
layer already has clean slots for the extra engine types.

## Out of scope
- Chart extraction (`datamodel/chart_extraction_options.py`,
  `models/stages/chart_extraction`) — separate subsystem, untouched.
- There is no extraction CLI today; this stays a Python-API-only change.

## Testing
- `tests/test_extraction_api.py`: dispatch to the API models, both prompt-style
  regimes (Granite schema-instruction wraps a JSON Schema; NuExtract passes a
  sample instance through unwrapped), and the `enable_remote_services=False`
  guard raising `OperationNotAllowed`.
- `tests/test_granite_vision_extraction.py`: the Granite preset carries the
  schema-instruction style and repo on `model_spec`; the pipeline hands the model
  the style-carrying (lowered) spec.
- `tests/test_extraction_text_channel.py`: capability-based channel resolution
  and the remote NuExtract out-of-band template payload.

## Docs / packaging
- `docling/.agents/skills/docling/references/extraction.md` has a "Choosing the
  engine" section with a remote example (endpoint on `engine_options`) and
  `enable_remote_services=True`.
- Slim extras: API-only extraction needs no torch. Check whether a
  `models-vlm-api`-style extra should cover extraction so users can run remote
  extraction without `models-vlm-inline`. Note it in `slim-packaging.md`.
