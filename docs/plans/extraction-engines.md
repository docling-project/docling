# Plan: generalize `DocumentExtractor` to multiple engines (esp. OpenAI-conformant API)

## Goal

Give the extraction pipeline the same engine flexibility the VLM *convert*
pipeline has. Previously extraction ran **only** local HuggingFace transformers
(NuExtract / Granite-Vision); it now also runs against an **OpenAI-conformant
remote API**, reusing the convert-side `ApiVlmModel`. mlx / vllm are a cheap
follow-on left for later.

## Key finding: the interface is already shared

`ExtractionVlmPipeline` interacts with its model through exactly one call:

```python
# docling/pipeline/extraction_vlm_pipeline.py
predictions = list(self.vlm_model.process_images([image], prompt))
```

`process_images(image_batch, prompt) -> Iterable[VlmPrediction]` is the
`BaseVlmModel` abstract interface (`docling/models/base_model.py`). Every
convert-side engine already implements it:

| Engine | Class | Options type |
|--------|-------|--------------|
| API (OpenAI-conformant) | `ApiVlmModel` (`models/vlm_pipeline_models/api_vlm_model.py`) | `ApiVlmOptions` |
| transformers | `HuggingFaceTransformersVlmModel` | `InlineVlmOptions` |
| mlx | `HuggingFaceMlxModel` | `InlineVlmOptions` |
| vllm | `VllmVlmModel` | `InlineVlmOptions` |

`VlmPipeline._initialize_legacy_vlm_models` (`pipeline/vlm_pipeline.py`) already
dispatches over these based on the options type. The extraction pipeline copies
that dispatch — no new abstraction; `process_images(images, prompt)` is the
whole contract.

The prompt already flows correctly: the pipeline builds the prompt from the
template and passes it straight into `process_images`.
`ApiVlmModel.process_images` uses that passed-in prompt (not
`vlm_options.prompt`), so the built prompt becomes the API text prompt with no
extra work.

## Design: prompt shaping lives on the spec

The convert side attaches prompt shaping to the model spec (`build_prompt` /
`decode_response`). Extraction follows the same rule so the pipeline never
interprets prompt style and an illegal model/style pairing cannot be
constructed.

- Extraction has its own spec types, `InlineExtractionVlmOptions` and
  `ApiExtractionVlmOptions` (`datamodel/extraction_options.py`). Both mix in
  `ExtractionVlmOptionsMixin`, which carries `extraction_prompt_style` plus
  `serialize_template` + `build_extraction_prompt`.
- `VlmExtractionPipelineOptions.vlm_options` is
  `InlineExtractionVlmOptions | ApiExtractionVlmOptions`. There is no standalone
  `extraction_prompt_style` field on the pipeline options — the style travels
  with the spec.
- Each preset welds a model to the only style it can honor, so illegal
  model/style pairings are unconstructable.

Both the template **serialization** and the prompt **embedding** are decided on
the spec, keyed on `ExtractionPromptStyle`, so transformers and API engines run
one shared path:

- **NuExtract** (`NU_EXTRACT_2B_TRANSFORMERS`, local-only): the template is fed
  through the model's *own* chat template via a special `template=` kwarg, so
  `build_extraction_prompt` returns it unwrapped. A Pydantic class serializes to
  a **sample instance** (via polyfactory). Not served over generic endpoints.
- **Granite schema-instruction** (`GRANITE_VISION_4_1_TRANSFORMERS`,
  `GRANITE_VISION_4_1_API`): the serialized template is wrapped in a plain-text
  instruction ("Extract structured data… Return a JSON object matching this
  schema… Return ONLY valid JSON") and fed through a standard chat conversation.
  A Pydantic class serializes to a real **JSON Schema** with field descriptions
  (`model_json_schema()`). This is the key-value extraction format from the
  Granite Vision model card (the format the model was evaluated with on the
  VAREX benchmark), and plain-text + standard chat is exactly what an
  OpenAI-conformant endpoint consumes.

Net: prompt style — not engine — decides how the template becomes text; the
engine only decides how that text + image are executed.

`ExtractionPromptStyle.GRANITE_VISION` is really a generic JSON-schema
extraction style and would work for any instruction-tuned VLM over the API, not
only Granite. (VAREX is the benchmark Granite was evaluated on, not the name of
the prompt format — it is not reused as the style name.) A neutral alias such as
`SCHEMA` is possible later but not part of this change.

## Changes

### 1. Extraction spec types — `datamodel/extraction_options.py`
`ExtractionVlmOptionsMixin` adds `extraction_prompt_style`, `serialize_template`,
and `build_extraction_prompt`. `InlineExtractionVlmOptions` and
`ApiExtractionVlmOptions` combine the mixin with `InlineVlmOptions` /
`ApiVlmOptions`. `_build_extraction_prompt` (the schema-instruction wrapper)
lives here and is re-exported from `models/extraction/prompt_utils.py`.

### 2. Widen the options type — `datamodel/pipeline_options.py`
`VlmExtractionPipelineOptions.vlm_options` is
`InlineExtractionVlmOptions | ApiExtractionVlmOptions`. Default stays
`NU_EXTRACT_2B_TRANSFORMERS`.

### 3. Dispatch on engine — `pipeline/extraction_vlm_pipeline.py`
`__init__` picks the model from the spec type, mirroring
`VlmPipeline._initialize_legacy_vlm_models`:

```python
opts = pipeline_options.vlm_options
if isinstance(opts, ApiExtractionVlmOptions):
    self.vlm_model = ApiVlmModel(
        enabled=True,
        enable_remote_services=pipeline_options.enable_remote_services,
        vlm_options=opts,
    )
else:  # InlineExtractionVlmOptions -> local transformers extraction model
    self.vlm_model = TransformersExtractionModel(...)
```

`_extract_data` / `_determine_status` are unchanged — `ApiVlmModel` returns real
`stop_reason`s, which the existing LENGTH/STOP_SEQUENCE → PARTIAL_SUCCESS logic
already handles. The granite transformers builder consumes the already-wrapped
prompt and does not wrap again.

### 4. Ship an API preset — `datamodel/vlm_model_specs.py`
`GRANITE_VISION_4_1_API`: an `ApiExtractionVlmOptions` targeting a Granite
schema-instruction endpoint (Granite Vision 4.1 on vLLM or Ollama,
`ResponseFormat.PLAINTEXT`, `temperature=0.0`), with
`extraction_prompt_style = GRANITE_VISION`. Users override `url`, `headers`
(bearer token), and `params["model"]`.

## Engine × prompt-style matrix

| Prompt style | transformers (local) | API / vllm |
|--------------|----------------------|------------|
| NuExtract    | ✅ (special `template=`) | ✗ (model-specific format) |
| Granite schema-instruction | ✅ | ✅ — this is what the API engine unlocks |

## Optional follow-on: local mlx / vllm engines
Cheap once the dispatch + shared prompt assembly exist: route
`InlineExtractionVlmOptions` with `inference_framework == MLX/VLLM` to the
existing `HuggingFaceMlxModel` / `VllmVlmModel`. They consume the
schema-instruction-wrapped text like any chat model, so they work for the
Granite schema-instruction style but not NuExtract's special format. Deferred
until asked.

## Out of scope
- Chart extraction (`datamodel/chart_extraction_options.py`,
  `models/stages/chart_extraction`) — separate subsystem, untouched.
- There is no extraction CLI today; this stays a Python-API-only change.

## Testing
- `tests/test_extraction_api.py`: dispatch to `ApiVlmModel`, both prompt-style
  regimes (Granite schema-instruction wraps a JSON Schema; NuExtract passes a
  sample instance through unwrapped), and the `enable_remote_services=False`
  guard raising `OperationNotAllowed`.
- `tests/test_granite_vision_extraction.py`: the Granite preset carries the
  schema-instruction style on the spec.

## Docs / packaging
- `docling/.agents/skills/docling/references/extraction.md` gains a "Choosing
  the engine" section with a remote example and `enable_remote_services=True`.
- Slim extras: API-only extraction needs no torch. Check whether a
  `models-vlm-api`-style extra should cover extraction so users can run remote
  extraction without `models-vlm-inline`. Note it in `slim-packaging.md`.
