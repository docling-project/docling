# Plan: generalize `DocumentExtractor` to multiple engines (esp. OpenAI-conformant API)

## Goal

Today the extraction pipeline runs **only** local HuggingFace transformers
(NuExtract / Granite-Vision). We want the same engine flexibility the VLM
*convert* pipeline already has — at minimum an **OpenAI-conformant remote API**
engine, ideally also mlx / vllm — reusing the convert-side machinery.

## Key finding: the interface is already shared

`ExtractionVlmPipeline` interacts with its model through exactly one call:

```python
# docling/pipeline/extraction_vlm_pipeline.py:76
predictions = list(self.vlm_model.process_images([image], prompt))
```

`process_images(image_batch, prompt) -> Iterable[VlmPrediction]` is the
`BaseVlmModel` abstract interface (`docling/models/base_model.py:49`). Every
convert-side engine already implements it:

| Engine | Class | Options type |
|--------|-------|--------------|
| API (OpenAI-conformant) | `ApiVlmModel` (`models/vlm_pipeline_models/api_vlm_model.py`) | `ApiVlmOptions` |
| transformers | `HuggingFaceTransformersVlmModel` | `InlineVlmOptions` |
| mlx | `HuggingFaceMlxModel` | `InlineVlmOptions` |
| vllm | `VllmVlmModel` | `InlineVlmOptions` |
| **extraction (current)** | `TransformersExtractionModel` | `InlineVlmOptions` + `ExtractionPromptStyle` |

`VlmPipeline._initialize_legacy_vlm_models` (`pipeline/vlm_pipeline.py:126`)
already dispatches over these based on the options type / `inference_framework`.
**We copy that dispatch into the extraction pipeline.** No new abstraction is
needed — the port `process_images(images, prompt)` is the whole contract.

The prompt already flows correctly: `_extract_data` computes
`prompt = self._serialize_template(template)` and passes it straight into
`process_images`. `ApiVlmModel.process_images` uses that passed-in prompt
(not `vlm_options.prompt`), so the serialized template becomes the API text
prompt with no extra work.

## What blocks the API engine today

1. **Type gate.** `VlmExtractionPipelineOptions.vlm_options` is typed
   `InlineVlmOptions` (`pipeline_options.py:1884`) — an `ApiVlmOptions` can't
   even be assigned.
2. **Hardcoded model.** `ExtractionVlmPipeline.__init__` unconditionally builds
   `TransformersExtractionModel` (`extraction_vlm_pipeline.py:49`).
3. **No preset.** No `ApiVlmOptions` extraction preset exists in
   `vlm_model_specs.py` (only the two local `InlineVlmOptions`).

Everything else is already in place: `enable_remote_services` is inherited from
`PipelineOptions` (`pipeline_options.py:1370`), and `ApiVlmModel` enforces it.

## Changes (recommended scope: add the API engine)

### 1. Widen the options type — `datamodel/pipeline_options.py`
Change `VlmExtractionPipelineOptions.vlm_options` from `InlineVlmOptions` to
`Union[InlineVlmOptions, ApiVlmOptions]` (discriminated on the existing `kind`
literal, same pattern as the convert side). Default stays
`NU_EXTRACT_2B_TRANSFORMERS`. `extraction_prompt_style` stays meaningful on the
API path too (it selects the VAREX serialization+wrapper, see §4); only the
NuExtract style is transformers-only.

### 2. Dispatch on engine — `pipeline/extraction_vlm_pipeline.py`
In `__init__`, replace the hardcoded `TransformersExtractionModel` with a
type/framework switch mirroring `VlmPipeline._initialize_legacy_vlm_models`:

```python
opts = pipeline_options.vlm_options
if isinstance(opts, ApiVlmOptions):
    self.vlm_model = ApiVlmModel(
        enabled=True,
        enable_remote_services=pipeline_options.enable_remote_services,
        vlm_options=opts,
    )
else:  # InlineVlmOptions -> local transformers extraction model
    self.vlm_model = TransformersExtractionModel(... as today ...)
```

`_extract_data` / `_determine_status` are unchanged — `ApiVlmModel` returns real
`stop_reason`s, which the existing LENGTH/STOP_SEQUENCE → PARTIAL_SUCCESS logic
already handles.

### 3. Ship an API preset — `datamodel/vlm_model_specs.py`
Add an `ApiVlmOptions` extraction preset targeting a **Granite/VAREX** endpoint
(e.g. Granite Vision 4.1 served on vLLM or Ollama, `response_format=PLAINTEXT`,
`temperature=0.0`), following the existing `GRANITE_VISION_OLLAMA` shape. Pair it
with `extraction_prompt_style = GRANITE_VISION` (VAREX). Users override `url`,
`headers` (bearer token), and `params["model"]`.

### 4. Make prompt construction engine-independent (the real design point)

Target models are **NuExtract** and **Granite Vision 4.1** — no commercial GPT
support needed. That pins the design, because these two already represent the
two prompt regimes, and Granite's is the generalizable one:

- **NuExtract** (`build_nuextract_inputs`): the raw template is fed through the
  model's *own* chat template via a special `template=` kwarg. Model-specific,
  not API-able — and that's fine, NuExtract isn't served over generic endpoints.
- **Granite Vision** (`build_granite_vision_inputs` → `_build_extraction_prompt`,
  `prompt_utils.py:102`): the template is wrapped in a plain-text instruction
  ("Extract structured data… Return a JSON object matching this schema… Return
  ONLY valid JSON") and fed through a **standard chat conversation**. This is
  exactly the VAREX format from the Granite model card, and plain-text + standard
  chat is exactly what an OpenAI-conformant endpoint consumes.

So the VAREX wrapper we need for the API engine **already exists** — it's just
trapped inside the transformers input builder. Two moves:

1. **Hoist the wrapper into the pipeline.** Move `_build_extraction_prompt` out
   of `build_granite_vision_inputs` up to the extraction pipeline's prompt
   assembly, keyed on `ExtractionPromptStyle`, so transformers / api / vllm all
   share it. NuExtract style → passthrough (model applies `template=`). Granite
   style → VAREX-wrapped plain text. The granite transformers builder then just
   applies the chat template to the already-finished text (drop its internal
   `_build_extraction_prompt` call — do **not** double-wrap).

2. **Make serialization style-aware too.** Template *serialization* also differs
   by regime, not just the wrapper: NuExtract wants a **sample instance** (current
   `_serialize_template`, via polyfactory); VAREX wants a real **JSON Schema**
   with field descriptions (the Granite card passes `{"type":"object",
   "properties":{…}}`). For a Pydantic model that's `model_json_schema()` instead
   of building a sample instance. Both serialization and wrapping become a
   function of prompt style and live in the one engine-independent place.

Net: prompt style — not engine — decides how the template becomes text; the
engine only decides how that text + image are executed. No GPT special-casing,
no NuExtract over-fitting.

**Naming nicety (optional):** `ExtractionPromptStyle.GRANITE_VISION` is really
"generic JSON-schema / VAREX extraction" and works for any instruction-tuned VLM
over the API, not only Granite. Consider a neutral alias (`SCHEMA` / `VAREX`)
with `GRANITE_VISION` kept as a deprecated alias. Not required for the feature.

### Engine × prompt-style matrix

| Prompt style | transformers (local) | API / vllm |
|--------------|----------------------|------------|
| NuExtract    | ✅ (special `template=`) | ✗ (model-specific format) |
| Granite / VAREX | ✅ | ✅ — **this is what the API engine unlocks** |

## Optional follow-on: local mlx / vllm engines
Cheap once the dispatch + shared prompt assembly exist: route `InlineVlmOptions`
with `inference_framework == MLX/VLLM` to the existing `HuggingFaceMlxModel` /
`VllmVlmModel`. They consume the VAREX-wrapped text like any chat model, so they
work for Granite/VAREX style but not NuExtract's special format. Defer until asked.

## Out of scope
- Chart extraction (`datamodel/chart_extraction_options.py`,
  `models/stages/chart_extraction`) — separate subsystem, untouched.
- There is no extraction CLI today; this stays a Python-API-only change.

## Testing
- Unit: extractor with an `ApiVlmOptions` preset against a mocked
  OpenAI-conformant endpoint (assert the serialized template reaches the request
  body and JSON is parsed into `ExtractedPageData.extracted_data`). Mirror
  `tests/test_extraction_vlm_streaming.py`, which already exercises the
  streaming/API request path.
- Guard: constructing the pipeline with `ApiVlmOptions` and
  `enable_remote_services=False` must raise `OperationNotAllowed` (inherited
  from `ApiVlmModel`).
- Keep existing local-model tests (`test_extraction.py`,
  `test_granite_vision_extraction.py`) green — CI-skipped, heavy datasets.

## Docs / packaging
- Update `docling/.agents/skills/docling/references/extraction.md` with an API
  engine example + `enable_remote_services=True`.
- Slim extras: API-only extraction needs no torch. Check whether a
  `models-vlm-api`-style extra should cover extraction so users can run remote
  extraction without `models-vlm-inline`. Note it in `slim-packaging.md`.

## Diff size estimate
~4 files for the core capability (options type, pipeline dispatch, one preset,
one test) + docs. No new modules, no new base classes.
