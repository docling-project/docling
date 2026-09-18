# Extraction: engines, formats, and channels

Structured extraction gets the same public model-spec and engine-options shape as
VLM conversion, plus multi-format input and a text payload channel alongside the
existing image path.

## Public configuration

`VlmExtractionPipelineOptions.vlm_options` is an `ExtractionVlmOptions`
(`StagePresetMixin` + `VlmEngineOptionsMixin`). It pairs:

- an `ExtractionVlmModelSpec`, which owns the repository, prompt style, accepted
  channels (`accepts_image` / `accepts_text`), generation defaults, and template
  serialization; and
- engine options, which own runtime concerns: device, endpoint, headers, timeout,
  concurrency, and parameter overrides.

Presets are `nuextract_2b`, `nuextract_3`, `lift` and `granite_vision_4_1`. Named specs:
`NU_EXTRACT_2B_TRANSFORMERS`, `GRANITE_VISION_4_1_TRANSFORMERS`, `NU_EXTRACT_API`,
`GRANITE_VISION_4_1_API`.

`nuextract_3` is opt-in and implements the documented local Transformers and vLLM
API contracts. Those deployments have offline contract coverage, not live-model
verification. Its preset rejects LM Studio/Ollama/OpenAI named engines because
native caller-template delivery is not established there. The recorded installed
LM Studio runtime drops extraction template controls.

Supply `target=ExtractionTarget(template=ExtractionTemplate(format="nuextract",
value={"invoice": "verbatim-string", "total": "number"}), instructions="Copy the
invoice identifier exactly")`, or use an explicit output schema/Pydantic target
in the bounded conversion subset. Native values stay intact; `verbatim-string`
selects exact copying. The caller's extraction template reaches `template` and
`instructions` chat kwargs; no replacement of the checkpoint Jinja is needed.
Thinking defaults false and extraction mode structured. Different targets on a
cached extractor remain call-local, with one request per selected absolute page.

The local 258048 input-token check derives from the published 262144 context minus
4096 output tokens (the model-card non-thinking example); it is an unmeasured upper
bound. Configure smaller limits for your deployment. API server context limits
are owned by the server; no capacity or extraction-quality claim is implied.

`lift` is opt-in for local Transformers and explicitly configured vLLM API,
with offline contract coverage and live verification still open. It requires
`output_schema`; examples never become schemas. Supply both, for example:

```python
options = ExtractionVlmOptions.from_preset("lift")
target = ExtractionTarget(
    output_schema={
        "type": "object",
        "properties": {"invoice": {"type": "string"}, "total": {"type": "number"}},
        "required": ["invoice", "total"],
    },
    template=ExtractionTemplate(
        format="example_json", value={"invoice": "INV-123", "total": 12.5}
    ),
    instructions="Copy the invoice identifier exactly",
)
```

Omitting `template` selects schema-only guidance. Explicit examples reach the
shared generic renderer/API content intact and are labelled illustrations, not
source facts. Lift rejects the native `nuextract` dialect and legacy `template=`
calls; use `target=` with an output schema. Each selected absolute page gets its
own request, including for text and mixed channels. This does not reproduce the
reference implementation's joint multi-page inference.

For vLLM, select `engine_options=ApiVlmEngineOptions(...)` and explicitly set
`output_mode="schema_constrained"` to send request-specific JSON Schema constraints.
The shared bounded subset rejects unsupported keywords before inference; provider
rejection never falls back. Prompt-only mode preserves the full original schema
for validation. Lift's advice to avoid enums, unions, references and
`additionalProperties` concerns schema simplicity; it is not a blanket decoder
ban. Backend-specific compilation/quality remains to be verified live.

The pinned Lift tokenizer supplies EOS IDs 248044/248046 (`<|endoftext|>` /
`<|im_end|>`); local generation uses both, and the API preset sends those stop
strings. The checkpoint's existing chat template receives `enable_thinking=False`.
Its 12384 output setting and 249760 local input-token upper bound (262144 published
context minus output budget) are static and unmeasured. They do not establish
memory capacity or extraction accuracy.

Lift's code is Apache 2.0; its weights have a
[modified OpenRAIL-M license](https://github.com/datalab-to/lift/blob/4ff031b8c83b44bb123d7eda42907b22ec1e1e56/MODEL_LICENSE),
including revenue/funding and competitive-use restrictions, attribution and
share-alike provisions. Operators must review the actual terms and allow-list
the deployment; adding a preset does not authorize deployment.

API parameters derive from the model spec (model identifier, temperature,
max_tokens) and are then overridden by `ApiVlmEngineOptions.params`, which has
final precedence. Named presets do not restate those values.

## Engines

Extraction supports the Transformers and OpenAI-compatible API engines only.
Construction rejects MLX, local vLLM, and auto-inline: the shared conversion
engine interface (`VlmEngineInput`) is one image plus one prompt string and
cannot carry text-only requests, image+text content arrays, or NuExtract's
out-of-band template. Local extraction runs on Transformers; vLLM is reached
through the API path. Nothing silently falls back to Transformers.

`ExtractionVlmPipeline.__init__` dispatches on `engine_options.engine_type` and
passes the `ExtractionVlmOptions` object directly to `TransformersExtractionModel`
or `ApiExtractionVlmModel`. There are no flat inline/API extraction DTOs or
lowering methods. The Transformers model reads device, dtype, quantization,
remote-code, KV-cache, compilation, repository, and revision from the spec and
engine options. `ApiExtractionVlmModel` handles both prompt styles over one shared
OpenAI-compatible response path (transport, response, usage, and stop-reason
parsing).

## Formats and source shape

Each input opens into an `InputDocument` via a format-keyed backend
(`_get_default_extraction_option`); backends are chosen automatically, never
user-facing. `DocumentExtractor()` defaults to IMAGE, PDF, DOCX, HTML, MD, and
DCLX; explicit unsupported formats fail at configuration. `ExtractionFormatOption.backend`
is optional — an override that sets only `pipeline_options` gets its backend filled
from the default map in `DocumentExtractor.__init__`.

| Format class | Formats | Offers images | Offers text |
|---|---|---|---|
| image-paginable | PDF, IMAGE | rendered pages | — |
| text-only | DOCX, HTML, MD | — | `DoclingDocument` |
| structure + images | DCLX | restored from archive | `DoclingDocument` |

The pipeline normalizes any input to a source view exposing page images and/or a
`DoclingDocument`, from which channels are drawn.

## Channels

`input_channels: ChannelSelection = AUTO`. The payload is an ordered list of
content items (image and/or text).

- `AUTO` = (what the format offers) ∩ (what the model accepts), preferring image.
  PDF/IMAGE/DCLX → image; DOCX/HTML/MD → text. Never combines automatically.
- `IMAGE` / `TEXT` force a single channel; `IMAGE_AND_TEXT` is an explicit opt-in.
- Forced channels are validated against both the format and the model capability
  with capability-worded errors. DCLX image availability is inspected on the
  selected archived pages, so an image-less archive falls back to text under `AUTO`
  and fails an explicit image request.

Validation is split: a `model_validator` on `VlmExtractionPipelineOptions` rejects
a forced channel that contradicts model capability at construction (no document
needed); format-dependent checks stay per-document in `_resolve_channel`.

## Prompts

Prompt behavior lives on `ExtractionVlmModelSpec`:

- **NuExtract** serializes a Pydantic model class as a sample instance and sends it
  out-of-band via `chat_template_kwargs.template` (local: `apply_chat_template`;
  remote: `extra_body.chat_template_kwargs.template`). Accepts image and text.
- **Granite Vision** serializes a Pydantic model class as JSON Schema wrapped in a
  schema-extraction instruction. Image only — absent from the text and combined
  channels.

## Text payload

The text channel uses backend-normalized Markdown. `InputFormat.MD` passes through
as raw file content (no `DoclingDocument` round-trip); DOCX/HTML/DCLX serialize
from the `DoclingDocument` using docling-core's markdown serializer with convert's
defaults. `_get_text_from_input` honors `input_doc.limits.page_range`, restricting
serialization to the pages in range (default range is byte-for-byte the whole
document; MD passthrough is unpaginated and unaffected).

## Sizing and resources

`scale` and `max_size` must be positive. PDF render scale is capped before
rendering; oversized DCLX images are resized. PDF pages are streamed and released
per page; the base pipeline unloads the input backend in `finally`.

The Transformers engine enforces `ExtractionVlmModelSpec.max_input_tokens`
(default `None` = disabled): input over the limit fails before generation as a
page error and document failure. The API engine relies on the provider's own
context-length errors, which propagate the same way.

## Results

Extraction is whole-document: one request, one `ExtractedPageData` with
`page_no=1`. Request and result types are already collections, so future page
grouping slots in without an API change. Generated-token counts exclude the input
prompt; stop strings are removed; length, stop-sequence, and end-of-sequence
reasons are reported. Invalid JSON, non-object JSON, filtered output, empty API
responses, and model errors all produce page errors or failures — success
requires at least one JSON object and no page or pipeline errors; mixed results
are partial successes.

## Compatibility

The released `main` surface is kept behind a deprecation shim: a plain
`InlineVlmOptions` as `vlm_options` (object or serialized form) is adapted to
`ExtractionVlmOptions` via a `model_validator` that emits `DeprecationWarning`.
Branch-only flat API options and lowering DTOs are not retained. `process_images`
remains as an image-only adapter so convert-side callers are untouched.

## Packaging and tests

API-only extraction imports without torch, Transformers, docling-parse, pypdfium2,
or qwen-vl-utils — local and format-specific backends import only when selected.
Tests cover engine dispatch and options, remote-service authorization, API
payloads and failures, channel selection, text serialization and page ranges,
DCLX fallback and cleanup, bounded PDF streaming, image sizing, status mapping,
local model loading, and legacy adaptation.

## Deferred

- **Page grouping / output granularity** (`PER_PAGE` / `CHUNKED` / `WHOLE_DOCUMENT`),
  the context budget, and per-page result numbering — request/result collection
  types are already in place for it.
- **API page concurrency** — extraction issues one page request at a time;
  `concurrency` applies only when a model receives a batch directly.
- **Local MLX / vLLM engine variants** — each needs its own extraction model until
  the engine layer is generalized.
- **More declarative formats** (XML, AsciiDoc, CSV) — one entry each.
- **Granite text mode** — the model cannot take a text payload.
