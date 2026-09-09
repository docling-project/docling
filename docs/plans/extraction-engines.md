# Extraction engines

> Status: shipped with the post-implementation review decisions below. The three
> possible extensions are now resolved.

## Goal

Give structured extraction the same public model-spec and engine-options shape
as VLM conversion while preserving extraction-specific request semantics.

## Public configuration

`VlmExtractionPipelineOptions.vlm_options` is an `ExtractionVlmOptions`. It
pairs:

- an `ExtractionVlmModelSpec`, which owns the repository, prompt style, accepted
  channels, generation defaults, and template serialization; and
- engine options, which own runtime concerns such as device, endpoint, headers,
  timeout, concurrency, and parameter overrides.

The bundled presets are `nuextract_2b` and `granite_vision_4_1`. The named
configurations are:

- `NU_EXTRACT_2B_TRANSFORMERS`
- `GRANITE_VISION_4_1_TRANSFORMERS`
- `NU_EXTRACT_API`
- `GRANITE_VISION_4_1_API`

API parameters are derived from the model spec, including its model identifier,
then overridden by `ApiVlmEngineOptions.params`.

## Engine support

Extraction currently supports Transformers and the OpenAI-compatible API engine
variants. Construction rejects MLX, local vLLM, and auto-inline options because
their shared conversion interface requires one image and one prompt. It cannot
represent text-only requests, image-and-text requests, or NuExtract's
out-of-band template.

`ExtractionVlmPipeline` dispatches on `engine_options.engine_type` and passes the
same `ExtractionVlmOptions` object directly to either
`TransformersExtractionModel` or `ApiExtractionVlmModel`. There are no flat
inline/API extraction DTOs or lowering methods.

The Transformers model honors its engine's device, dtype, quantization,
remote-code, KV-cache, and compilation settings. The API model supports both
prompt styles and propagates transport and response failures to the pipeline.

## Prompts and channels

Prompt behavior belongs to `ExtractionVlmModelSpec`:

- NuExtract serializes the template as a sample instance when given a Pydantic
  model class and sends it through `chat_template_kwargs.template`. It accepts
  image and text content.
- Granite Vision serializes a Pydantic model class as JSON Schema and wraps it
  in a schema extraction instruction. It accepts images only.

`ChannelSelection.AUTO` chooses an image only when the source provides images
and the model accepts them, otherwise it chooses text. DCLX capability is based
on the selected pages' restored images, so an image-less DCLX archive falls back
to text. Explicit incompatible selections fail loudly.

PDF images are rendered lazily and limited by `max_size`; DCLX images are
resized when needed. Text extraction uses backend-normalized Markdown and
honors page ranges for paginated declarative documents.

## Compatibility

The released legacy surface remains only as a deprecation shim: a plain
`InlineVlmOptions`, including its serialized form, is adapted to
`ExtractionVlmOptions` and emits `DeprecationWarning`. Legacy API extraction
options introduced only on the development branch are not retained.

## Results and lifecycle

Invalid JSON, non-object JSON, filtered output, empty API responses, and model
errors cannot produce a successful extraction. A document is successful only
when it contains extracted data and no page or pipeline errors; mixed results
are partial successes. Input and page backends are unloaded after execution.

## Packaging and tests

API-only extraction imports without Transformers, torch, docling-parse,
pypdfium2, or qwen-vl-utils. Tests cover meaningful boundaries: engine dispatch
and options, remote-service authorization, API payloads and failures, channel
selection, text serialization and page ranges, DCLX fallback and cleanup,
bounded PDF streaming, image sizing, status mapping, local model loading, and
legacy adaptation.

## Post-implementation review decisions

The review found and resolved the following issues:

| Finding | Adopted decision |
|---------|------------------|
| Engine selection advertised more than execution supported. | Support is explicit: Transformers and OpenAI-compatible API variants are accepted; MLX, local vLLM, and auto-inline are rejected during option validation. Nothing silently falls back to Transformers. |
| Flat inline/API extraction DTOs duplicated the modern model-spec configuration. | The DTOs and lowering methods were removed. Both execution models consume `ExtractionVlmOptions` directly. |
| Legacy compatibility included branch-only API options. | Only the released `InlineVlmOptions` surface remains as deprecated compatibility. Object and serialized forms are adapted; the branch-only API compatibility path was removed. |
| Transformers engine options were lost while lowering the model spec. | The model now reads device, dtype, quantization, remote-code, KV-cache, compilation, repository, and revision settings from the modern spec and engine options. |
| API presets and overrides could omit the model identifier or override the wrong layer. | Model and generation defaults come from the spec; `ApiVlmEngineOptions.params` has final precedence. The API receives `model`, `temperature`, and `max_tokens` without duplicating those values in named presets. |
| Granite and NuExtract API execution had separate routing and transport parsing. | `ApiExtractionVlmModel` handles both prompt styles, while one shared OpenAI-compatible response path performs transport, response, usage, and stop-reason parsing. |
| API failures, empty content, filtered output, or malformed JSON could still report success. | These cases now produce page errors or failures. Success requires at least one JSON object and no page or pipeline errors; mixed results are partial successes. |
| Local extraction did not report meaningful token counts or stop reasons. | Generated-token counts exclude the input prompt, stop strings are removed, and length, stop-sequence, and end-of-sequence reasons are reported. |
| `max_size` was configured but ignored. | PDF render scale is capped before rendering and oversized DCLX images are resized. `scale` and `max_size` must be positive. |
| DCLX was assumed to contain page images. | Channel availability is inspected on the selected archived pages. `AUTO` falls back to text when images are absent; an explicit image request fails. |
| Extraction did not consistently release input resources. | The base extraction pipeline unloads the input backend in `finally`; streamed PDF page backends and transient images remain bounded and are released per page. |
| Serialized deprecated options were not migrated. | Serialized `InlineVlmOptions` are recognized and passed through the same warning-emitting adapter as object inputs. |
| Importing API-only extraction pulled in local model and PDF dependencies. | Transformers and format-specific backends are imported only when selected. API-only extraction can import without torch, Transformers, docling-parse, pypdfium2, or qwen-vl-utils. |
| `DocumentExtractor()` enabled formats without extraction backends. | Its defaults are limited to IMAGE, PDF, DOCX, HTML, MD, and DCLX. Explicit unsupported formats fail during configuration. |
| Tests restated presets or validated their own fakes. | Trivial checks were removed or replaced with behavioral coverage of dispatch, loading options, payloads, failures, status, page ranges, sizing, streaming, fallback, cleanup, compatibility, and slim imports. |
| Implementation comments and the plan described obsolete development stages. | Decision-history comments were removed from production code and this plan now describes the resulting architecture. |

## Resolved extension decisions

### Local MLX, vLLM, and auto-inline

Decided: complete local-engine parity is not a product requirement now. These
engines stay explicitly unsupported rather than partially emulated. Local
extraction is served by the Transformers engine, and vLLM is reached through the
OpenAI-compatible API path. Adapters or a generalized engine contract will be
designed only if a concrete local MLX/vLLM extraction need appears.

### API page concurrency

Decided: keep the one-live-page memory bound. Document extraction still issues
one page request at a time, so API `concurrency` applies only when the model
receives a batch directly. A bounded producer/consumer path will be added only
if a real many-page API workload shows that per-page latency dominates.

### Text grouping and result numbering

Decided: keep whole-document text extraction — one request, one
`ExtractedPageData` with `page_no=1`. No page grouping or multi-page result
numbering is introduced.

Oversized input is rejected rather than allowed to overrun the model. The
Transformers engine enforces `ExtractionVlmModelSpec.max_input_tokens`: when the
tokenized input exceeds the configured limit, the request fails before
generation with a clear error, which the pipeline records as a page error and a
document failure. The limit is disabled by default (`None`). The API engine is
not guarded here; OpenAI-compatible providers report their own context-length
errors, which already propagate as page errors or failures.

## Verification boundary

Focused extraction tests and changed-file validation pass. The model-backed
tests require local model artifacts and are skipped under the repository's CI
condition. In the current Python 3.14 development environment, collecting them
without the CI condition aborts while importing the installed MLX extension
through unrelated ASR model detection; this is not handled in the extraction
implementation.
