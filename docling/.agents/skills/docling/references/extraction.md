# Structured extraction (DocumentExtractor)

Extraction pulls selected fields from a source into JSON objects. Conversion
produces a full `DoclingDocument`. Install the formats and engine you use:

```bash
pip install "docling-slim[extract-core,format-pdf,models-vlm-inline]"
```

## Source and spec

```python
from pydantic import BaseModel
from docling.datamodel.extraction import ExtractionSpec
from docling.document_extractor import DocumentExtractor


class Invoice(BaseModel):
    bill_no: str
    total: float


extractor = DocumentExtractor()
spec = ExtractionSpec.from_pydantic(
    Invoice, instructions="Copy the invoice identifier exactly"
)
result = extractor.extract(source="invoice.pdf", spec=spec)
for item in result.items:
    print(item.scope, item.extracted_data, item.validation_status, item.errors)
```

`extract_all(source=[...], spec=spec, raises_on_error=False)` yields one
`DocumentExtractionResult` per source. The envelope's `input`, `status`, and
`errors` identify the owning document; `items` contains its ordered outcomes.
Each item retains raw output, errors, token/usage metadata, and stop reason.
`PageScope.page_no` is the absolute positive source page; `DocumentScope` has no
page number. Paginated inputs make one independent request per selected page,
including text. Unpaginated sources make one document-scoped text request.
Chunks are internal and are never caller inputs.

An `ExtractionSpec` has two fields with one role each:

| Spec has | Model gets | Output validation |
|---|---|---|
| `json_schema` | The converted schema | Against `json_schema` |
| `json_schema` + `native_template` | `native_template`, as is | Against `json_schema` |
| `native_template` | `native_template`, as is | None (`not_requested`) |

- `json_schema` is the contract: a JSON Schema (Draft 2020-12). It always
  defines the validation. Without a `native_template`, Docling also converts it
  into the model input (best effort). A schema with no equivalent in the model
  input is rejected before inference; the error tells you to add a
  `native_template`.
- `native_template` overrides the model input only. It is a plain object in the
  native template format of the model (today NuExtract, for example
  `{"bill_no": "verbatim-string", "total": "number"}`). It goes to the model as
  is, is not checked, and never changes the validation. Use it for native types
  such as `verbatim-string` that a JSON Schema cannot express.
- `instructions` is free text added to the prompt. Instructions alone are
  invalid: at least one of `json_schema` and `native_template` is required.

`ExtractionSpec.from_pydantic(Model, native_template=..., instructions=...)`
transfers `model_json_schema()`, without Python validators or generated sample
values. Docling does not check that the template and the schema agree; if they
do not, items get `validation_status="failed"`. Missing values are neither
repaired nor coerced. Validation is `not_requested` without a schema, `not_run`
after inference/parse failure, or `passed`/`failed`. Schema failure cannot report
success.

Do not confuse `native_template` with the deprecated `template=` argument of
`DocumentExtractor.extract()` (see "Legacy SDK compatibility" below). They are
different inputs.

## Models, templates, and engines

Configure `VlmExtractionPipelineOptions.vlm_options` using
`ExtractionVlmOptions.from_preset(...)`, and pass it through an
`ExtractionFormatOption` for each configured input format.

| Preset | Guidance | Implemented engines |
|---|---|---|
| `nuextract_2b` (default) | `native_template`, or best-effort schema conversion | Transformers; documented API transport |
| `granite_vision_4_1` | `json_schema` as text in the prompt; no `native_template` | Transformers; API (image channel) |
| `nuextract_3` | `native_template`, or best-effort schema conversion | Transformers; explicitly configured vLLM API |
| `lift` | `json_schema` as text in the prompt (required); no `native_template` | Transformers; explicitly configured vLLM API |

The NuExtract conversion maps objects, homogeneous arrays, scalar types, string
enums, `const` and one-value enums (as their scalar type), and nullable unions
(`type: [T, "null"]`, or `anyOf`/`oneOf` of one type plus null). Annotations and
constraints (`description`, `required`, `minimum`, `pattern`, ...) do not reach
the model, but they stay in the validation. A real union such as a `oneOf`
between two object shapes has no equivalent and is rejected; add a
`native_template` for it.

Generic chat presets (`granite_vision_4_1`, `lift`) have no native template
format and reject a spec with `native_template`. Use `json_schema` and
`instructions` there. They get the schema as text in the prompt;
`VlmExtractionPipelineOptions(strip_schema_annotations=True)` removes `title`,
`description`, `examples`, `default` and `$comment` from that text, while the
validation still uses the full schema. Each cached extractor prepares fresh specs per call.

API extraction requires `enable_remote_services=True` and an
`ApiVlmEngineOptions` endpoint. `output_mode="prompt_only"` is the default;
`schema_constrained` requires a `json_schema` and explicitly configured vLLM API.
The decoder constraint always comes from `json_schema`, also when a
`native_template` overrides the prompt.
It rejects unsupported decoder assertions before HTTP and never retries without
constraints after rejection. Original-schema validation runs in either mode.

NuExtract3 and Lift have offline contract coverage; their local/vLLM live
verification remains unrun. Installed NuExtract3 LM Studio is incompatible with
caller-template delivery; its preset rejects named LM Studio/Ollama/OpenAI
engines. Lift's weight license needs operator review. Presets do not certify
quality, capacity, or deployment eligibility. Qwen3.5/Gemma integrations are deferred.

## Formats and channels

PDF/IMAGE supply images. DOCX/HTML/MD supply text. DCLX supplies structured text
and archived page images where present. `input_channels` chooses `AUTO`, `IMAGE`,
`TEXT`, or `IMAGE_AND_TEXT`. AUTO prefers supported images and falls back to
text; mixed content is explicit. Forced channels must be supported by both
source and model. Granite accepts images only. All other implemented presets
accept text, images, and mixed content.

Paginated text is selected by provenance; ambiguous/unscoped text is rejected
rather than attributed to page 1. Markdown uses normalized input; other text
formats serialize their `DoclingDocument`. PDF rendering and image copies are
bounded and released as the pipeline advances.

## Service wire contract

`ExtractSourcesRequest(extraction_spec=spec, sources=[...], options=..., target=...)`
carries the extraction contract on the top-level `extraction_spec` field. The
top-level `target` independently selects an in-body or artifact-storage
destination. `options` (`ExtractDocumentsOptions`) is purely operational:
`extraction_preset` or `extraction_custom_config`, `extraction_output_mode`,
`extraction_input_channels`, absolute `page_range`, and `extraction_raw_text`.
Every field defaults. No Python classes, validators, grouping fields, or public
chunks are accepted over the wire.

`extraction_raw_text` is `on_failure` by default: `raw_text` is returned only for
items that failed (no `extracted_data`, an error, or a failed validation). Set it
to `always` to get the raw model answer for all items.

`extraction_output_mode="schema_constrained"` without
`extraction_spec.json_schema` is rejected by request validation (client
`ValidationError`, serve 422). docling-serve also rejects with 422 a
`native_template` for a generic chat preset, and the inputs that its operator
policy blocks (for example `native_template` when
`allow_extraction_native_template` is off, or `instructions` over the length
limit).

Sync and async clients expose `extract` / `extract_all` (in-body convenience,
same `spec=` argument and `DocumentExtractionResult` as the local
`DocumentExtractor`) and `submit_extract` (job handle, storage targets,
callbacks), which mirrors the wire request: `source, spec, options=...,
target=<destination>`. The endpoint is `/v1/extract/source/async`;
the returned job supports polling/watching/result retrieval. `submit_extract`
in-body results are `ExtractDocumentResponse.documents`, containing JSON-safe
`ExtractionDocumentResult`s with `source_index`, `source_uri`, `filename`,
`status`, `errors`, and canonical `items`. Presigned and storage
destinations return `PresignedUrlConvertResponse` /
`PresignedUrlConvertDocumentResponse`, as for `submit`. The per-document file is
`<stem>.extraction.json` with the artifact type `extraction`. `extract_all` runs
one job per source with bounded concurrency. Runtime backends are never
serialized. Deployment requires matching Jobkit/Serve contracts. See
[service-client.md](service-client.md).

## Legacy SDK compatibility

The released `template=` SDK input still accepts strings, dictionaries, Pydantic
classes and instances, emits a deprecation warning, and returns
`ExtractionResult.pages`. Classes retain legacy style-dependent sample/schema
serialization; they do not implicitly acquire new output validation. Unpaginated
sources require `spec=`. Released inline options/imports and `process_images()`
remain available. The unreleased service wire accepts only the explicit spec
and item envelopes, with no old `template`/`pages` aliases.
