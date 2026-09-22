# Structured extraction (DocumentExtractor)

Extraction pulls selected fields from a source into JSON objects. Conversion
produces a full `DoclingDocument`. Install the formats and engine you use:

```bash
pip install "docling-slim[extract-core,format-pdf,models-vlm-inline]"
```

## Source and target

```python
from pydantic import BaseModel
from docling.datamodel.extraction import ExtractionTarget, ExtractionTemplate
from docling.document_extractor import DocumentExtractor

class Invoice(BaseModel):
    bill_no: str
    total: float

extractor = DocumentExtractor()
target = ExtractionTarget.from_pydantic(
    Invoice,
    template=ExtractionTemplate(
        format="nuextract", value={"bill_no": "string", "total": "number"}
    ),
    instructions="Copy the invoice identifier exactly",
)
result = extractor.extract(source="invoice.pdf", target=target)
for item in result.items:
    print(item.scope, item.extracted_data, item.validation_status, item.errors)
```

`extract_all(source=[...], target=target, raises_on_error=False)` yields one
`DocumentExtractionResult` per source. The envelope's `input`, `status`, and
`errors` identify the owning document; `items` contains its ordered outcomes.
Each item retains raw output, errors, token/usage metadata, and stop reason.
`PageScope.page_no` is the absolute positive source page; `DocumentScope` has no
page number. Paginated inputs make one independent request per selected page,
including text. Unpaginated sources make one document-scoped text request.
Chunks are internal and are never caller inputs.

`output_schema` is the original JSON Schema used for validation.
`ExtractionTarget.from_pydantic(Model)` transfers `model_json_schema()`, without
Python validators or generated sample values. A tagged `template` guides the
model and never becomes a schema. At least one schema or template is required;
instructions alone are invalid. Missing values are neither repaired nor coerced.
Validation is `not_requested` without a schema, `not_run` after inference/parse
failure, or `passed`/`failed`. Schema failure cannot report success.

## Models, templates, and engines

Configure `VlmExtractionPipelineOptions.vlm_options` using
`ExtractionVlmOptions.from_preset(...)`, and pass it through an
`ExtractionFormatOption` for each configured input format.

| Preset | Guidance | Implemented engines |
|---|---|---|
| `nuextract_2b` (default) | Tagged native `nuextract`, or bounded schema conversion | Transformers; documented API transport |
| `granite_vision_4_1` | Tagged `example_json`, schema-only, or both | Transformers; API (image channel) |
| `nuextract_3` | Native `nuextract`, or bounded schema conversion | Transformers; explicitly configured vLLM API |
| `lift` | `example_json`, schema-only, or both; output schema required | Transformers; explicitly configured vLLM API |

Generic caller examples are explicit JSON values, for example:

```python
target = ExtractionTarget.from_pydantic(
    Invoice,
    template=ExtractionTemplate(
        format="example_json", value={"bill_no": "INV-42", "total": 4.2}
    ),
    instructions="Read values from the source; examples are illustrations",
)
```

Select a generic preset for that target. NuExtract's native dialect describes
types, not sample data. Unsupported formats/schema conversion fail before
inference; examples do not supply hidden schemas. Each cached extractor prepares
fresh targets per call.

API extraction requires `enable_remote_services=True` and an
`ApiVlmEngineOptions` endpoint. `output_mode="prompt_only"` is the default;
`schema_constrained` requires an output schema and explicitly configured vLLM API.
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

`ExtractSourcesRequest(extraction_target=target, sources=[...], options=..., target=...)`
carries the extraction contract on the top-level `extraction_target` field. The
top-level `target` independently selects an in-body or artifact-storage
destination. `options` (`ExtractDocumentsOptions`) is purely operational — model
selection, channel, absolute `page_range`, and `output_mode` — and every field
defaults. No Python classes, validators, grouping fields, or public chunks are
accepted over the wire.

Sync and async clients expose `extract` / `extract_all` (in-body convenience) and
`submit_extract` (job handle, storage targets, callbacks); all take unpacked
`source, extraction_target, options=..., target=...` arguments. The endpoint is
`/v1/extract/source/async`; the returned job supports polling/watching/result
retrieval. In-body results are `ExtractDocumentResponse.documents`, containing
JSON-safe `ExtractionDocumentResult`s with `source_index`, `source_uri`,
`filename`, `status`, `errors`, and canonical `items`. Presigned and storage
destinations return `PresignedUrlConvertResponse` /
`PresignedUrlConvertDocumentResponse`, as for `submit`. `extract_all` runs one job
per source with bounded concurrency. Runtime backends are never serialized.
Deployment requires matching Jobkit/Serve contracts; their migration follows this
Docling stage. See [service-client.md](service-client.md).

## Legacy SDK compatibility

The released `template=` SDK input still accepts strings, dictionaries, Pydantic
classes and instances, emits a deprecation warning, and returns
`ExtractionResult.pages`. Classes retain legacy style-dependent sample/schema
serialization; they do not implicitly acquire new output validation. Unpaginated
sources require `target=`. Released inline options/imports and `process_images()`
remain available. The unreleased service wire accepts only the explicit target
and item envelopes, with no old `template`/`pages` aliases.
