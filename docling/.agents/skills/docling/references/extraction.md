# Structured extraction (DocumentExtractor)

Beta feature. Conversion (`DocumentConverter`) turns a document into a full
`DoclingDocument`. **Extraction** (`DocumentExtractor`) does something different:
it pulls **specific, typed fields** out of a document according to a template —
e.g. invoice number and total from a scanned invoice, or a set of contract
fields. Use it when the user wants *values*, not the whole document.

Requires the `extract-core` extra (see [slim-packaging.md](slim-packaging.md)):

```bash
pip install "docling-slim[extract-core,format-pdf,models-vlm-inline]"
# (included in the full `docling` package)
```

## Entry point

```python
from docling.document_extractor import DocumentExtractor
from docling.datamodel.base_models import InputFormat

extractor = DocumentExtractor(allowed_formats=[InputFormat.PDF, InputFormat.IMAGE])
```

`extract(source, template, ...)` returns an `ExtractionResult`;
`extract_all(sources, template, ...)` returns an iterator of them. `source` is a
path, URL, or `DocumentStream`.

## Templates — four ways to describe what to pull

The `template` argument accepts a string, a dict, a Pydantic model **class**, or
a Pydantic model **instance** (`Union[str, dict, BaseModel, Type[BaseModel]]`).

```python
# 1. JSON-ish string
result = extractor.extract(source="invoice.pdf",
                           template='{"bill_no": "string", "total": "float"}')

# 2. dict template
result = extractor.extract(source="invoice.pdf",
                           template={"bill_no": "string", "total": "float"})

# 3. Pydantic model class (recommended — typed, self-documenting)
from pydantic import BaseModel

class Invoice(BaseModel):
    bill_no: str
    total: float

result = extractor.extract(source="invoice.pdf", template=Invoice)

# 4. Pydantic instance (fields double as examples / defaults)
result = extractor.extract(source="invoice.pdf",
                           template=Invoice(bill_no="INV-0001", total=0.0))
```

Prefer a **Pydantic model class** for durable schemas — it documents intent and
gives you validation on the way out.

## Choosing the engine

Extraction runs a vision model, configured through
`VlmExtractionPipelineOptions.vlm_options`. The prompt style travels **with the
spec** (`extraction_prompt_style` on the options object), so picking a preset
picks its style — you never set them separately:

- **NuExtract** (default, `NU_EXTRACT_2B_TRANSFORMERS`): local
  `numind/NuExtract-2.0-2B`. The template is consumed via the model's own chat
  template; this style is local-only.
- **Granite schema-instruction** (`GRANITE_VISION_4_1_TRANSFORMERS`,
  `GRANITE_VISION_4_1_API`): the serialized JSON Schema wrapped in a plain-text
  instruction prompt (the key-value extraction format from the Granite Vision
  model card). Works with local Granite Vision **and** any OpenAI-conformant
  endpoint serving it.

To run remotely, build the options from an `ApiExtractionVlmOptions` preset with
`enable_remote_services=True`, then wire them into the `DocumentExtractor` the
usual way. Only these two objects differ from a local setup:

```python
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITE_VISION_4_1_API

# The preset already carries the Granite schema-instruction style; just point it at your endpoint.
api_options = GRANITE_VISION_4_1_API.model_copy(update={
    "url": "https://my-endpoint/v1/chat/completions",
    "headers": {"Authorization": "Bearer <TOKEN>"},
    "params": {"model": "ibm-granite/granite-vision-4.1-4b"},
})
pipeline_options = VlmExtractionPipelineOptions(
    vlm_options=api_options,
    enable_remote_services=True,  # required for any remote engine
)
```

Pass `pipeline_options` to `ExtractionFormatOption(pipeline_cls=ExtractionVlmPipeline, ...)`
as usual; `extract(...)` / `extract_all(...)` then run inference on the remote
endpoint instead of locally.

## Reading the result

`ExtractionResult` has `status` (a `ConversionStatus`), `errors`, and `pages`
(one `ExtractedPageData` per page). Each page carries `extracted_data`
(the dict of pulled fields), `raw_text`, and per-page `errors`.

```python
from docling.datamodel.base_models import ConversionStatus

result = extractor.extract(source="invoice.pdf", template=Invoice)

if result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
    for page in result.pages:
        print(page.page_no, page.extracted_data)   # e.g. {"bill_no": "...", "total": 42.0}
else:
    print("extraction failed:", result.errors)
```

## Many documents

```python
for result in extractor.extract_all(
    source=["a.pdf", "b.pdf", "https://example.com/c.pdf"],
    template=Invoice,
    raises_on_error=False,     # keep going past individual failures
):
    print(result.input.file.name, result.status)
```

See [python-sdk.md](python-sdk.md) for the same status/error handling pattern on
the conversion side, and [service-client.md](service-client.md) to run
extraction-style workloads against a remote service.
