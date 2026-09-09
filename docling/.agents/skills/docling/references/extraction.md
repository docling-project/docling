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

- **NuExtract** (default, `NU_EXTRACT_2B_TRANSFORMERS`; remote `NU_EXTRACT_API`):
  `numind/NuExtract-2.0`. The template is consumed via the model's own chat
  template (carried out-of-band, not in the message content). NuExtract is the
  only style that can take a **text** payload, so it drives the text-only formats
  below.
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

## Formats and channels

Extraction accepts more than paginable images. The **format** decides which
payload **channels** are available; you provide only the file, the backend is
chosen automatically (as in convert):

| Formats | Channels offered | Default (`AUTO`) |
|---------|------------------|------------------|
| PDF, IMAGE | page image | image |
| DOCX, HTML, MD | document text | text |
| DCLX | page image **and** text | image |

DCLX (a saved `DoclingDocument` archive) is the structure-plus-images format: it
restores both the page images and the structured text from the archive, so it is
the one format that can drive the combined channel.

`input_channels` (`ChannelSelection`, default `AUTO`) picks the channel:

- `AUTO` — page image if the format has one, else text (reproduces today's
  PDF/IMAGE behavior; DCLX defaults to image).
- `IMAGE` / `TEXT` — force one channel. Requesting a channel a format cannot
  provide is a loud error (e.g. `IMAGE` on DOCX, or `TEXT` with a Granite spec,
  which cannot take text).
- `IMAGE_AND_TEXT` — explicit opt-in, sends each page's image **and** that page's
  text (image first). Only formats offering both channels support it (DCLX);
  NuExtract only.

The text channel is markdown. Markdown input passes through as-is; DOCX/HTML/DCLX
are serialized from their `DoclingDocument` with convert's defaults, overridable
via `markdown_params` (a docling-core `MarkdownParams`).

Text extraction over a remote NuExtract endpoint:

```python
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_model_specs import NU_EXTRACT_API

api_options = NU_EXTRACT_API.model_copy(update={
    "url": "https://my-endpoint/v1/chat/completions",
    "headers": {"Authorization": "Bearer <TOKEN>"},
    "params": {"model": "numind/NuExtract-2.0-8B"},
})
pipeline_options = VlmExtractionPipelineOptions(
    vlm_options=api_options,
    enable_remote_services=True,
)
# A DOCX/HTML/MD source now resolves to the text channel automatically.
```

Image-bearing channels (`IMAGE`, `IMAGE_AND_TEXT`) run one model request per
page and yield one `ExtractedPageData` per page. A text-only document yields a
single `ExtractedPageData` (`page_no=1`) for the whole document. Batching
multiple pages into a single request is a later addition.

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
