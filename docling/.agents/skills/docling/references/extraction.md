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
