# Python SDK reference — PDF attachments

This page documents the PDF attachment options exposed through the Python SDK.

## `PdfPipelineOptions`

`docling.datamodel.pipeline_options.PdfPipelineOptions` exposes two attachment controls.
They are also surfaced via the CLI as `--process-attachments` / `--attachments-max-depth`
and are fully described by their `Field` metadata — this file mirrors those descriptions
for discoverability from the SDK reference.

### `process_attachments`

```python
process_attachments: bool = False
Field(description="Process PDF embedded file attachments and convert each supported one into a separate Markdown file. Off by default.")
```

When `True`, the `DocumentConverter` extracts embedded files from PDFs via
`docling-parse` and converts each supported attachment into a sibling document.
Results appear on `ConversionResult.attachments` and as `AttachmentItem`s on
`result.document.attachments` (with `status`/`target`/`prov`). Export writes
sidecar Markdown files under `<stem>_attachments/`.

### `attachments_max_depth`

```python
attachments_max_depth: int = 1
Field(ge=0, description="Maximum recursion depth for attachment conversion. 0 means list attachments without converting; 1 (default) converts top-level attachments and lists their own attachments as depth_limited.")
```

`0` lists attachments without converting (all become `depth_limited`).
`1` (default) converts top-level attachments and marks any nested
attachments as `depth_limited`. Higher values recurse deeper.

Both fields are inherited by `ThreadedPdfPipelineOptions`.

## Example

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

pdf_opts = PdfPipelineOptions(process_attachments=True, attachments_max_depth=1)
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)})
result = converter.convert("document.pdf")
for att in result.document.attachments:
    print(att.name, att.status, att.target)
for child in result.attachments:
    child.document.save_as_markdown(f"/tmp/{child.input.file.stem}.md")
```

Also see `docs/examples/attachments.py`.
