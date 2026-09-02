# Docling Python SDK reference

Use the Python SDK when you need more than a one-off conversion: tuning the
pipeline, chunking for RAG, inspecting document structure, or embedding Docling
in an application. The entry point is `DocumentConverter`; behavior is
controlled by **`PipelineOptions`** subclasses passed per input format.

## Minimal conversion

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("report.pdf")   # local path or http(s) URL

doc = result.document                       # a DoclingDocument
print(doc.export_to_markdown())
data = doc.export_to_dict()                 # structured / lossless
```

`result.status` reports success/failure; `result.document` is always a
`DoclingDocument`.

## Configuring the pipeline with PipelineOptions

`DocumentConverter(format_options=...)` maps an **`InputFormat`** to a
**`FormatOption`** carrying the pipeline options for that format.

> **API note (Docling 2.81+):** `format_options` keys must be `InputFormat`
> enum members with the matching `FormatOption` value — e.g.
> `{InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(...))}`.
> Using string keys like `{"pdf": PdfPipelineOptions(...)}` fails at runtime with
> `AttributeError: 'PdfPipelineOptions' object has no attribute 'backend'`.

### Standard PDF pipeline

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(
    do_ocr=True,                  # OCR scanned/image regions (False to skip)
    do_table_structure=True,      # detect table structure (False = faster)
    do_code_enrichment=False,     # code understanding
    do_formula_enrichment=False,  # formula understanding
    generate_page_images=False,   # keep page raster images on the document
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)
result = converter.convert("report.pdf")
```

Useful `PdfPipelineOptions` / base fields:

| Field | Purpose |
|---|---|
| `do_ocr` | Run OCR (default engine EasyOCR) |
| `do_table_structure` | Detect table structure |
| `tagged_structure` | `"off"` (default), `"prefer"` or `"require"`: derive layout, heading levels and alt text from a tagged PDF's structure tree (docling-parse backend) |
| `do_code_enrichment` / `do_formula_enrichment` | Enrich code / formulas |
| `ocr_options` | Choose/parametrize the OCR engine (see below) |
| `table_structure_options` | e.g. `TableFormerMode.ACCURATE` vs `FAST` |
| `heading_hierarchy_options` | Infer section-header levels; off by default (see below) |
| `images_scale` / `generate_page_images` | Control rasterization |
| `accelerator_options` | Pick device / thread count |
| `artifacts_path` | Use pre-downloaded model artifacts (offline) |
| `enable_remote_services` | Gate all outbound HTTP (required for any remote model) |

### Choosing an OCR engine

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, RapidOcrOptions, TesseractOcrOptions, OcrMacOptions,
)

# Default is EasyOCR:
opts = PdfPipelineOptions(do_ocr=True)
# Alternatives:
opts = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions())    # lightweight
opts = PdfPipelineOptions(do_ocr=True, ocr_options=TesseractOcrOptions())# system Tesseract
opts = PdfPipelineOptions(do_ocr=True, ocr_options=OcrMacOptions())      # macOS only
```

Each engine is an optional dependency — see [slim-packaging.md](slim-packaging.md).

### Recovering heading levels

PDF headings all come out at `level=1` unless this stage is enabled. It infers the level from the
PDF bookmarks, the outline numbering (`PART I` / `1.` / `1.1` / `(a)`) and the font styling, in
that order of precedence.

```python
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)

opts = PdfPipelineOptions(
    heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
    generate_parsed_pages=True,  # required by the font-style signal only
)
```

The service equivalent is `do_pdf_heading_hierarchy` plus `pdf_heading_hierarchy_options`.

### Selecting the accelerator

```python
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions

opts = PdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA, num_threads=8),
)
```

## VLM pipeline (local inference)

Processes each page as an image through a vision-language model, replacing the
layout + OCR stack. Best for complex layouts, handwriting, and formulas.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.pipeline.vlm_pipeline import VlmPipeline

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    generate_page_images=True,
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
result = converter.convert("report.pdf")
```

Model presets (`docling.datamodel.vlm_model_specs`):

| Preset | Backend | Device | Notes |
|---|---|---|---|
| `GRANITEDOCLING_TRANSFORMERS` | HF Transformers | CPU/GPU | Default (CLI `--vlm-model granite_docling`) |
| `SMOLDOCLING_TRANSFORMERS` | HF Transformers | CPU/GPU | Lighter (CLI `--vlm-model smoldocling`) |
| `GRANITEDOCLING_VLLM` | vLLM | GPU | High-throughput batch |
| `GRANITEDOCLING_MLX` | MLX | Apple MPS | M-series Macs |

### Hybrid mode (`force_backend_text`) — SDK only

Use deterministic PDF text extraction for text regions while routing images and
tables through the VLM. Reduces hallucination on text-heavy pages.

```python
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    force_backend_text=True,     # hybrid
    generate_page_images=True,
)
```

## VLM pipeline (remote API) — SDK only

Send page images to any OpenAI-compatible endpoint (vLLM, LM Studio, Ollama, or
a hosted API). Endpoint URL / model / key configuration is only available here,
not on the CLI.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

vlm_opts = ApiVlmOptions(
    url="http://localhost:8000/v1/chat/completions",
    params=dict(model="ibm-granite/granite-docling-258M", max_tokens=4096),
    headers={"Authorization": "Bearer YOUR_KEY"},  # omit if not needed
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    timeout=120,
)
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_opts,
    generate_page_images=True,
    enable_remote_services=True,   # REQUIRED — gates all outbound HTTP
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )
    }
)
result = converter.convert("report.pdf")
```

`enable_remote_services=True` is mandatory for any remote model — Docling blocks
outbound HTTP by default.

> To offload the *entire* conversion to a remote service (rather than just the
> VLM step), use the Service Client — see [service-client.md](service-client.md).

## Chunking for RAG

Chunking is SDK-only. The **hybrid chunker** splits by heading hierarchy, then
subdivides oversized sections by token count. Pass a `BaseTokenizer` object
(the tokenizer API changed in docling-core 2.8.0 — a raw string no longer works).

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

tokenizer = HuggingFaceTokenizer.from_pretrained(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
)
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

for chunk in chunker.chunk(result.document):
    embed_text = chunker.contextualize(chunk)   # heading-prefixed text to embed
    print(chunk.meta.headings)                  # heading breadcrumb
    print(chunk.meta.origin.page_no)            # source page
```

OpenAI tokenizer (requires `pip install 'docling-core[chunking-openai]'`):

```python
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
    max_tokens=8192,
)
```

Chunking needs the `feat-chunking` extra — see [slim-packaging.md](slim-packaging.md).

## Inspecting document structure

```python
doc = result.document

for item, level in doc.iterate_items():
    if item.label.name == "SECTION_HEADER":
        print(f"{'#' * level} {item.text}")

for table in doc.tables:
    print(table.export_to_dataframe())   # pandas DataFrame
    print(table.export_to_markdown())

for picture in doc.pictures:
    print(picture.caption_text(doc))     # caption if present
```

## Batch conversion & error handling

Convert many sources with `convert_all`, and don't let one bad file abort the
run. Check `result.status` (a `ConversionStatus`) per document.

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus, InputFormat

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.DOCX],   # reject everything else
)

sources = ["a.pdf", "b.docx", "https://example.com/c.pdf"]
for result in converter.convert_all(sources, raises_on_error=False):
    if result.status == ConversionStatus.SUCCESS:
        print(result.input.file.name, "->", len(result.document.pages), "pages")
    elif result.status == ConversionStatus.PARTIAL_SUCCESS:
        print(result.input.file.name, "partial:", result.errors)
    else:  # FAILURE / SKIPPED
        print(result.input.file.name, "failed:", result.errors)
```

- `raises_on_error=True` (default) raises on the first failure — set `False` to
  collect results for every input instead.
- `ConversionStatus` values: `SUCCESS`, `PARTIAL_SUCCESS`, `FAILURE`,
  `SKIPPED`, `PENDING`, `STARTED`.
- `allowed_formats=` restricts which input formats the converter will accept.

## Audio / video (ASR)

Docling transcribes audio (and video) into a `DoclingDocument` via an ASR
pipeline. Needs the `format-audio` (or `format-video`) extra.

```python
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
result = converter.convert("interview.wav")
print(result.document.export_to_markdown())   # timestamped transcript
```

The CLI can also transcribe: `docling interview.wav --to md --output /tmp/`.

## Exporting images and tables

To keep and export page/figure images, tell the pipeline to generate them, then
choose how images are referenced on export.

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions

opts = PdfPipelineOptions(
    generate_page_images=True,
    generate_picture_images=True,
    images_scale=2.0,            # higher-resolution rasters
)
```

```python
from docling_core.types.doc import ImageRefMode

doc = result.document

# Save Markdown/HTML with images embedded (base64) or referenced (separate files)
doc.save_as_markdown("out.md", image_mode=ImageRefMode.EMBEDDED)
doc.save_as_markdown("out.md", image_mode=ImageRefMode.REFERENCED)
doc.save_as_html("out.html", image_mode=ImageRefMode.REFERENCED)

# Tables → pandas / CSV
for table in doc.tables:
    df = table.export_to_dataframe(doc=doc)
    df.to_csv("table.csv", index=False)

# Figures → PNG files (needs generate_picture_images=True)
for i, pic in enumerate(doc.pictures):
    img = pic.get_image(doc)
    if img is not None:
        img.save(f"figure_{i}.png")
```

## Offline / air-gapped models

Pre-download model artifacts, then point conversions at them so no network is
needed at run time.

```bash
docling-tools models download --output-dir /models   # fetch artifacts
```

```python
from pathlib import Path
from docling.datamodel.pipeline_options import PdfPipelineOptions

opts = PdfPipelineOptions(artifacts_path=Path("/models"))
```

`artifacts_path` (available on all `PipelineOptions`) overrides the default
Hugging Face cache. You can also set `HF_HOME` to relocate that cache. The CLI
equivalent is `docling report.pdf --artifacts-path /models`.

## Checking versions

```python
from importlib.metadata import version
print(version("docling"), version("docling-core"))
```
