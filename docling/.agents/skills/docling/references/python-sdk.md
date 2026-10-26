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
data = doc.export_to_dict()                 # preserves the extracted structure
```

`result.status` reports success/failure; `result.document` is always a
`DoclingDocument`. For exact values, formulas, complex tables, or suspicious
output, use [quality-verification.md](quality-verification.md). Preserve raw
structured output and provenance alongside readable exports; successful
conversion does not establish accuracy against the source.

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
| `extract_form_fields` | Convert docling-parse PDF widgets (AcroForm) into fillable fields with keys from printed text |
| `do_code_enrichment` / `do_formula_enrichment` | Enrich code / formulas |
| `code_formula_options` | Code/formula model, crop margin (`expansion_factor`), and stopping repeated output (`stop_on_repetition`, on by default) |
| `ocr_options` | Choose/parametrize the OCR engine (see below) |
| `table_structure_options` | e.g. `TableFormerMode.ACCURATE` vs `FAST` |
| `heading_hierarchy_options` | Infer section-header levels; off by default (see below) |
| `use_reading_order_separators` | Use visible PDF rules as reading-order signals; on by default |
| `images_scale` / `generate_page_images` | Control rasterization |
| `accelerator_options` | Pick device / thread count |
| `artifacts_path` | Use pre-downloaded model artifacts (offline) |
| `enable_remote_services` | Gate all outbound HTTP (required for any remote model) |

`extract_form_fields=True` associates each widget with a field key from printed
text, using page geometry. The key can be text in or beside its cell, or a
paragraph that contains the widget. Checkbox and radio option text becomes an
option key; a shared prompt stays as ordinary text. Additional context becomes
a field hint. A section header or formula used as a key stays inside the key.
Fields keep their widget order and sit among the surrounding text by page
geometry. A widget in a detected table goes into its table cell, with a key
only from text in that same cell. PDF field
names are not used as keys, and PDF-specific widget metadata is not stored in
the `DoclingDocument`. Use `generate_parsed_pages=True` to retain raw widgets
on parsed pages when needed. Backends without page-local widgets produce no
fields; table cells need `do_table_structure=True`. Not available in the
service client, and not supported by the DocTags export.

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

OCR can also run on a remote KServe v2 / Triton server. Page crops are sent to
that server, so `enable_remote_services=True` is required:

```python
from docling.datamodel.pipeline_options import KserveV2OcrOptions

opts = PdfPipelineOptions(
    do_ocr=True,
    enable_remote_services=True,   # REQUIRED for KserveV2OcrOptions
    ocr_options=KserveV2OcrOptions(url="localhost:8001", model_name="rapidocr"),
)
```

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

## Native PDF pipeline (no models)

Extracts what the PDF already contains — one text item per native text cell, one
picture per embedded bitmap — with docling-parse only. Very fast, but the result
has no reading order, headings or tables. PDF input only.

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption
from docling_core.types.doc.page import TextCellUnit

pipeline_options = NativePdfPipelineOptions(
    text_cell_unit=TextCellUnit.LINE,  # or WORD / CHAR
    generate_page_images=True,  # parse *and* render each page; False = parse only
    images_scale=2.0,  # 144 DPI page images
    generate_picture_images=True,  # decode the embedded bitmaps
    # parser_threads defaults to all but one CPU thread (no model runs here, so
    # accelerator_options.num_threads does not apply)
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: NativePdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = converter.convert("report.pdf")
```

CLI equivalent: `docling report.pdf --pipeline native --from pdf`.

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

### Progress events

`DocumentConverter(show_progress=True)` prints progress bars for pages and
enrichment to stderr with no extra code. For your own UI, `progress_callback=...`
(on `convert` / `convert_all` for one call, or on the constructor for all
calls) receives typed events from
`docling.datamodel.progress` while documents convert: document start and end
(with `document_index` in the batch and the final `status`), pipeline phases,
`PageCompletedProgress` (`completed_pages` / `total_pages`) for PDF and image
pipelines, and `EnrichmentProgress` (`completed_items` / `total_items` per
enrichment step; match on the stable `step`, not on `label`). Enrichment runs
after the last page, so a page count at 100% does not mean the document is
done. The callback must be thread-safe when
`settings.perf.doc_batch_concurrency > 1`.

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

For a Transformers-format Whisper checkpoint, install
`docling-slim[feat-asr-transformers]` and select an explicit preset such as
`asr_model_specs.WHISPER_TINY_TRANSFORMERS`. The presets cover tiny, base,
small, medium, large, and large-v3-turbo. Set
`pipeline_options.artifacts_path` to a directory that contains `config.json`
and the Transformers model files, or to its parent directory with a
`openai--whisper-tiny` subdirectory. The local directory is used without a
Hugging Face lookup. For MLX Whisper, use `WHISPER_TINY_MLX` and provide an
MLX checkpoint with `config.json` and `weights.safetensors` or `weights.npz`.

Other explicit Transformers presets are `GRANITE_4_0_1B_SPEECH` and
`NEMOTRON_3_5_ASR_STREAMING`. These models return
one transcript item that covers the full audio duration. The Transformers
Whisper presets provide model segment timestamps.

The CLI can also transcribe: `docling interview.wav --to md --output /tmp/`.

Additional presets include `GRANITE_SPEECH_5_0_2B`,
`GRANITE_SPEECH_5_0_470M_TURBOCTC`, its `_NC` variant,
`GRANITE_SPEECH_4_1_2B`, its `_PLUS` variant, and
`NEMOTRON_SPEECH_STREAMING_EN_0_6B`. Copy a preset with
`.model_copy(deep=True)` before changing `revision`, `language`, or `prompt`.
Granite and TurboCTC use FP32 weights on MPS to avoid BF16 normalization failures.
Plus defaults to plain transcription; speaker and timestamp tags are not parsed
into tracks.
The `_NC` preset uses a noncommercial license.
For real decoding checks, use `scripts/validate_asr_models.py` in a repository
checkout. Repeat `--model` to select presets; without it, the script runs every
ASR preset on the same fragment. It is outside normal pytest collection.

Video files (`.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`) use `VideoPipeline` with
`VideoPipelineOptions`, which also samples frames. `max_sampled_frames` defaults
to `200` (set `None` for no limit). When `document_timeout` is set, all FFmpeg
calls share that budget; running out yields `PARTIAL_SUCCESS` with a `TIMEOUT`
error (`result.has_timeout_errors()`).

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

### Re-converting a saved DoclingDocument JSON

Converting a `.json` DoclingDocument again (e.g. to re-export it) ignores image
references to local files (paths, `file:` URIs), with a warning; embedded
`data:` images and `http(s)` URLs are kept. So a document saved with
`ImageRefMode.REFERENCED` comes back without its images. For a trusted file,
opt in (SDK only; the CLI always ignores them):

```python
from docling.datamodel.backend_options import DeclarativeBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, DoclingJSONFormatOption

converter = DocumentConverter(format_options={
    InputFormat.JSON_DOCLING: DoclingJSONFormatOption(
        backend_options=DeclarativeBackendOptions(enable_local_fetch=True)
    ),
})
```

Or save with `ImageRefMode.EMBEDDED` so the JSON carries its images.

## HTML: images on remote hosts

HTML image downloads are off by default. Enable them on the HTML backend
options; downloads only connect to public addresses, re-check every redirect
and stop at `max_remote_image_bytes`. An image larger than `max_image_pixels`
(width x height, default 89,478,485) is skipped with a warning.

```python
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

html_options = HTMLBackendOptions(
    fetch_images=True,
    enable_remote_fetch=True,
    headers={"Authorization": "Bearer TOKEN"},  # optional
    # Origins that receive `headers`. Default (None): only the source page's
    # origin; local files/streams then get no headers.
    headers_allowed_origins=["https://example.com", "https://cdn.example.com"],
)
converter = DocumentConverter(
    format_options={InputFormat.HTML: HTMLFormatOption(backend_options=html_options)}
)
```
## DOCX, XLSX and EPUB size limits

These formats are ZIP archives. Docling checks the decompressed sizes before it
parses them and fails the file with `ArchiveLimitError` (a `DocumentLoadError`)
when one is too large. Defaults: DOCX/XLSX 128 MiB per member, 512 MiB in total,
10,000 members; EPUB content documents 10 MB each, 100 MB in total. Raise them
for large trusted files with `max_file_bytes`, `max_total_bytes` and
`max_member_count` on `MsWordBackendOptions`, `MsExcelBackendOptions` or
`EpubBackendOptions`:

```python
from docling.datamodel.backend_options import MsWordBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, WordFormatOption

converter = DocumentConverter(format_options={
    InputFormat.DOCX: WordFormatOption(
        backend_options=MsWordBackendOptions(max_file_bytes=512 * 1024 * 1024)
    ),
})
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
