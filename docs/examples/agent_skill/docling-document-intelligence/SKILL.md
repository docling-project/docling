---
name: docling-document-intelligence
description: >
  Parse, convert, chunk, and analyze documents using Docling. Use this skill
  when the user provides a document (PDF, DOCX, PPTX, HTML, image) as a file
  path or URL and wants to: extract text or structured content, convert to
  Markdown or JSON, chunk the document for RAG ingestion, analyze document
  structure (headings, tables, figures, reading order), or run quality
  evaluation with iterative pipeline tuning. Triggers: "parse this PDF",
  "convert to markdown", "chunk for RAG", "extract tables", "analyze document
  structure", "prepare for ingestion", "process document", "evaluate docling
  output", "improve conversion quality".
license: MIT
compatibility: Requires Python 3.10+, docling>=2.81.0, docling-core>=2.67.1
metadata:
  author: docling-project
  version: "2.0"
  upstream: https://github.com/docling-project/docling
allowed-tools: Bash(docling:*) Bash(python3:*) Bash(pip:*)
---

# Docling Document Intelligence Skill

Use this skill to parse, convert, chunk, and analyze documents with Docling.
It handles both local file paths and URLs, and outputs either Markdown or
structured JSON (`DoclingDocument`).

Conversion uses the **`docling` CLI** (installed with `pip install docling`).
The Python API is used only for features the CLI does not expose (chunking,
VLM remote-API endpoint configuration, hybrid `force_backend_text` mode).

## Scope

| Task | Covered |
|---|---|
| Parse PDF / DOCX / PPTX / HTML / image | ✅ |
| Convert to Markdown | ✅ |
| Export as DoclingDocument JSON | ✅ |
| Chunk for RAG (hybrid: heading + token) | ✅ (Python API) |
| Analyze structure (headings, tables, figures) | ✅ (Python API) |
| OCR for scanned PDFs | ✅ (auto-enabled) |
| Multi-source batch conversion | ✅ |

## Step-by-Step Instructions

### 1. Resolve the input

Determine whether the user supplied a **local path** or a **URL**.
The `docling` CLI accepts both directly.

```bash
docling path/to/file.pdf
docling https://example.com/a.pdf
```

### 2. Choose a pipeline

Docling has two pipeline families. Pick based on document type and hardware.

| Pipeline | CLI flag | Best for | Key tradeoff |
|---|---|---|---|
| **Standard** (default) | `--pipeline standard` | Born-digital PDFs, speed | No GPU needed; OCR for scanned pages |
| **VLM** | `--pipeline vlm` | Complex layouts, handwriting, formulas | Needs GPU; slower |

See [pipelines.md](pipelines.md) for the full decision matrix, OCR engine table
(EasyOCR, RapidOCR, Tesseract, macOS), and VLM model presets.

### 3. Convert the document

#### CLI (preferred for straightforward conversions)

```bash
# Markdown (default output)
docling report.pdf --output /tmp/

# JSON (structured, lossless)
docling report.pdf --to json --output /tmp/

# VLM pipeline
docling report.pdf --pipeline vlm --output /tmp/

# VLM with specific model
docling report.pdf --pipeline vlm --vlm-model granite_docling --output /tmp/

# Custom OCR engine
docling report.pdf --ocr-engine tesserocr --output /tmp/

# Disable OCR or tables for speed
docling report.pdf --no-ocr --output /tmp/
docling report.pdf --no-tables --output /tmp/

# Remote VLM services
docling report.pdf --pipeline vlm --enable-remote-services --output /tmp/
```

The CLI writes output files to the `--output` directory, named after the
input file (e.g. `report.pdf` → `report.md` or `report.json`).

**CLI reference:** <https://docling-project.github.io/docling/reference/cli/>

#### Python API (for advanced features)

Use the Python API when you need features the CLI does not expose:
chunking, VLM remote-API endpoint configuration, or hybrid
`force_backend_text` mode.

**Docling 2.81+ API note:** `DocumentConverter(format_options=...)` expects
`dict[InputFormat, FormatOption]` (e.g. `InputFormat.PDF` → `PdfFormatOption`).
Using string keys like `{"pdf": PdfPipelineOptions(...)}` fails at runtime with
`AttributeError: 'PdfPipelineOptions' object has no attribute 'backend'`.

**Standard pipeline (default):**
```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

converter = DocumentConverter()
result = converter.convert("report.pdf")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=True, do_table_structure=True),
        ),
    }
)
result = converter.convert("report.pdf")
```

**VLM pipeline — local (GraniteDocling via HF Transformers):**
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

**VLM pipeline — remote API (vLLM / LM Studio / Ollama):**

This is only available via the Python API; the CLI does not expose endpoint
URL, model name, or API key configuration.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

vlm_opts = ApiVlmOptions(
    url="http://localhost:8000/v1/chat/completions",
    params=dict(model="ibm-granite/granite-docling-258M", max_tokens=4096),
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    timeout=120,
)
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_opts,
    generate_page_images=True,
    enable_remote_services=True,  # required — gates all outbound HTTP
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

**Hybrid mode (force_backend_text) — Python API only:**

Uses deterministic PDF text extraction for text regions while routing
images and tables through the VLM. Reduces hallucination on text-heavy pages.

```python
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    force_backend_text=True,
    generate_page_images=True,
)
```

`result.document` is a `DoclingDocument` object in all cases.

### 4. Choose output format

**Markdown** (default, human-readable):
```bash
docling report.pdf --to md --output /tmp/
```
Or via Python: `result.document.export_to_markdown()`

**JSON / DoclingDocument** (structured, lossless):
```bash
docling report.pdf --to json --output /tmp/
```
Or via Python: `result.document.export_to_dict()`

> If the user does not specify a format, ask: "Should I output Markdown or
> structured JSON (DoclingDocument)?"

### 5. Chunk for RAG (hybrid strategy)

Chunking is only available via the Python API.

Default: **hybrid chunker** — splits first by heading hierarchy, then
subdivides oversized sections by token count. This preserves semantic
boundaries while respecting model context limits.

The tokenizer API changed in docling-core 2.8.0. Pass a `BaseTokenizer`
object, not a raw string:

**HuggingFace tokenizer (default):**
```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

tokenizer = HuggingFaceTokenizer.from_pretrained(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
)
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
chunks = list(chunker.chunk(result.document))

for chunk in chunks:
    embed_text = chunker.contextualize(chunk)
    print(chunk.meta.headings)        # heading breadcrumb list
    print(chunk.meta.origin.page_no)  # source page number
```

**OpenAI tokenizer (for OpenAI embedding models):**
```python
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
    max_tokens=8192,
)
# Requires: pip install 'docling-core[chunking-openai]'
```

For chunking strategies and tokenizer details, see the Docling documentation
on chunking and `HybridChunker`.

### 6. Analyze document structure

Use the `DoclingDocument` object directly to inspect structure:

```python
doc = result.document

for item, level in doc.iterate_items():
    if hasattr(item, 'label') and item.label.name == 'SECTION_HEADER':
        print(f"{'#' * level} {item.text}")

for table in doc.tables:
    print(table.export_to_dataframe())   # pandas DataFrame
    print(table.export_to_markdown())

for picture in doc.pictures:
    print(picture.caption_text(doc))     # caption if present
```

For the full API surface, see Docling's structure and table export docs.

### 7. Evaluate output and iterate (required for "best effort" conversions)

After **every** conversion where the user cares about fidelity (not quick
previews), run the bundled evaluator on the JSON export, then refine the
pipeline if needed. This is how the agent **checks its work** and **improves
the run** without guessing.

**Step A — Produce JSON and optional Markdown**

```bash
docling "<source>" --to json --output /tmp/
docling "<source>" --to md --output /tmp/
```

**Step B — Evaluate**

```bash
python3 scripts/docling-evaluate.py /tmp/<filename>.json --markdown /tmp/<filename>.md
```

If the user expects tables (invoices, spreadsheets in PDF), add
`--expect-tables`. Tighten gates with `--fail-on-warn` in CI-style checks.

The script prints a JSON report to stdout: `status` (`pass` | `warn` | `fail`),
`metrics`, `issues`, and `recommended_actions` (concrete `docling` CLI
flags to try next).

**Step C — Refinement loop (max 3 attempts unless the user says otherwise)**

1. If `status` is `warn` or `fail`, apply **one** primary change from
   `recommended_actions` (e.g. switch `--pipeline vlm`, change
   `--ocr-engine`, ensure tables are enabled).
2. Re-convert with `docling`, re-run `scripts/docling-evaluate.py`.
3. Stop when `status` is `pass`, or after 3 iterations — then summarize what
   worked and any remaining issues for the user.

**Step D — Self-improvement log (skill memory)**

After a successful pass **or** after the final iteration, append one entry to
[improvement-log.md](improvement-log.md) in this skill directory:

- Source type (e.g. scanned PDF, digital PDF, DOCX)
- First-run problems (from `issues`)
- Pipeline + flags that fixed or best mitigated them
- Final `status` and one line of subjective quality notes

This log is optional for the user to git-ignore; it is for **local** learning
so future runs on similar documents start closer to the right pipeline.

### 8. Agent quality checklist (manual, if script unavailable)

If `scripts/docling-evaluate.py` cannot run, still verify:

| Check | Action if bad |
|---|---|
| Page count matches source (roughly) | Re-run; try `--pipeline vlm` if layout is complex |
| Markdown is not near-empty | Enable OCR / VLM |
| Tables missing when visually obvious | Remove `--no-tables`; try `--pipeline vlm` |
| `\ufffd` replacement characters | Different `--ocr-engine` or `--pipeline vlm` |
| Same line repeated many times | `--pipeline vlm` or hybrid `force_backend_text` (Python API) |

## Common Edge Cases

| Situation | Handling |
|---|---|
| Scanned / image-only PDF | Standard pipeline with OCR, or `--pipeline vlm` for best quality |
| Password-protected PDF | `--pdf-password PASSWORD`; will raise `ConversionError` if wrong |
| Very large document (500+ pages) | Standard pipeline with `--no-tables` for speed |
| Complex layout / multi-column | `--pipeline vlm`; standard may misorder reading flow |
| Handwriting or formulas | `--pipeline vlm` only — standard OCR will not handle these |
| URL behind auth | Pre-download to temp file; pass local path |
| Tables with merged cells | `table.export_to_markdown()` handles spans; VLM often more accurate |
| Non-UTF-8 encoding | Docling normalises internally; no special handling needed |
| VLM hallucinating text | `force_backend_text=True` via Python API for hybrid mode |
| VLM API call blocked | `--enable-remote-services` (CLI) or `enable_remote_services=True` (Python) |
| Apple Silicon | `--vlm-model granite_docling` with MLX backend, or `GRANITEDOCLING_MLX` preset (Python API) |

## Pipeline reference

Full decision matrix, all OCR engine options, VLM model presets, and API
server configuration: [pipelines.md](pipelines.md)

## Output conventions

- Always report the number of pages and conversion status.
- When evaluation is in scope, report evaluator `status`, top `issues`, and
  which refinement attempt produced the final output.
- For Markdown output: wrap in a fenced code block only if the user will copy/paste it; otherwise render directly.
- For JSON output: pretty-print with `indent=2` unless the user specifies otherwise.
- For chunks: report total chunk count, min/max/avg token counts.
- For structure analysis: summarise heading tree + table count + figure count before going into detail.

## Dependencies

```bash
pip install docling docling-core
# For OpenAI tokenizer support:
pip install 'docling-core[chunking-openai]'
```

The `docling` CLI is included with the `docling` package — no separate install needed.

Check installed versions (prefer distribution metadata — `docling` may not set `__version__`):

```python
from importlib.metadata import version
print(version("docling"), version("docling-core"))
```
