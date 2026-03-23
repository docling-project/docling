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
  version: "1.4"
  upstream: https://github.com/docling-project/docling
allowed-tools: Bash(python3:*) Bash(pip:*)
---

# Docling Document Intelligence Skill

Use this skill to parse, convert, chunk, and analyze documents with Docling.
It handles both local file paths and URLs, and outputs either Markdown or
structured JSON (`DoclingDocument`).

## Scope

| Task | Covered |
|---|---|
| Parse PDF / DOCX / PPTX / HTML / image | ✅ |
| Convert to Markdown | ✅ |
| Export as DoclingDocument JSON | ✅ |
| Chunk for RAG (hybrid: heading + token) | ✅ |
| Analyze structure (headings, tables, figures) | ✅ |
| OCR for scanned PDFs | ✅ (auto-enabled) |
| Multi-source batch conversion | ✅ |

## Step-by-Step Instructions

### 1. Resolve the input

Determine whether the user supplied a **local path** or a **URL**.

- Local path → pass as `str` or `Path` directly to `DocumentConverter`
- URL → pass as `str`; Docling fetches it automatically
- Multiple inputs → pass a list

```python
sources = ["path/to/file.pdf"]          # local
sources = ["https://example.com/a.pdf"] # URL
sources = ["file1.pdf", "file2.docx"]   # batch
```

### 2. Choose a pipeline

Docling has three pipelines. Pick based on document type and hardware.

| Pipeline | Best for | Key tradeoff |
|---|---|---|
| **Standard** (default) | Born-digital PDFs, speed | No GPU needed; OCR for scanned pages |
| **VLM local** | Complex layouts, handwriting, formulas | Needs GPU; slower |
| **VLM API** | Production scale, remote inference | Requires inference server |

See [pipelines.md](pipelines.md) for the full decision matrix, OCR engine table
(EasyOCR, RapidOCR, Tesseract, macOS; Tesseract CLI and future engines such as
Nemotron in Python only when supported by your Docling version), and VLM presets.

### 3. Convert the document

**Docling 2.81+ API note:** `DocumentConverter(format_options=...)` expects
`dict[InputFormat, FormatOption]` (e.g. `InputFormat.PDF` → `PdfFormatOption`).
Using string keys like `{"pdf": PdfPipelineOptions(...)}` fails at runtime with
`AttributeError: 'PdfPipelineOptions' object has no attribute 'backend'`.

**Standard pipeline (default):**
```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Defaults: standard PDF pipeline, OCR + tables
converter = DocumentConverter()
result = converter.convert(sources[0])

# Custom PdfPipelineOptions (same API as scripts/docling-convert.py --pipeline standard)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=True, do_table_structure=True),
        ),
    }
)
result = converter.convert(sources[0])
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
result = converter.convert(sources[0])
```

**VLM pipeline — remote API (vLLM / LM Studio / Ollama):**
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
result = converter.convert(sources[0])
```

`result.document` is a `DoclingDocument` object in all three cases.

### 3. Choose output format

**Markdown** (default, human-readable):
```python
md = result.document.export_to_markdown()
```

**JSON / DoclingDocument** (structured, lossless):
```python
import json
doc_json = result.document.model_dump()  # dict
doc_json_str = result.document.export_to_dict()  # serialisable dict
```

> If the user does not specify a format, ask: "Should I output Markdown or
> structured JSON (DoclingDocument)?"

### 4. Chunk for RAG (hybrid strategy)

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
    # contextualize() is the correct method for embedding-ready text —
    # it enriches chunk.text with heading breadcrumb metadata
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

### 5. Analyze document structure

Use the `DoclingDocument` object directly to inspect structure:

```python
doc = result.document

# Iterate headings
for item, level in doc.iterate_items():
    if hasattr(item, 'label') and item.label.name == 'SECTION_HEADER':
        print(f"{'#' * level} {item.text}")

# Extract tables
for table in doc.tables:
    print(table.export_to_dataframe())   # pandas DataFrame
    print(table.export_to_markdown())

# Extract figures / images
for picture in doc.pictures:
    print(picture.caption_text(doc))     # caption if present
```

For the full API surface, see Docling’s structure and table export docs.

### 6. Evaluate output and iterate (required for “best effort” conversions)

After **every** conversion where the user cares about fidelity (not quick
previews), run the bundled evaluator on the JSON export, then refine the
pipeline if needed. This is how the agent **checks its work** and **improves
the run** without guessing.

**Step A — Produce JSON and optional Markdown**

```bash
# From the bundle root (directory containing scripts/ and SKILL.md):
python3 scripts/docling-convert.py "<source>" --format json --out /tmp/docling-out.json
python3 scripts/docling-convert.py "<source>" --format markdown --out /tmp/docling-out.md
```

**Step B — Evaluate**

```bash
python3 scripts/docling-evaluate.py /tmp/docling-out.json --markdown /tmp/docling-out.md
```

If the user expects tables (invoices, spreadsheets in PDF), add
`--expect-tables`. Tighten gates with `--fail-on-warn` in CI-style checks.

The script prints a JSON report to stdout: `status` (`pass` | `warn` | `fail`),
`metrics`, `issues`, and `recommended_actions` (concrete `scripts/docling-convert.py`
flags to try next).

**Step C — Refinement loop (max 3 attempts unless the user says otherwise)**

1. If `status` is `warn` or `fail`, apply **one** primary change from
   `recommended_actions` (e.g. switch standard → VLM, change OCR engine,
   ensure tables are enabled, hybrid `--force-backend-text`).
2. Re-convert, re-export JSON, re-run `scripts/docling-evaluate.py`.
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

### 7. Agent quality checklist (manual, if script unavailable)

If `scripts/docling-evaluate.py` cannot run, still verify:

| Check | Action if bad |
|---|---|
| Page count matches source (roughly) | Re-run; try VLM if layout is complex |
| Markdown is not near-empty | Enable OCR / VLM |
| Tables missing when visually obvious | Enable table structure; try VLM |
| `\ufffd` replacement characters | Different OCR or VLM |
| Same line repeated many times | VLM or hybrid `--force-backend-text` |

## Common Edge Cases

| Situation | Handling |
|---|---|
| Scanned / image-only PDF | Standard pipeline with OCR, or VLM pipeline for best quality |
| Password-protected PDF | Will raise `ConversionError`; surface to user |
| Very large document (500+ pages) | Standard pipeline with `do_table_structure=False` for speed |
| Complex layout / multi-column | Prefer VLM pipeline; standard may misorder reading flow |
| Handwriting or formulas | VLM pipeline only — standard OCR will not handle these |
| URL behind auth | Pre-download to temp file; pass local path |
| Tables with merged cells | `table.export_to_markdown()` handles spans; VLM pipeline often more accurate |
| Non-UTF-8 encoding | Docling normalises internally; no special handling needed |
| VLM hallucinating text | Set `force_backend_text=True` for hybrid mode (PDF text + VLM layout) |
| VLM API call blocked | `enable_remote_services=True` is mandatory on `VlmPipelineOptions` |
| Apple Silicon | Use `GRANITEDOCLING_MLX` preset for MPS acceleration |

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

Install from the bundled requirements file (always pulls latest compatible):

```bash
pip install -r scripts/requirements.txt
```

Or manually:

```bash
pip install docling docling-core
# For OpenAI tokenizer support:
pip install 'docling-core[chunking-openai]'
```

Check installed versions (prefer distribution metadata — `docling` may not set `__version__`):

```python
from importlib.metadata import version
print(version("docling"), version("docling-core"))
```
