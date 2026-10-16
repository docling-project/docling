---
name: docling
description: >
  Use Docling to understand the content of documents in any supported format —
  PDF (born-digital or scanned), DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc,
  CSV, images, audio, and XML — by converting them into a unified
  DoclingDocument (Markdown or structured JSON). Use this skill whenever you
  need to read, parse, convert, extract, or chunk a document you cannot read
  directly: "what's in this PDF", "convert this to markdown", "extract the
  tables", "chunk this for RAG", "read this scanned document", "parse this
  DOCX/PPTX". Covers the `docling` CLI, the Python SDK (DocumentConverter +
  PipelineOptions), the remote Service Client (self-hosted or managed
  docling-serve), and the docling-slim install extras for a minimal dependency
  footprint.
license: MIT
compatibility: Requires Python 3.10+
metadata:
  author: docling-project
  version: "1.0"
  upstream: https://github.com/docling-project/docling
allowed-tools: Bash(docling:*) Bash(docling-tools:*) Bash(python3:*) Bash(python:*) Bash(uvx:*) Bash(uv:*) Bash(pip:*)
---

# Docling

Docling converts documents — PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc,
CSV, images, audio, and XML — into a single unified representation, the
**`DoclingDocument`**, which you can export as **Markdown** (human-readable) or
**JSON** (preserves the extracted document structure). JSON preserves the
conversion result; it does not guarantee fidelity to the source document.
Reach for Docling whenever you need to understand the content of a file you
cannot read directly, especially PDFs
(including scanned ones, via OCR or a vision-language model).

## The fastest thing that works: the CLI

If you just need to read a document's content, run the CLI. It is installed with
the `docling` package and accepts a local path **or** a URL:

```bash
docling report.pdf --to md --output /tmp/        # → /tmp/report.md
docling https://example.com/paper.pdf --to json --output /tmp/
```

Output files are named after the input (`report.pdf` → `report.md`). Default
output directory is the current directory. This handles the majority of
"what's in this file" requests. See **[references/cli.md](references/cli.md)**
for pipelines (standard vs VLM vs native), OCR engines, tables, scanned PDFs,
passwords, and every flag.

## Choosing how to use Docling

| You need to… | Use | Reference |
|---|---|---|
| Read / convert a file once, from the shell | **CLI** (`docling …`) | [references/cli.md](references/cli.md) |
| Convert programmatically, tune the pipeline, batch, ASR, export images/tables | **Python SDK** (`DocumentConverter` + `PipelineOptions`) | [references/python-sdk.md](references/python-sdk.md) |
| Pull specific typed fields out of a document (not the whole doc) | **DocumentExtractor** (structured extraction, beta) | [references/extraction.md](references/extraction.md) |
| Chunk documents for retrieval / feed a RAG index | **Chunking + framework loaders** | [references/rag.md](references/rag.md) |
| Offload conversion to a remote service — low latency, scalable, no local ML deps or GPU | **Service Client** (self-hosted or managed docling-serve) | [references/service-client.md](references/service-client.md) |
| Install only the dependencies you actually use | **docling-slim** extras | [references/slim-packaging.md](references/slim-packaging.md) |

Rules of thumb:

- **Local, one-off, no code** → CLI.
- **Custom pipeline, chunking, structure analysis, embedding in an app** → Python SDK.
- **Many documents, low-latency, no GPU/ML install to manage, scale on demand**
  → Service Client against a `docling-serve` endpoint (self-hosted or the
  managed **Docling for IBM watsonx** service).
- **Minimize install size / avoid pulling torch and OCR engines you don't need**
  → docling-slim with targeted extras.

## Running without installing (uvx)

You can run the CLI without a persistent install:

```bash
uvx --from docling docling report.pdf --to md --output /tmp/
```

## Source fidelity

For ordinary reading, use the lightweight CLI path. When the requested result
depends on exact numbers, formulas, table relationships, or source quotations,
or conversion shows suspicious output, read
[references/quality-verification.md](references/quality-verification.md).

- Conversion success and page coverage are separate from content accuracy.
  Report the extent of verification, not just the conversion status.
- Preserve raw output and available provenance before correcting a value or
  changing reading order. Check the affected source region; do not infer an
  exponent, sign, or unit from plausibility.
- Prefer a targeted retry with a concrete hypothesis over rerunning the entire
  document. Keep unresolved findings qualified.

## Output conventions

- Always report the conversion status and (for PDFs) the processed page range/count, separately from content verification.
- If the user does not specify a format, ask whether they want **Markdown** (readable) or **JSON / DoclingDocument** (structured).
- For tables, prefer `export_to_markdown()` / `export_to_dataframe()` on the table item (Python) — see [references/python-sdk.md](references/python-sdk.md).
- If a converted PDF comes back near-empty, repeated, or full of `�`, compare the source image, native text layer, raw structured output, and export before choosing an OCR or VLM retry. See [references/quality-verification.md](references/quality-verification.md) and [references/cli.md](references/cli.md); a retry is not evidence that the content is correct.
