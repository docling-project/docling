# Docling CLI reference

The `docling` CLI is the quickest way to understand the content of any supported
document. It ships with the `docling` package (`pip install docling`) and needs
no code. Use it any time you need to read, convert, or extract content from a
file the agent cannot open directly — PDFs above all, but also DOCX, PPTX, XLSX,
HTML, Markdown, AsciiDoc, CSV, images, audio, and XML.

Authoritative flag list: <https://docling-project.github.io/docling/reference/cli/>
(or `docling --help`). This file is a task-oriented summary.

## Basics

```bash
docling <source> [--to md|json|html|text|doctags] [--output DIR]
```

- `<source>` is a **local path or an http(s) URL** — both work directly.
- Output files are named after the input (`report.pdf` → `report.md`).
- `--output` defaults to the current directory. Use `--output /tmp/` to redirect.
- `--to` may be repeated to emit several formats in one run.

```bash
docling report.pdf --to md --output /tmp/              # Markdown (readable)
docling report.pdf --to json --output /tmp/            # DoclingDocument JSON (lossless)
docling https://arxiv.org/pdf/2408.09869 --to md       # convert a URL
docling report.pdf --to md --to json --output /tmp/    # both at once
```

Supported input formats include: `pdf`, `docx`/`doc`, `pptx`/`ppt`, `xlsx`/`xls`,
`html`, `md`, `asciidoc`, `csv`, `odt`/`ods`/`odp`, images, `audio`, and several
XML flavors. Restrict/force detection with `--from` (repeatable), e.g.
`--from pdf --from docx ./inbox` to batch-convert a directory.

## Picking a pipeline (PDF / images)

Docling has two pipeline families for PDFs and images. Choose with `--pipeline`.

| Pipeline | Flag | Best for | Tradeoff |
|---|---|---|---|
| **Standard** (default) | `--pipeline standard` | Born-digital PDFs, speed | CPU-only OK; OCR handles scanned pages |
| **VLM** | `--pipeline vlm` | Complex layout, handwriting, formulas, figures with text | Needs GPU (or Apple MPS); slower |

```bash
docling report.pdf --pipeline vlm --output /tmp/
docling report.pdf --pipeline vlm --vlm-model granite_docling --output /tmp/
docling report.pdf --pipeline vlm --vlm-model smoldocling --output /tmp/
```

Decision guide:

| Document | Use |
|---|---|
| Born-digital PDF (text selectable) | Standard (fast, no GPU) |
| Scanned / image-only PDF | Standard with OCR, or `--pipeline vlm` for best quality |
| Complex/multi-column layout, dense tables | `--pipeline vlm` |
| Handwriting or formulas | `--pipeline vlm` (standard OCR won't handle these) |
| Air-gapped / no GPU | Standard |
| Speed-critical, accuracy secondary | Standard with `--no-ocr` and/or `--no-tables` |

## OCR (scanned PDFs and images)

OCR is on by default in the standard pipeline. Control it with:

```bash
docling scan.pdf --ocr-engine easyocr --output /tmp/     # default engine
docling scan.pdf --ocr-engine rapidocr --output /tmp/    # lightweight
docling scan.pdf --ocr-engine tesserocr --output /tmp/   # needs system Tesseract
docling scan.pdf --ocr-engine ocrmac --output /tmp/      # macOS Vision (mac only)
docling scan.pdf --force-ocr --output /tmp/              # re-OCR even extractable text
docling report.pdf --no-ocr --output /tmp/               # skip OCR (faster)
docling scan.pdf --ocr-lang en --ocr-lang de --output /tmp/   # restrict languages
```

OCR engines are optional dependencies — see
[slim-packaging.md](slim-packaging.md) for the `feat-ocr-*` extras.

## Tables, enrichment, and other content

```bash
docling report.pdf --no-tables --output /tmp/            # skip table structure (faster)
docling report.pdf --table-mode accurate --output /tmp/  # vs. fast
docling report.pdf --enrich-code --output /tmp/          # code understanding
docling report.pdf --enrich-formula --output /tmp/       # formula understanding
docling report.pdf --enrich-picture-classes --output /tmp/
docling report.pdf --enrich-picture-description --output /tmp/
```

## Common situations

| Situation | Handling |
|---|---|
| Scanned / image-only PDF | Standard with OCR, or `--pipeline vlm` |
| Password-protected PDF | `--pdf-password PASSWORD` (raises `ConversionError` if wrong) |
| Very large document (500+ pages) | Standard, add `--no-tables` for speed; set `--device` / `--num-threads` |
| Complex / multi-column layout | `--pipeline vlm` (standard may misorder reading flow) |
| Handwriting or formulas | `--pipeline vlm` only |
| Output near-empty | Enable OCR, or switch to `--pipeline vlm` |
| `�` replacement characters | Try a different `--ocr-engine`, or `--pipeline vlm` |
| Same line repeated many times | `--pipeline vlm` (or hybrid `force_backend_text`, Python SDK only) |
| Choose GPU/CPU explicitly | `--device cuda` / `--device cpu` / `--device mps` |

## Remote VLM services from the CLI

The CLI can route pages to a remote VLM with
`--pipeline vlm --enable-remote-services`, but the endpoint URL, model name, and
API key must be configured through the **Python SDK** (`ApiVlmOptions`) — see
[python-sdk.md](python-sdk.md). Docling blocks outbound HTTP unless
`--enable-remote-services` is set.

> Do not confuse this with offloading the **whole** conversion to a
> `docling-serve` endpoint — that is `docling convert-remote` and the Service
> Client. See [service-client.md](service-client.md).

## Offline models

Pre-download models and point the CLI at them for air-gapped runs:

```bash
docling-tools models download --output-dir /models   # fetch model artifacts
docling report.pdf --artifacts-path /models --output /tmp/
```

`--artifacts-path` overrides the default Hugging Face cache; you can also set
`HF_HOME` to relocate that cache.

## Quick verification checklist

After a conversion where fidelity matters, sanity-check the output:

- Page count roughly matches the source (re-run with `--pipeline vlm` if not).
- Markdown is not near-empty (enable OCR / VLM if it is).
- Expected tables are present (drop `--no-tables`; try `--pipeline vlm`).
- No blocks of `�` or endlessly repeated lines (switch OCR engine or pipeline).
