# docling-slim packaging & install extras

`docling-slim` is the same codebase as `docling` but with **no default
dependencies** — you opt into exactly the formats, OCR engines, models, and
features you need via install extras. This keeps environments small and avoids
pulling heavy packages (torch, OCR engines, VLM stacks) you won't use.

- `pip install docling` → the batteries-included meta-package (equivalent to
  `docling-slim[standard]`).
- `pip install docling-slim[...]` → pick your own footprint.

## Choosing extras — recipes

| Goal | Install |
|---|---|
| PDF → Markdown/JSON, no ML, plus CLI | `docling-slim[format-pdf,cli]` |
| Only talk to a remote service (no local models) | `docling-slim[service-client]` |
| Office files (DOCX/PPTX/XLSX) | `docling-slim[format-office]` |
| Web (HTML + Markdown) | `docling-slim[format-web]` |
| Scanned PDFs with a lightweight OCR engine | `docling-slim[format-pdf,feat-ocr-rapidocr]` |
| Local layout/table/OCR ML models | `docling-slim[models-local]` |
| Local VLM pipeline | `docling-slim[models-vlm-inline]` |
| Chunking for RAG | `docling-slim[feat-chunking]` |
| Everything the default package has | `docling-slim[standard]` |
| Kitchen sink (all optional features) | `docling-slim[all]` |

Extras compose — combine them with commas:
`pip install "docling-slim[format-pdf,models-local,feat-ocr-rapidocr,feat-chunking,cli]"`.

## Extras catalog

**Core**
- `convert-core` — numpy/pillow/rtree/scipy (base of most format extras)
- `extract-core` — structured extraction support

**Formats**
- `format-pdf` (`format-pdf-pypdfium2`, `format-pdf-docling`)
- `format-office` (`format-docx`, `format-pptx`, `format-xlsx`)
- `format-web` (`format-html`, `format-markdown`)
- `format-opendocument` (ODT/ODS/ODP)
- `format-latex`, `format-email`
- `format-xml-jats`, `format-xml-uspto`, `format-xml-xbrl`
- `format-html-render` (Playwright for JS-rendered HTML)
- `format-audio` (ASR), `format-video` (ASR + diarization; needs a C compiler)

**OCR engines** — install only the one(s) you use
- `feat-ocr-rapidocr` (lightweight), `feat-ocr-rapidocr-onnx`
- `feat-ocr-easyocr`
- `feat-ocr-tesserocr` (needs system Tesseract)
- `feat-ocr-mac` (macOS Vision)
- `feat-ocr-nemotron`

**Models**
- `models-local` — torch + docling-ibm-models (local layout/table/OCR models)
- `models-vlm-inline` — local vision-language models
- `models-remote` — Triton client for remote model serving
- `models-onnxruntime` — ONNX runtime backends

**Features & tooling**
- `feat-chunking` — HybridChunker / RAG chunking
- `service-client` — remote `docling-serve` client + `docling convert-remote`
- `cli` — the `docling` / `docling-tools` command-line entry points

**Bundles**
- `standard` — the default `docling` set: `format-pdf`, `models-local`,
  `feat-ocr-rapidocr`, `format-office`, `format-web`, `format-latex`,
  `format-email`, `feat-chunking`, `extract-core`, `service-client`, `cli`.
- `all` — `standard` plus VLM, audio, HTML render, all XML, remote/onnx models,
  and the extra OCR engines. (Excludes `format-video`, which needs a compiler.)

> The definitive extras list lives in the project's `pyproject.toml` under
> `[project.optional-dependencies]`. If an extra name here doesn't resolve,
> check there for the current spelling.

## Notes

- The `docling` CLI requires the `cli` extra (bundled in `standard`/`docling`).
- `models-local` is the heavy one (pulls torch); omit it if you only convert
  non-PDF formats or use a remote service.
- To convert with zero local models, use `service-client` and point at a
  `docling-serve` endpoint — see [service-client.md](service-client.md).
