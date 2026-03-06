# docling-rs

A high-performance Rust rewrite of [Docling](https://github.com/docling-project/docling), the document conversion library. Converts DOCX, XLSX, PPTX, PDF, and many more formats into structured Markdown and JSON, with no Python runtime or ML models required.

## Highlights

- **Single binary** with zero external dependencies
- **100% compatible** with Python Docling output for DOCX, XLSX, and PPTX
- **PDF extraction powered by `pdf_oxide`** for spatial text assembly with reading order, and **`pdfium-render`** for vector diagram rendering
- **160+ end-to-end tests** validated against Python Docling groundtruth
- Supports 15+ input formats and 7 output formats
- Image extraction with `--image-export-mode referenced`

## Supported Formats

### Input

| Format | Backend | Python Compatibility |
|--------|---------|---------------------|
| DOCX | `quick_xml` | 100% (tables, lists, images, equations, formatting) |
| XLSX / XLSM | `quick_xml` | 100% (tables, merged cells, images, multi-sheet) |
| PPTX | `quick_xml` | 100% (slides, tables, images, grouped shapes) |
| PDF | `pdf_oxide` + `pdfium-render` | Spatial text assembly, vector diagram extraction, raster image extraction |
| CSV | Built-in parser | Full |
| HTML | `scraper` | Full |
| Markdown | `pulldown-cmark` | Full |
| AsciiDoc | Custom parser | Full |
| LaTeX | Custom parser | Full |
| XML (JATS) | `quick_xml` | Full |
| XML (USPTO) | `quick_xml` | Full |
| XML (XBRL) | `quick_xml` | Full |
| WebVTT | Custom parser | Full |
| JSON (DoclingDocument) | `serde_json` | Full |
| Image (PNG, JPEG, TIFF) | `image` | Metadata only (no OCR) |

### Output

Markdown, JSON (DoclingDocument), YAML, HTML, plain text, DocTags, WebVTT

## Installation

### From source

```bash
git clone https://github.com/zynga/docling-rs.git
cd docling-rs/docling-rs
cargo build --release
# Binary at: target/release/docling-rs
```

## Usage

### CLI

```bash
# Convert a DOCX to Markdown
docling-rs convert document.docx

# Convert with image extraction
docling-rs convert --to md --image-export-mode referenced document.docx

# Convert to JSON
docling-rs convert --to json document.xlsx

# Multiple output formats
docling-rs convert --to md --to json document.pptx

# Specify output directory
docling-rs convert -o ./output document.pdf

# Process multiple files
docling-rs convert *.docx *.pdf
```

### Options

```
docling-rs convert [OPTIONS] <SOURCE>...

Arguments:
  <SOURCE>...    Input files or directories

Options:
  -f, --from <FROM>                Input format (auto-detected if omitted)
      --to <TO>                    Output format(s) [default: md]
  -o, --output <OUTPUT>            Output directory [default: .]
  -v, --verbose                    Verbosity level (-v info, -vv debug)
      --image-export-mode <MODE>   placeholder | embedded | referenced
      --abort-on-error             Stop on first error
      --document-timeout <SECS>    Per-document timeout
      --num-threads <N>            Thread count [default: 4]
```

## Python Compatibility

Verified against Python Docling with comprehensive end-to-end tests:

| Format | JSON Match | Texts | Tables | Images | Groups |
|--------|-----------|-------|--------|--------|--------|
| **DOCX** | 100% structural | Identical | Identical | Identical | Identical |
| **XLSX** | 100% structural | Identical | Identical | Identical | Identical |
| **PPTX** | 100% structural | Identical | Identical | Identical | Identical |
| **PDF** | High quality | Spatial text assembly | Ruled-line detection | Raster + vector diagrams | List groups |

### PDF Pipeline

The Rust PDF backend uses a dual-library approach:

- **`pdf_oxide`** for text extraction: spatial text assembly with XY-Cut reading order, font metadata, and PDF artifact detection. Produces properly ordered, non-fragmented text blocks.
- **`pdfium-render`** for image extraction: renders page regions to capture vector-drawn diagrams (architecture diagrams, flowcharts) that are not embedded as raster images. The PDFium binary is auto-downloaded and cached via `pdfium-auto`.
- **`lopdf`** for table detection from ruled lines.
- Heuristic classification for headings, lists, captions, and footnotes using font-size ratios.
- Edge density analysis and aspect ratio filtering to reject gradient backgrounds and decorative strips from rendered regions.

### PDF Limitations (vs Python)

Python Docling uses ML models (RT-DETR, TableFormer, OCR) which provide:
- Visual layout classification (the ML model identifies picture regions on the page)
- Borderless table detection
- OCR for scanned documents
- RTL text reordering

The Rust version does not use ML models, relying on heuristics and PDF structure instead.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `pdf-oxide` | Yes | Use `pdf_oxide` for spatial PDF text extraction |
| `pdfium-render` | Yes | Use `pdfium-render` + `pdfium-auto` for vector diagram rendering |

To build without pdfium (smaller binary, no diagram rendering):

```bash
cargo build --release --no-default-features --features pdf-oxide
```

## Project Structure

```
docling-rs/          # Main Rust crate
  src/
    backend/         # Format-specific parsers (docx, xlsx, pptx, pdf, ...)
    models/          # DoclingDocument data model
    export/          # Output formatters (markdown, json, html, ...)
    main.rs          # CLI entry point
e2e/                 # End-to-end test suite (160+ tests)
tests/data/          # Test fixtures and groundtruth
```

## Testing

```bash
# Run all 160+ E2E tests
cd e2e && cargo test

# Run format-specific tests
cargo test test_docx
cargo test test_xlsx
cargo test test_pptx
cargo test test_pdf

# Run unit tests
cd docling-rs && cargo test
```

## License

MIT
