# docling-rs

A high-performance Rust rewrite of [Docling](https://github.com/docling-project/docling), the document conversion library. Converts DOCX, XLSX, PPTX, PDF, and many more formats into structured Markdown and JSON, with no Python runtime or ML models required.

## Highlights

- **Single 8.7 MB binary** with zero external dependencies
- **100% compatible** with Python Docling output for DOCX, XLSX, and PPTX
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
| PDF | `lopdf` | Text extraction with heuristics (no ML models) |
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
| **PDF** | Heuristic-based | ~20% of Python count | Ruled-line only | Extracted | List groups |

### PDF Limitations

The Rust PDF backend uses `lopdf` for direct text extraction with heuristic-based layout analysis. Python Docling uses ML models (RT-DETR, TableFormer, OCR) which provide significantly richer output. The Rust version:
- Extracts embedded text (no OCR for scanned documents)
- Detects tables from ruled lines only (no borderless tables)
- Uses font-size heuristics for heading detection
- Supports basic two-column layout reordering
- No RTL (right-to-left) text reordering

## Project Structure

```
docling-rs/          # Main Rust crate
  src/
    backend/         # Format-specific parsers (docx, xlsx, pptx, pdf, ...)
    models/          # DoclingDocument data model
    export/          # Output formatters (markdown, json, html, ...)
    cli.rs           # CLI entry point
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
