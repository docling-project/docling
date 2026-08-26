# Docling Slim

**Lightweight SDK for parsing documents with minimal dependencies and opt-in extras**

Docling Slim is a minimal-dependency version of Docling that allows you to install only the components you need. It provides the core document processing functionality with ~50MB of base dependencies, and you can add specific features through optional extras.

## When to Use Docling Slim

- **Use `docling`** (recommended): If you want the full-featured experience with all standard capabilities
- **Use `docling-slim`**: If you need fine-grained control over dependencies or want to minimize installation size

## For Most Users: Use the Main Docling Package

We recommend most users install the full-featured `docling` package instead:

```bash
pip install docling
```

The `docling` package includes all standard features, the CLI tools, and is the easiest way to get started. Visit the [main Docling documentation](https://docling-project.github.io/docling/) for complete guides and examples.

## Installation

### With Specific Features
```bash
# PDF support with local models
pip install docling-slim[format-pdf,models-local]

# Office formats only
pip install docling-slim[format-office]

# PDF + CLI
pip install docling-slim[format-pdf,cli]

# Docling service client for using the Docling Serve API
pip install docling-slim[service-client]
```

## Available Extras

### Convenience Bundles

| Extra | Description | Use Case |
|-------|-------------|----------|
| `standard` | All standard features (same as `docling` package) | Full-featured usage |
| `all` | All available extras | Complete installation |

### CLI

| Extra | Description | Use Case |
|-------|-------------|----------|
| `cli` | Command-line interface (typer, rich) | CLI tools (docling, docling-tools) |

### Core Components

| Extra | Description | Use Case |
|-------|-------------|----------|
| `convert-core` | Core conversion components (numpy, pillow, scipy) | Basic document conversion |
| `extract-core` | Structured information extraction | Data extraction from documents |

### Format Support

#### PDF Formats

| Extra | Description | Use Case |
|-------|-------------|----------|
| `format-pdf` | PDF parsing (pypdfium2 + docling-parse) | PDF documents |
| `format-pdf-pypdfium2` | PDF rendering only | Lightweight PDF support |
| `format-pdf-docling` | Advanced PDF parsing | Complex PDF layouts |

#### Office Formats (office = docx + pptx + xlsx)

| Extra | Description | Use Case |
|-------|-------------|----------|
| `format-office` | All Office formats | Microsoft Office documents |
| `format-docx` | Microsoft Word documents | .docx files |
| `format-pptx` | Microsoft PowerPoint | .pptx files |
| `format-xlsx` | Microsoft Excel | .xlsx files |

#### Web Formats (web = html + markdown)

| Extra | Description | Use Case |
|-------|-------------|----------|
| `format-web` | HTML and Markdown | Web content |
| `format-html` | HTML parsing | Web pages and HTML files |
| `format-markdown` | Markdown parsing | .md files |

#### Other Formats

| Extra | Description | Use Case |
|-------|-------------|----------|
| `format-latex` | LaTeX documents | .tex files |
| `format-xml-xbrl` | XBRL financial reports | Financial documents |
| `format-html-render` | HTML rendering with Playwright | Dynamic web content |
| `format-audio` | Audio transcription (Whisper) | .wav, .mp3 files |

### OCR Engines

| Extra | Description | Use Case |
|-------|-------------|----------|
| `feat-ocr-rapidocr` | RapidOCR (lightweight) | Fast OCR |
| `feat-ocr-rapidocr-onnx` | RapidOCR with ONNX runtime | Optimized OCR |
| `feat-ocr-easyocr` | EasyOCR | Multi-language OCR |
| `feat-ocr-tesserocr` | Tesseract OCR | High-accuracy OCR |
| `feat-ocr-mac` | macOS native OCR | macOS only |

### Models

| Extra | Description | Use Case |
|-------|-------------|----------|
| `models-local` | Local PyTorch models | GPU/CPU inference |
| `models-remote` | Remote model serving (Triton) | Production deployments |
| `models-onnxruntime` | ONNX Runtime acceleration | Optimized inference |
| `models-vlm-inline` | Vision Language Models | Image understanding, inline processing |

### Other features

| Extra | Description | Use Case |
|-------|-------------|----------|
| `feat-chunking` | Document chunking | RAG applications |
| `service-client` | Docling service client | Remote processing |


## License

MIT License - See [LICENSE](https://github.com/docling-project/docling/blob/main/LICENSE)
