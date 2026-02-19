Docling can parse various documents formats into a unified representation (Docling
Document), which it can export to different formats too — check out
[Architecture](../concepts/architecture.md) for more details.

Below you can find a listing of all supported input and output formats.

## Supported input formats

| Format | Description |
|--------|-------------|
| PDF | |
| DOCX, XLSX, PPTX | Default formats in MS Office 2007+, based on Office Open XML |
| Markdown | |
| AsciiDoc | Human-readable, plain-text markup language for structured technical content |
| LaTeX | Scientific document preparation system |
| HTML, XHTML | |
| CSV | |
| PNG, JPEG, TIFF, BMP, WEBP | Image formats |
| WebVTT | Web Video Text Tracks format for displaying timed text |

Schema-specific support:

| Format | Description |
|--------|-------------|
| USPTO XML | XML format followed by [USPTO](https://www.uspto.gov/patents) patents |
| JATS XML | XML format followed by [JATS](https://jats.nlm.nih.gov/) articles |
| Docling JSON | JSON-serialized [Docling Document](../concepts/docling_document.md) |

## Supported output formats

| Format | Description |
|--------|-------------|
| HTML | Both image embedding and referencing are supported |
| Markdown | |
| JSON | Lossless serialization of Docling Document |
| Text | Plain text, i.e. without Markdown markers |
| [Doctags](https://arxiv.org/pdf/2503.11576) | Markup format for efficiently representing the full content and layout characteristics of a document |
