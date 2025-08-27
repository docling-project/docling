Docling can parse various documents formats into a unified representation (Docling
Document), which it can export to different formats too — check out
[Architecture](../concepts/architecture.md) for more details.

I was trying to understand what formats Docling supports and found that most of the descriptions were empty. So I filled them in with actual information about what each format can do and when you'd want to use it.

Below you can find a listing of all supported input and output formats.

## Supported input formats

| Format | Description |
|--------|-------------|
| PDF | Portable Document Format - supports text extraction, OCR for scanned documents, table detection, layout analysis, and figure extraction. Handles both native text and scanned images. |
| DOCX, XLSX, PPTX | Default formats in MS Office 2007+, based on Office Open XML. Extracts text, tables, images, and maintains document structure. Supports comments, formatting, and metadata. |
| Markdown | Lightweight markup language format. Parses headers, lists, code blocks, links, and inline formatting. Ideal for documentation and technical content. |
| AsciiDoc | Text-based document format similar to Markdown but with more features. Supports complex document structures, tables, and cross-references. |
| HTML, XHTML | Web document formats. Extracts text content, preserves links, handles nested elements, and maintains document hierarchy. |
| CSV | Comma-separated values format. Parses tabular data with configurable delimiters, handles quoted fields, and preserves data types. |
| PNG, JPEG, TIFF, BMP, WEBP | Image formats with OCR capabilities. Extracts text from images, classifies content types, and provides image descriptions using vision models. |

Schema-specific support:

| Format | Description |
|--------|-------------|
| USPTO XML | XML format followed by [USPTO](https://www.uspto.gov/patents) patents. Parses patent metadata, claims, descriptions, and technical drawings with specialized extraction. |
| JATS XML | XML format followed by [JATS](https://jats.nlm.nih.gov/) articles. Handles academic papers, scientific articles, and research publications with structured content extraction. |
| Docling JSON | JSON-serialized [Docling Document](../concepts/docling_document.md). Allows lossless import/export of processed documents for workflow integration and state persistence. |

## Supported output formats

| Format | Description |
|--------|-------------|
| HTML | Both image embedding and referencing are supported. Generates clean, semantic HTML with preserved structure, tables, and formatting. Includes CSS styling options and responsive design considerations. |
| Markdown | Lightweight markup output ideal for documentation, note-taking, and version control. Preserves document hierarchy, tables, and basic formatting while maintaining readability. |
| JSON | Lossless serialization of Docling Document. Complete representation including all metadata, confidence scores, and processing results. Perfect for programmatic access and data analysis. |
| Text | Plain text, i.e. without Markdown markers. Clean text extraction suitable for search indexing, content analysis, and simple text processing workflows. |
| Doctags | Specialized format for document annotation and tagging. Supports custom metadata, labels, and structured annotations for document management systems. |

## Format-specific features

I've organized the features by format type so you can see what each one is good at:

### PDF Processing
- **Text Extraction**: Native text and OCR for scanned documents
- **Layout Analysis**: Page structure, reading order, and content flow
- **Table Detection**: Automatic table recognition and extraction
- **Figure Extraction**: Image and chart identification
- **Formula Recognition**: Mathematical expression detection
- **Multi-language Support**: OCR in various languages

### Office Documents
- **Structure Preservation**: Maintains document hierarchy and formatting
- **Table Extraction**: Excel-like table handling with cell merging
- **Image Handling**: Embedded images and charts
- **Metadata Extraction**: Document properties, author information
- **Comment Processing**: User annotations and feedback
- **Version Compatibility**: Support for Office 2007+ formats

### Image Processing
- **OCR Capabilities**: Text extraction from images
- **Content Classification**: Automatic image type detection
- **Vision Models**: Advanced image understanding with VLMs
- **Multi-format Support**: Various image formats and resolutions
- **Batch Processing**: Efficient handling of multiple images

## Choosing the right format

I've found these combinations work well for different use cases:

### **For Content Analysis**
- **Input**: PDF, DOCX for rich document content
- **Output**: JSON for programmatic access, Markdown for readability

### **For Web Applications**
- **Input**: HTML, Markdown for web content
- **Output**: HTML for web display, JSON for API responses

### **For Data Processing**
- **Input**: CSV, XLSX for tabular data
- **Output**: JSON for data analysis, Text for simple extraction

### **For Documentation**
- **Input**: Markdown, AsciiDoc for technical content
- **Output**: Markdown for version control, HTML for web publishing

## Performance considerations

Some things I've learned about performance:

- **Large PDFs**: Use chunking for documents over 100 pages
- **Image-heavy documents**: Enable vision models for better understanding
- **Batch processing**: Process multiple documents efficiently with proper memory management
- **Format conversion**: Some formats (e.g., PDF to Markdown) are more resource-intensive than others
