# EPUB Conversion Examples

This guide demonstrates how to convert EPUB files using Docling's EPUB backend.

## Basic Usage

```python
from docling.document_converter import DocumentConverter

# Create converter instance
converter = DocumentConverter()

# Convert an EPUB file
result = converter.convert("path/to/book.epub")

# Access the converted document
doc = result.document

# Export to markdown (without images)
markdown_text = doc.export_to_markdown()
print(markdown_text)

# Export to markdown with embedded images (base64)
markdown_with_images = doc.export_to_markdown(image_mode='embedded')
print(markdown_with_images)
```

## With Image Extraction

```python
from docling.document_converter import DocumentConverter, EpubFormatOption
from docling.datamodel.backend_options import EpubBackendOptions

# Configure EPUB options to extract images from the archive
epub_options = EpubBackendOptions(
    fetch_images=True,           # Extract images from EPUB
    enable_local_fetch=True,     # Allow reading local image files
    enable_remote_fetch=False,   # Disable fetching remote images
)

# Create converter with EPUB options
converter = DocumentConverter(
    format_options={
        'epub': EpubFormatOption(backend_options=epub_options)
    }
)

# Convert the EPUB
result = converter.convert("path/to/book.epub")

# Export with embedded images (base64-encoded)
markdown_with_images = result.document.export_to_markdown(image_mode='embedded')

# Or export with image references (saves images to separate files)
markdown_with_refs = result.document.export_to_markdown(
    image_mode='ref',
    image_export_dir='./images'
)
```

## Accessing Metadata

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("path/to/book.epub")

# Access document metadata
doc = result.document
print(f"Title: {doc.name}")

# The EPUB backend extracts metadata from the OPF file
# including title, author, language, and other Dublin Core metadata
```

## Batch Conversion

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Convert all EPUB files in a directory
epub_dir = Path("path/to/epub/directory")
for epub_file in epub_dir.glob("*.epub"):
    print(f"Converting {epub_file.name}...")
    result = converter.convert(str(epub_file))
    
    # Export to markdown with embedded images
    output_path = epub_file.with_suffix(".md")
    output_path.write_text(result.document.export_to_markdown(image_mode='embedded'))
    print(f"Saved to {output_path}")
```

## Features

The EPUB backend:

- **Parses EPUB structure**: Reads the container.xml and content.opf files to understand the book's organization
- **Extracts metadata**: Retrieves title, author, language, and other Dublin Core metadata from the OPF file
- **Preserves reading order**: Processes content files in the order specified by the spine element
- **Handles internal links**: Automatically fixes cross-file references (e.g., footnote links) when combining XHTML files
- **Extracts images**: Can extract and embed images from the EPUB archive when `fetch_images=True`
- **Leverages HTML backend**: Uses the existing HTML backend for robust XHTML content processing

## Image Handling

The EPUB backend can extract images from the EPUB archive and include them in the output:

```python
from docling.document_converter import DocumentConverter, EpubFormatOption
from docling.datamodel.backend_options import EpubBackendOptions

# Enable image extraction
epub_options = EpubBackendOptions(
    fetch_images=True,
    enable_local_fetch=True,
    enable_remote_fetch=False,
)

converter = DocumentConverter(
    format_options={
        'epub': EpubFormatOption(backend_options=epub_options)
    }
)

result = converter.convert("path/to/book.epub")

# Export with embedded images (base64-encoded in markdown)
markdown = result.document.export_to_markdown(image_mode='embedded')

# Or export with image references (saves images as separate files)
markdown = result.document.export_to_markdown(
    image_mode='ref',
    image_export_dir='./images'
)
```

**Image Export Modes:**

- `image_mode='placeholder'` (default): Replaces images with `<!-- image -->` comments
- `image_mode='embedded'`: Embeds images as base64 data URIs in the markdown
- `image_mode='ref'`: Saves images to separate files and references them in markdown

## Supported EPUB Versions

The backend supports EPUB 2 and EPUB 3 formats, which are the most common versions used for e-books.

## Technical Details

EPUB files are ZIP archives containing:
- XHTML content files
- Metadata (OPF file)
- Navigation structure
- Images and other resources

The backend:
1. Extracts the ZIP archive
2. Parses the container.xml to locate the OPF file
3. Reads the OPF file to get metadata and reading order
4. Combines all XHTML content files in spine order
5. Fixes internal cross-file links
6. Delegates to the HTML backend for final processing

## Known Limitations

### Internal Anchor Links

Internal anchor links (such as footnote references) are partially supported:

- **Links are converted**: References like `[1](#note-1)` will appear in the output
- **Anchor targets are not preserved**: The corresponding anchor IDs (e.g., `id="note-1"`) are lost during HTML-to-DoclingDocument conversion
- **Impact**: Clicking on footnote links in the exported Markdown won't jump to the footnote location

This is a limitation of the underlying HTML backend's conversion process, which focuses on extracting content structure rather than preserving HTML anchor IDs.

**Example:**
```markdown
<!-- In the text -->
...five versts [1](#note-1) from Durnovka...

<!-- At the end (footnote section) -->
1. A verst is two-thirds of a mile. [↩︎](#noteref-1)
```

The links `[1](#note-1)` and `[↩︎](#noteref-1)` will be present, but the anchor targets they reference won't be accessible in the Markdown output.