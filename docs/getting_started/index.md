Ready to kick off your Docling journey? Let's dive right into it!

I was trying to get started with Docling and found that the getting started guide was really basic - just some navigation cards but no actual tutorial. So I wrote this step-by-step guide to help other users actually get up and running instead of getting stuck.

## Quick Start (5 minutes)

### Step 1: Install Docling
```bash
# Using pip (recommended for beginners)
pip install docling

# Using uv (if you prefer modern package management)
uv add docling
```

### Step 2: Your First Conversion
```python
from docling.document_converter import DocumentConverter

# Convert a PDF to Markdown
converter = DocumentConverter()
result = converter.convert("path/to/your/document.pdf")
markdown_text = result.document.export_to_markdown()
print(markdown_text)
```

### Step 3: Try the CLI
```bash
# Convert a document from the command line
docling path/to/your/document.pdf

# Convert to HTML instead
docling path/to/your/document.pdf --output-format html
```

## Detailed Installation Guide

<div class="grid">
  <a href="../installation/" class="card"><b>Installation</b><br />Quickly install Docling in your environment</a>
  <a href="../usage/" class="card"><b>Usage</b><br />Get a jumpstart on basic Docling usage</a>
  <a href="../concepts/" class="card"><b>Concepts</b><br />Learn Docling fundamentals and get a glimpse under the hood</a>
  <a href="../examples/" class="card"><b>Examples</b><br />Try out recipes for various use cases, including conversion, RAG, and more</a>
  <a href="../integrations/" class="card"><b>Integrations</b><br />Check out integrations with popular AI tools and frameworks</a>
  <a href="../reference/document_converter/" class="card"><b>Reference</b><br />See more API details</a>
</div>

## Common Use Cases

### **Document Conversion**
Convert documents between formats:
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# PDF to Markdown
result = converter.convert("document.pdf")
markdown = result.document.export_to_markdown()

# PDF to HTML
html = result.document.export_to_html()

# PDF to JSON (for programmatic access)
json_data = result.document.export_to_dict()
```

### **Batch Processing**
Handle multiple documents:
```python
import glob
from pathlib import Path

converter = DocumentConverter()
pdf_files = glob.glob("documents/*.pdf")

for pdf_file in pdf_files:
    result = converter.convert(pdf_file)
    output_path = Path(pdf_file).with_suffix('.md')
    with open(output_path, 'w') as f:
        f.write(result.document.export_to_markdown())
```

### **RAG Applications**
Build search and retrieval systems:
```python
from docling.document_converter import DocumentConverter
from docling.chunking import chunk_document

# Convert and chunk documents
converter = DocumentConverter()
result = converter.convert("document.pdf")
chunks = chunk_document(result.document, chunk_size=1000)

# Use chunks with your favorite vector database
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.text[:100]}...")
```

## Common Pitfalls & Solutions

I ran into these issues when I was getting started, so here's how to avoid them:

### **Installation Issues**
- **PyTorch conflicts**: Use `pip install "docling[mac_intel]"` on Intel Macs
- **OCR dependencies**: Install system packages for Tesseract if using OCR features
- **Memory issues**: Use smaller models or enable chunking for large documents

### **Performance Tips**
- **First run**: Models download automatically (can take time)
- **Large documents**: Use chunking to process in smaller pieces
- **GPU acceleration**: Install CUDA-enabled PyTorch for faster processing

### **Common Errors**
- **File not found**: Check file paths and permissions
- **Unsupported format**: Verify the file format is supported
- **Memory errors**: Reduce batch size or enable chunking

## Configuration Options

### **Basic Configuration**
```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions

# Enable OCR for scanned documents
pipeline_options = PipelineOptions()
pipeline_options.do_ocr = True

converter = DocumentConverter(pipeline_options=pipeline_options)
```

### **Advanced Configuration**
```python
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, PdfFormatOption

# Customize PDF processing
pdf_options = PdfPipelineOptions()
pdf_options.extract_tables = True
pdf_options.extract_figures = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
    }
)
```

## Next Steps

I've organized this so you can learn progressively:

### **Beginner Path**
1. **Try the examples** → [Examples](../examples/) - Hands-on learning
2. **Learn concepts** → [Concepts](../concepts/) - Understand the system
3. **Explore usage** → [Usage Guide](../usage/) - Advanced features

### **Intermediate Path**
1. **Custom pipelines** → [Advanced Options](../usage/advanced_options.md)
2. **Integration** → [Framework Integrations](../integrations/)
3. **Customization** → [Plugin System](../concepts/plugins.md)

### **Advanced Path**
1. **API Reference** → [Reference](../reference/) - Complete API docs
2. **Contributing** → [GitHub](https://github.com/docling-project/docling) - Join development
3. **Technical Report** → [arXiv Paper](https://arxiv.org/abs/2408.09869) - Deep dive

## Getting Help

If you get stuck (I did several times):

### **Documentation**
- **This guide** - Start here for basics
- **Examples** - See code in action
- **API Reference** - Detailed technical information

### **Community**
- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share ideas
- **Contributing** - Help improve Docling

### **Resources**
- **Technical Report** - Academic paper with implementation details
- **Blog Posts** - Community tutorials and case studies
- **Video Tutorials** - Visual learning resources

## Congratulations!

You've completed the getting started guide! You now know how to:
- Install Docling
- Convert your first document
- Use the command-line interface
- Handle common issues
- Configure basic options

**What's next?** Choose your path:
- **Build something** - Try the examples
- **Learn more** - Explore concepts and usage
- **Get involved** - Join the community

---

The journey has just begun! Join us and become a part of the growing Docling community!

- <a href="https://github.com/docling-project/docling">:fontawesome-brands-github: GitHub</a>
- <a href="https://linkedin.com/company/docling/">:fontawesome-brands-linkedin: LinkedIn</a>
