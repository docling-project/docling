# %% [markdown]
# Simple conversion: one document to Markdown
# ==========================================
#
# What this example does
# - Converts a single source (URL or local file path) to a unified Docling
#   document and prints Markdown to stdout.
#
# Requirements
# - Python 3.9+
# - Install Docling: `pip install docling`
#
# How to run
# - Use the default sample URL: `python docs/examples/minimal.py`
# - To use your own file or URL, edit the `source` variable below.
#
# Notes
# - The converter auto-detects supported formats (PDF, DOCX, HTML, PPTX, images, etc.).
# - For batch processing or saving outputs to files, see `docs/examples/batch_convert.py`.

from docling.document_converter import DocumentConverter

# Change this to a local path or another URL if desired.
# Note: using the default URL requires network access; if offline, provide a
# local file path (e.g., Path("/path/to/file.pdf")).
source = "https://arxiv.org/pdf/2408.09869"

converter = DocumentConverter()
result = converter.convert(source)

# Print Markdown to stdout.
print(result.document.export_to_markdown())
