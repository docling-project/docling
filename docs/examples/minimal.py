# %% [markdown]
# What this example does
# - Converts a single source (URL or local file path) to a unified Docling
#   document and prints Markdown to stdout.
# - You can either directly use Docling or host an instance of Docling Serve
#
# Using Docling
# Requirements
# - Python 3.9+
# - Install Docling: `pip install docling`
#
# Using Docling Serve
# - Python 3.10+
# - Install Docling Serve: `pip install "docling-serve[ui]"`
# - Run the Docling Serve Server: `docling-serve run --enable-ui`
#
# How to run
# - Use the default sample URL: `python docs/examples/minimal.py`
# - To use your own file or URL, edit the `source` variable below.
#
# Notes
# - The converter auto-detects supported formats (PDF, DOCX, HTML, PPTX, images, etc.).
# - For batch processing or saving outputs to files, see `docs/examples/batch_convert.py`.
# %%

from docling.document_converter import DocumentConverter
from docling.service_client import DoclingServiceClient

# Change this to a local path or another URL if desired.
# Note: using the default URL requires network access; if offline, provide a
# local file path (e.g., Path("/path/to/file.pdf")).
source = "https://arxiv.org/pdf/2408.09869"

###########################################################################
# The two sections below are for either direct Docling usage or Docling Serve
# Uncomment exactly one section at a time

# Docling (local conversion)
# --------------------------------------
converter = DocumentConverter()
result = converter.convert(source)

# Docling Serve (remote conversion)
# --------------------------------------
# Replace SERVE_URL with your hosted URL if it's different
# SERVE_URL = "http://localhost:5001"
# with DoclingServiceClient(url=SERVE_URL) as client:
#     result = client.convert(source=source)

###########################################################################

print(result.document.export_to_markdown())
