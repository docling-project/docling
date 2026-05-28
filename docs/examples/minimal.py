# %% [markdown]
# What this example does
# - Converts a single source (URL or local file path) to a unified Docling
#   document with Docling Serve and prints Markdown to stdout.
#
# Requirements
# - Python 3.10+
# - Install Docling Serve: `pip install "docling-serve[ui]"`
# - Run the Docling Serve Server: `docling-serve run --enable-ui`
#
# Notes
# - The converter auto-detects supported formats (PDF, DOCX, HTML, PPTX, images, etc.).
# - For batch processing or saving outputs to files, see `docs/examples/batch_convert.py`.
#
# How to run
# - Use the default sample URL: `python docs/examples/minimal.py`
# - To use your own file or URL, edit the `source` variable below.
# %%

from docling.service_client import DoclingServiceClient

# Replace SERVE_URL with your hosted URL if it's different
SERVE_URL = "http://localhost:5001"

with DoclingServiceClient(url=SERVE_URL) as client:
    result = client.convert(source="https://arxiv.org/pdf/2408.09869")

print(result.document.export_to_markdown())
