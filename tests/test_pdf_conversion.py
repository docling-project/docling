# test_pdf_conversion.py
from pathlib import Path
from docling.datamodel.document import InputDocument
from patched_backend import PyPdfiumDocumentBackend

pdf_path = Path(r"C:\Users\Admin\Desktop\demo\docling\docling\cli\PDF.pdf")

# Initialize InputDocument with patched backend
input_doc = InputDocument(
    path_or_stream=pdf_path,
    format="pdf",
    backend=PyPdfiumDocumentBackend
)

# Extract text page by page
for page_no in range(input_doc.page_count):
    page = input_doc._backend.load_page(page_no)
    text_cells = page._compute_text_cells()
    page_text = " ".join([cell.text for cell in text_cells])
    print(f"\n--- Page {page_no + 1} ---\n{page_text}")

# Unload PDF from memory
input_doc._backend.unload()
