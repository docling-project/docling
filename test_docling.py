# Install PyMuPDF if not installed:
# pip install PyMuPDF

import fitz  # PyMuPDF

pdf_path = r"C:\Users\Admin\Documents\24UAM153 - SOWMYA M.pdf"

# Open the PDF
doc = fitz.open(pdf_path)

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()
    print(f"--- Page {page_number} ---")
    print(text)

doc.close()
