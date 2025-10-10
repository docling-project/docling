# your_test_script.py

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

def main():
    # Use a raw string for Windows path to avoid unicode errors
    pdf_path = r"C:\Users\Admin\Documents\SRS-Sample.pdf"

    # Pages to exclude (1-based indexing)
    exclude_pages = [2]  # adjust as needed

    # Initialize DocumentConverter
    converter = DocumentConverter()

    # Convert the PDF while excluding specified pages
    result = converter.convert(
        pdf_path,
        page_range=(1, 1000),  # or leave default if not needed
        exclude_pages=exclude_pages
    )

    # Print some info
    print("Conversion status:", result.status)
    print("Number of pages processed:", len(result.pages))
    for page in result.pages:
        print(f"Page {page.page_no + 1}, size: {page.size}")

if __name__ == "__main__":
    main()
