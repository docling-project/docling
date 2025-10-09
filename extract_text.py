from docling.document_converter import DocumentConverter
from bs4 import BeautifulSoup
from pathlib import Path

# Replace with your PDF or DOCX file path
file_path = Path( r"C:\Users\Admin\Documents\24UAM153 - SOWMYA M.pdf")

# Initialize Docling converter
converter = DocumentConverter()

try:
    # Convert the document
    result = converter.convert(file_path)
    doc = result.document

    try:
        # Try to extract text directly (some Docling versions may allow this)
        text = doc.text
        print("Extracted text directly from document.\n")

    except AttributeError:
        # Fallback: extract from internal HTML if direct extraction fails
        print("Direct text extraction failed. Using internal HTML...")

        # Save internal HTML
        doc.save_as_html("output.html")

        # Read HTML and extract visible text
        with open("output.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Clean rich tables by removing all attributes
        for table in soup.find_all("table"):
            table.attrs = {}

        # Extract all visible text
        text = soup.get_text(separator="\n", strip=True)
        print("Extracted text from HTML after cleaning tables.\n")

    # Print the text
    print(text)

    # Save the text to a .txt file
    output_txt = file_path.stem + "_extracted.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nText saved to '{output_txt}' successfully.")

except Exception as e:
    print("An error occurred during conversion or extraction:", e)
