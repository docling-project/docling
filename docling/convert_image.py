from pathlib import Path
import base64
import requests
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# -------------------------------
# Helper function to fix OCR degree misreads
# -------------------------------
def fix_degree_chars(text: str) -> str:
    misreads = ["P", "9", "'", "`"]
    for c in misreads:
        text = text.replace(c, "°")
    return text

# -------------------------------
# MathPix formula extraction
# -------------------------------
def extract_formula_latex(image_path: Path, app_id: str, app_key: str) -> str:
    """
    Send an image to MathPix API and get LaTeX.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode()

    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json"
    }

    payload = {
        "src": f"data:image/jpg;base64,{image_base64}",
        "formats": ["latex_styled"],
    }

    response = requests.post("https://api.mathpix.com/v3/text", json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("latex_styled", "")
    else:
        print("MathPix API error:", response.text)
        return ""

# -------------------------------
# Configure pipeline options
# -------------------------------
ocr_options = PdfPipelineOptions()
ocr_options.ocr_options.lang = ["no"]  # Norwegian OCR, change if needed

converter = DocumentConverter(
    format_options={
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=ocr_options),
        InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_options),
    }
)

# -------------------------------
# Input file path
# -------------------------------
path_file = Path(r"C:\Users\Admin\Desktop\demo\docling\docling\cli\degreetest123.docx")  # Your image

# -------------------------------
# Convert document
# -------------------------------
result = converter.convert(path_file)

# Export Markdown and fix OCR
raw_md = result.document.export_to_markdown()
fixed_md = fix_degree_chars(raw_md)

# -------------------------------
# Replace formula placeholders using MathPix
# -------------------------------
# Set your MathPix credentials here
MATHPIX_APP_ID = "YOUR_APP_ID"
MATHPIX_APP_KEY = "YOUR_APP_KEY"

while "<!-- formula-not-decoded -->" in fixed_md:
    # Replace one placeholder at a time
    latex = extract_formula_latex(path_file, MATHPIX_APP_ID, MATHPIX_APP_KEY)
    if not latex:
        latex = "FORMULA_UNRECOGNIZED"
    fixed_md = fixed_md.replace("<!-- formula-not-decoded -->", f"${latex}$", 1)

# -------------------------------
# Output the result
# -------------------------------
print("---------- Fixed Markdown with formulas ----------")
print(fixed_md)

# Save to a file
output_file = path_file.with_suffix(".md")
output_file.write_text(fixed_md, encoding="utf-8")
print(f"\nSaved fixed Markdown to {output_file}")
