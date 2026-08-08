"""Example: convert PDF attachments.

Requires:
    pip install docling[convert-core,format-pdf-docling]

Run:
    python docs/examples/attachments.py
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Enable attachment processing (off by default)
pdf_opts = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    process_attachments=True,
    attachments_max_depth=1,
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
    }
)

source = Path("tests/data/pdf/2305.03393v1.pdf")  # replace with a PDF that has embedded files
if source.exists():
    result = converter.convert(source)
    print(f"Status: {result.status}")
    print(f"Attachments: {len(result.document.attachments)}")
    for att in result.document.attachments:
        print(f"  - {att.name} [{att.status}] target={att.target} prov={len(att.prov)}")
    for child in result.attachments:
        print(f"  -> child {child.input.file.name}: {child.status}")
        out = Path(f"/tmp/{child.input.file.stem}.md")
        child.document.save_as_markdown(out)
        print(f"     saved to {out}")
    # Parent markdown will contain inline links for annotated attachments
    # and a trailing ## Attachments section for unanchored ones
    print(result.document.export_to_markdown()[:1000])
else:
    print(f"Demo file not found: {source} — create a PDF with embedded files to try this.")
