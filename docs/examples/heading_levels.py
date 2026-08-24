# %% [markdown]
# Recover the section-header hierarchy of a PDF.
#
# What this example does
# - Converts a PDF with heading-level inference enabled.
# - Prints the recovered outline, indented by level, and the Markdown headings it produces.
#
# Prerequisites
# - Install Docling. No extra model is downloaded for this stage.
#
# How to run
# - From the repo root: `python docs/examples/heading_levels.py`.
# - The script prints the heading tree and a Markdown excerpt to stdout.
#
# Input document
# - Defaults to `tests/data/pdf/sources/2203.01017v2.pdf`, limited to the first pages.
#   Change `input_doc_path` as needed.
#
# Notes
# - Without `HeadingHierarchyOptions(enabled=True)` every heading stays at level 1 and the
#   Markdown export is a flat list of `#` headings.
# - `generate_parsed_pages=True` is only needed for the font-style signal, which reads the
#   parsed PDF cells. Bookmarks and numbering work without it.
# - See the [PDF heading levels](../../usage/heading_levels/) guide for the full set of options.

# %%

from pathlib import Path

from docling_core.types.doc.document import SectionHeaderItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    input_doc_path = Path("tests/data/pdf/sources/2203.01017v2.pdf")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    # Keep the parsed PDF cells around: the font-style signal is read from them.
    pipeline_options.generate_parsed_pages = True
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    doc = converter.convert(input_doc_path, page_range=(1, 6)).document

    print("Recovered heading hierarchy:")
    for item in doc.texts:
        if isinstance(item, SectionHeaderItem):
            print(f"{'  ' * (item.level - 1)}L{item.level}  {item.text}")

    print("\nMarkdown headings:")
    for line in doc.export_to_markdown().splitlines():
        if line.startswith("#"):
            print(line)


if __name__ == "__main__":
    main()
