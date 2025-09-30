# %% [markdown]
# Inspect the contents associated with each picture in a converted document.
#
# What this example does
# - Converts a PDF and iterates over each PictureItem.
# - Prints the caption and the textual items contained within the picture region.
#
# How to run
# - From the repo root: `python docs/examples/inspect_picture_content.py`.
#
# Notes
# - Uncomment `picture.get_image(doc).show()` to visually inspect each picture.
# - Adjust `source` to point to a different PDF if desired.

# %%

from docling_core.types.doc import TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Change this to a local path if desired
source = "tests/data/pdf/amt_handbook_sample.pdf"

pipeline_options = PdfPipelineOptions()
# Higher scale yields sharper crops when inspecting picture content.
pipeline_options.images_scale = 2
pipeline_options.generate_page_images = True

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

result = doc_converter.convert(source)

doc = result.document

for picture in doc.pictures:
    # picture.get_image(doc).show()  # display the picture
    print(picture.caption_text(doc), " contains these elements:")

    for item, level in doc.iterate_items(root=picture, traverse_pictures=True):
        if isinstance(item, TextItem):
            print(item.text)

    print("\n")
