from docling_core.types.doc import TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

source = "tests/data/pdf/amt_handbook_sample.pdf"

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2
pipeline_options.generate_page_images = True

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

result = doc_converter.convert(source)

doc = result.document

for picture in doc.pictures:
    # picture.get_image(doc).show() # display the picture
    print(picture.caption_text(doc), " contains these elements:")

    for item, level in doc.iterate_items(root=picture, traverse_pictures=True):
        if isinstance(item, TextItem):
            print(item.text)

    print("\n")
