# %% [markdown]
# Translate extracted text content and regenerate Markdown with embedded images.
#
# What this example does
# - Converts a PDF and saves original Markdown with embedded images.
# - Translates text elements and table cell contents, then saves a translated Markdown.
#
# Prerequisites
# - Install Docling. Add a translation library of your choice inside `translate()`.
#
# How to run
# - From the repo root: `python docs/examples/translate.py`.
# - The script writes original and translated Markdown to `scratch/`.
#
# Notes
# - `translate()` is a placeholder; integrate your preferred translation API/client.
# - Image generation is enabled to preserve embedded images in the output.

# %%

import logging
from pathlib import Path

from docling_core.types.doc import ImageRefMode, TableItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0


# FIXME: put in your favorite translation code ....
def translate(text: str, src: str = "en", dest: str = "de"):
    _log.warning("!!! IMPLEMENT HERE YOUR FAVORITE TRANSLATION CODE!!!")
    # from googletrans import Translator

    # Initialize the translator
    # translator = Translator()

    # Translate text from English to German
    # text = "Hello, how are you?"
    # translated = translator.translate(text, src="en", dest="de")

    return text


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")  # ensure this directory exists before saving

    # Important: For operating with page images, we must keep them, otherwise the DocumentConverter
    # will destroy them for cleaning up memory.
    # This is done by setting PdfPipelineOptions.images_scale, which also defines the scale of images.
    # scale=1 correspond of a standard 72 DPI image
    # The PdfPipelineOptions.generate_* are the selectors for the document elements which will be enriched
    # with the image field
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(input_doc_path)
    conv_doc = conv_res.document
    doc_filename = conv_res.input.file.name

    # Save markdown with embedded pictures in original text
    # Tip: create the `scratch/` folder first or adjust `output_dir`.
    md_filename = output_dir / f"{doc_filename}-with-images-orig.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TextItem):
            element.orig = element.text
            element.text = translate(text=element.text)

        elif isinstance(element, TableItem):
            for cell in element.data.table_cells:
                cell.text = translate(text=cell.text)

    # Save markdown with embedded pictures in translated text
    md_filename = output_dir / f"{doc_filename}-with-images-translated.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)


if __name__ == "__main__":
    main()
