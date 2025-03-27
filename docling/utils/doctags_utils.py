from pathlib import Path
from typing import cast

from docling_core.types.doc import DoclingDocument, Size
from docling_core.types.doc.document import DocTagsDocument
from docling_core.types.doc.tokens import DocumentToken
from PIL import Image as PILImage


def remove_doctags_content(doctags: str, images: list[PILImage.Image]) -> str:
    dt_list = (
        doctags.removeprefix(f"<{DocumentToken.DOCUMENT.value}>")
        .removesuffix(f"\n</{DocumentToken.DOCUMENT.value}>")
        .split(f"\n<{DocumentToken.PAGE_BREAK.value}>\n")
    )
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        cast(list[str | Path], dt_list), cast(list[PILImage.Image | Path], images)
    )
    doc = DoclingDocument(name="dummy")
    doc.load_from_doctags(doctags_doc)

    for idx, image in enumerate(images):
        size = Size(width=float(image.width), height=float(image.height))
        doc.add_page(page_no=idx + 1, size=size)

    return doc.export_to_document_tokens(add_content=False)
