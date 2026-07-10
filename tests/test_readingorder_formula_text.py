from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.document import DoclingDocument

from docling.datamodel.base_models import Cluster, TextElement
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def test_formula_element_keeps_extracted_text():
    """A FORMULA element must carry its extracted text into the document.

    With formula enrichment disabled (the default), the assembled item's text
    is never backfilled, so leaving it empty loses the content in every
    export (issue #3780). The enrichment stage overwrites text uncondition-
    ally when it runs, so carrying the extracted text is safe either way.
    """
    extracted = "................ OCR line misclassified as formula"
    element = TextElement(
        label=DocItemLabel.FORMULA,
        id=1,
        page_no=1,
        cluster=Cluster(
            id=1,
            label=DocItemLabel.FORMULA,
            bbox=BoundingBox(l=0, t=0, r=10, b=10),
        ),
        text=extracted,
    )
    doc = DoclingDocument(name="test")
    model = ReadingOrderModel(ReadingOrderOptions())

    new_item, _ = model._handle_text_element(element, doc, None, 792.0)

    assert new_item.text == extracted
    assert new_item.orig == extracted
