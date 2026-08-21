from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    Size,
)
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BasePageElement, Cluster, FigureElement
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)

PAGE_NO = 1
PAGE_HEIGHT = 100.0


def _child(cluster_id: int, label: DocItemLabel, text: str) -> Cluster:
    cell = TextCell(
        index=cluster_id,
        rect=BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=10, r_y1=0, r_x2=10, r_y2=5, r_x3=0, r_y3=5
        ),
        text=text,
        orig=text,
        from_ocr=False,
    )
    return Cluster(
        id=cluster_id,
        label=label,
        bbox=BoundingBox(l=0, t=0, r=10, b=5),
        cells=[cell],
    )


def _element_with(children: list[Cluster]) -> BasePageElement:
    return FigureElement(
        label=DocItemLabel.PICTURE,
        id=1,
        page_no=PAGE_NO,
        cluster=Cluster(
            id=1,
            label=DocItemLabel.PICTURE,
            bbox=BoundingBox(l=0, t=0, r=20, b=20),
            children=children,
        ),
    )


def _doc() -> DoclingDocument:
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=PAGE_NO, size=Size(width=100.0, height=PAGE_HEIGHT))
    return doc


def _model() -> ReadingOrderModel:
    return ReadingOrderModel(options=ReadingOrderOptions())


def test_caption_child_is_registered_on_the_picture():
    doc = _doc()
    picture = doc.add_picture()
    element = _element_with([_child(2, DocItemLabel.CAPTION, "Figure 1. A caption")])

    _model()._add_child_elements(element, picture, doc)

    assert len(picture.captions) == 1
    assert picture.captions[0].resolve(doc).text == "Figure 1. A caption"


def test_caption_child_survives_markdown_export():
    doc = _doc()
    picture = doc.add_picture()
    element = _element_with([_child(2, DocItemLabel.CAPTION, "Figure 1. A caption")])

    _model()._add_child_elements(element, picture, doc)

    assert "Figure 1. A caption" in doc.export_to_markdown()


def test_non_caption_children_are_not_registered_as_captions():
    doc = _doc()
    picture = doc.add_picture()
    element = _element_with(
        [
            _child(2, DocItemLabel.TEXT, "axis label noise"),
            _child(3, DocItemLabel.PAGE_FOOTER, "page 4"),
        ]
    )

    _model()._add_child_elements(element, picture, doc)

    assert picture.captions == []


def test_caption_child_under_a_group_needs_no_back_reference():
    # Groups are walked by the serializers, so a caption parented to one is
    # already reachable. Only floating items need the back-reference.
    doc = _doc()
    group = doc.add_group(label=GroupLabel.UNSPECIFIED)
    element = _element_with([_child(2, DocItemLabel.CAPTION, "Figure 1. A caption")])

    _model()._add_child_elements(element, group, doc)

    assert "Figure 1. A caption" in doc.export_to_markdown()
