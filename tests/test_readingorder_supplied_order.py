# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A stage upstream of reading order may already know the order of a page.

A layout model can predict reading order (docling issue #3958) and the
structure tree of a tagged PDF defines it (ISO 32000-2, 14.7). The reading-order
stage honours ``Cluster.reading_order`` on any page where every element carries
an index, and runs the predictor on every other page as before.
"""

from pathlib import PurePath

from docling_core.types.doc import BoundingBox, DocItemLabel, GroupLabel, Size
from docling_core.types.doc.document import GroupItem

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    ContainerElement,
    InputFormat,
    Page,
    PageElement,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)

PAGE = Size(width=500, height=500)


def _text(
    cid: int,
    text: str,
    top: float,
    *,
    page_no: int = 1,
    order: int | None = None,
    label: DocItemLabel = DocItemLabel.TEXT,
) -> TextElement:
    cluster = Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(l=50, t=top, r=450, b=top + 40),
        reading_order=order,
    )
    return TextElement(label=label, id=cid, text=text, page_no=page_no, cluster=cluster)


def _conversion_result(elements: list[PageElement], pages: int = 1) -> ConversionResult:
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    return ConversionResult(
        input=input_doc,
        pages=[Page(page_no=no, size=PAGE) for no in range(1, pages + 1)],
        assembled=AssembledUnit(elements=elements, body=elements),
    )


def _texts(elements: list[PageElement], pages: int = 1) -> list[str]:
    doc = ReadingOrderModel(ReadingOrderOptions())(_conversion_result(elements, pages))
    return [item.text for item in doc.texts]


def test_supplied_order_replaces_the_predictor_on_a_fully_indexed_page() -> None:
    # Three paragraphs stacked top to bottom. The predictor reads them in
    # geometric order; a supplied order that contradicts geometry must win.
    def page(order: tuple[int | None, int | None, int | None]) -> list[PageElement]:
        return [
            _text(1, "top", 50, order=order[0]),
            _text(2, "middle", 200, order=order[1]),
            _text(3, "bottom", 350, order=order[2]),
        ]

    assert _texts(page((None, None, None))) == ["top", "middle", "bottom"]
    assert _texts(page((1, 2, 0))) == ["bottom", "top", "middle"]


def test_page_with_any_unindexed_element_is_predicted_whole() -> None:
    # One element without an index disqualifies the page: the indices that are
    # present are ignored rather than interleaved with predicted positions.
    elements = [
        _text(1, "top", 50, order=2),
        _text(2, "middle", 200, order=None),
        _text(3, "bottom", 350, order=0),
    ]
    assert _texts(elements) == ["top", "middle", "bottom"]


def test_supplied_and_predicted_pages_keep_page_order() -> None:
    # Page 1 is fully indexed (reverse of geometry); page 2 is not indexed and
    # goes through the predictor. Pages still follow one another in order.
    # Full stops keep the predictor's paragraph-merge heuristic from joining
    # fragments across pages, which would hide the ordering under test.
    elements = [
        _text(1, "p1 top.", 50, page_no=1, order=1),
        _text(2, "p1 bottom.", 350, page_no=1, order=0),
        _text(3, "p2 top.", 50, page_no=2),
        _text(4, "p2 bottom.", 350, page_no=2),
    ]
    assert _texts(elements, pages=2) == [
        "p1 bottom.",
        "p1 top.",
        "p2 top.",
        "p2 bottom.",
    ]


def test_supplied_order_applies_within_a_container() -> None:
    # Siblings are ordered per parent. Children of a form container carry their
    # own indices and are sorted among themselves, independently of the page's
    # top-level elements.
    child_top = _text(2, "child top", 100, order=1)
    child_bottom = _text(3, "child bottom", 200, order=0)
    container_cluster = Cluster(
        id=1,
        label=DocItemLabel.FORM,
        bbox=BoundingBox(l=0, t=80, r=500, b=260),
        reading_order=1,
        children=[child_top.cluster, child_bottom.cluster],
    )
    container = ContainerElement(
        label=DocItemLabel.FORM, id=1, page_no=1, cluster=container_cluster
    )
    after = _text(4, "after form", 400, order=0)
    elements: list[PageElement] = [container, child_top, child_bottom, after]

    doc = ReadingOrderModel(ReadingOrderOptions())(_conversion_result(elements))

    form = next(
        item
        for item in doc.groups
        if isinstance(item, GroupItem) and item.label == GroupLabel.FORM_AREA
    )
    assert [child.resolve(doc).text for child in form.children] == [
        "child bottom",
        "child top",
    ]
    # At the top level the paragraph (index 0) precedes the form (index 1)
    # although it sits lower on the page.
    top_level = [child.resolve(doc) for child in doc.body.children]
    assert [getattr(item, "text", "form") for item in top_level] == [
        "after form",
        "form",
    ]
