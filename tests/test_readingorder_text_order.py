# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)

_PAGE_SIZE = Size(width=612, height=792)


def _element(
    index: int,
    label: DocItemLabel,
    bbox: tuple[float, float, float, float],
    text: str = "",
) -> TextElement:
    # With text, the cluster holds one text cell, numbered in storage order.
    # Without text, it has no cells, as a formula drawn as vector graphics.
    left, top, right, bottom = bbox
    box = BoundingBox(l=left, t=top, r=right, b=bottom)
    cells = []
    if text:
        cells = [
            TextCell(
                index=index,
                rect=BoundingRectangle.from_bounding_box(box),
                text=text,
                orig=text,
                from_ocr=False,
            )
        ]
    return TextElement(
        label=label,
        id=index,
        text=text,
        page_no=1,
        cluster=Cluster(id=index, label=label, bbox=box, cells=cells),
    )


def _reading_order(elements: list[TextElement]) -> list[int]:
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    conv_res = ConversionResult(
        input=input_doc,
        pages=[Page(page_no=1, size=_PAGE_SIZE)],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    page_height = _PAGE_SIZE.height
    id_by_box = {
        element.cluster.bbox.to_bottom_left_origin(page_height).as_tuple(): element.id
        for element in elements
    }
    return [id_by_box[item.prov[0].bbox.as_tuple()] for item, _ in doc.iterate_items()]


def test_formulas_without_text_cells_stay_in_their_columns() -> None:
    # Layout postprocessing stores the clusters with text cells first, in text
    # order, and the two formulas, which have none, after them by position. So
    # the formulas get consecutive cids, although they are in different columns.
    elements = [
        _element(0, DocItemLabel.TEXT, (40, 60, 290, 200), "Left column, first."),
        _element(1, DocItemLabel.TEXT, (40, 260, 290, 400), "Left column, second."),
        _element(2, DocItemLabel.TEXT, (322, 60, 572, 150), "Right column, first."),
        _element(3, DocItemLabel.TEXT, (322, 260, 572, 400), "Right column, second."),
        _element(4, DocItemLabel.FORMULA, (60, 210, 270, 250)),
        _element(5, DocItemLabel.FORMULA, (342, 211, 552, 249)),
    ]

    assert _reading_order(elements) == [0, 4, 1, 2, 5, 3]


def test_line_split_into_two_clusters_stays_one_row() -> None:
    # Each line of a label/value list is split into two clusters on one
    # baseline. The text cells go label, value, label, value, so each value
    # directly follows its label in the cids and continues the same row.
    elements = [
        _element(0, DocItemLabel.TEXT, (72, 80, 160, 92), "Name:"),
        _element(1, DocItemLabel.TEXT, (180, 80, 400, 92), "Ada Lovelace"),
        _element(2, DocItemLabel.TEXT, (72, 100, 160, 112), "Role:"),
        _element(3, DocItemLabel.TEXT, (180, 100, 400, 112), "Analyst"),
        _element(4, DocItemLabel.TEXT, (72, 120, 160, 132), "Year:"),
        _element(5, DocItemLabel.TEXT, (180, 120, 400, 132), "1843"),
    ]

    assert _reading_order(elements) == [0, 1, 2, 3, 4, 5]
