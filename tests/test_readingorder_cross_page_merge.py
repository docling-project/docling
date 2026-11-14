# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A paragraph that continues on a page of another height keeps its boxes.

The two pages have different heights. Each half of the paragraph sits 50 pt
below the top of its page, so in the bottom-left origin each box must be
measured from the height of its own page.
"""

from pathlib import PurePath

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size

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

PAGE_HEIGHTS = {1: 792.0, 2: 700.0}


def _text(cid: int, page_no: int, text: str) -> TextElement:
    return TextElement(
        label=DocItemLabel.TEXT,
        id=cid,
        page_no=page_no,
        text=text,
        cluster=Cluster(
            id=cid,
            label=DocItemLabel.TEXT,
            bbox=BoundingBox(l=72, t=50, r=540, b=90),
        ),
    )


def test_continuation_box_uses_the_height_of_its_own_page() -> None:
    elements = [
        _text(0, 1, "The sentence starts on the first page and continues on the"),
        _text(0, 2, "second page, which is shorter than the first page."),
    ]
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("input.pdf"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.PDF,
        ),
        pages=[
            Page(page_no=page_no, size=Size(width=612, height=height))
            for page_no, height in PAGE_HEIGHTS.items()
        ],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    (item,) = doc.texts
    assert [prov.page_no for prov in item.prov] == [1, 2]
    for prov in item.prov:
        height = PAGE_HEIGHTS[prov.page_no]
        assert prov.bbox == BoundingBox(
            l=72,
            t=height - 50,
            r=540,
            b=height - 90,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
