# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FigureElement,
    InputFormat,
    Page,
    PageElement,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.postprocessing.reading_order_rb import (
    PageElement as ReadingOrderPageElement,
    ReadingOrderPredictor,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def test_side_column_caption_links_to_its_picture() -> None:
    # Figure A and its caption share a top edge. Reading down the main column
    # puts body text between them, while figure B is next in the margin column.
    side_caption = "Figure 15-7. A picture with a caption in the side column."
    lower_caption = "Figure 15-8. A picture with a caption below it."
    elements: list[PageElement] = []
    for cid, (label, coords, text) in enumerate(
        [
            (DocItemLabel.PICTURE, (73, 83, 354, 228), ""),
            (DocItemLabel.CAPTION, (381, 83, 512, 188), side_caption),
            (DocItemLabel.TEXT, (73, 242, 354, 412), "Body text below figure A."),
            (DocItemLabel.TEXT, (73, 432, 354, 592), "More body text."),
            (DocItemLabel.PICTURE, (382, 433, 512, 572), ""),
            (DocItemLabel.CAPTION, (381, 577, 512, 602), lower_caption),
        ]
    ):
        left, top, right, bottom = coords
        cluster = Cluster(
            id=cid, label=label, bbox=BoundingBox(l=left, t=top, r=right, b=bottom)
        )
        if label == DocItemLabel.PICTURE:
            elements.append(
                FigureElement(id=cid, label=label, page_no=1, cluster=cluster)
            )
        else:
            elements.append(
                TextElement(id=cid, label=label, page_no=1, cluster=cluster, text=text)
            )

    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("input.pdf"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=1, size=Size(width=612, height=792))],
        assembled=AssembledUnit(elements=elements, body=elements),
    )
    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    upper_picture, lower_picture = sorted(
        doc.pictures, key=lambda pic: pic.prov[0].bbox.l
    )
    assert [ref.resolve(doc).text for ref in upper_picture.captions] == [side_caption]
    assert [ref.resolve(doc).text for ref in lower_picture.captions] == [lower_caption]
    assert upper_picture.captions[0].resolve(doc).parent == upper_picture.get_ref()
    assert doc.export_to_markdown().count(side_caption) == 1
    doc.validate_document()


@pytest.mark.parametrize(
    ("caption_box", "body_box", "caption_page", "expected"),
    [
        pytest.param((10, 650, 73, 545), (10, 520, 73, 500), 1, {2: [0]}, id="left"),
        pytest.param((400, 650, 460, 545), (100, 490, 350, 450), 1, {}, id="wide-gap"),
        pytest.param(
            (370, 520, 460, 450), (100, 490, 350, 450), 1, {}, id="low-overlap"
        ),
        pytest.param(
            (370, 650, 460, 545), (355, 625, 365, 600), 1, {}, id="blocked-gap"
        ),
        pytest.param(
            (370, 650, 460, 545), (100, 490, 350, 450), 2, {}, id="other-page"
        ),
    ],
)
def test_side_caption_geometry(
    caption_box: tuple[int, int, int, int],
    body_box: tuple[int, int, int, int],
    caption_page: int,
    expected: dict[int, list[int]],
) -> None:
    # Body text breaks the run of graphics in the reading order.
    elements = []
    for cid, (label, coords, page_no) in enumerate(
        [
            (DocItemLabel.CAPTION, caption_box, caption_page),
            (DocItemLabel.TEXT, body_box, 1),
            (DocItemLabel.PICTURE, (100, 650, 350, 500), 1),
        ]
    ):
        left, top, right, bottom = coords
        elements.append(
            ReadingOrderPageElement(
                cid=cid,
                label=label,
                page_no=page_no,
                page_size=Size(width=520, height=792),
                coord_origin=CoordOrigin.BOTTOMLEFT,
                l=left,
                t=top,
                r=right,
                b=bottom,
            )
        )
    elements.sort(key=lambda element: element.page_no)

    assert ReadingOrderPredictor().predict_to_captions(elements) == expected
