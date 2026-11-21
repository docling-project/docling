# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size

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


@pytest.mark.parametrize(
    ("continuation_page", "following_page", "expected"),
    [
        pytest.param(
            2, 2, ["The first paragraph continues here", "Another paragraph"], id="page"
        ),
        pytest.param(
            1,
            1,
            ["The first paragraph continues here", "Another paragraph"],
            id="column",
        ),
        pytest.param(
            2,
            3,
            ["The first paragraph continues here Another paragraph"],
            id="three-pages",
        ),
    ],
)
def test_paragraph_merge_requires_a_boundary_after_each_fragment(
    continuation_page: int, following_page: int, expected: list[str]
) -> None:
    continuation_left = 320 if continuation_page == 1 else 50
    first_right = 280 if continuation_page == 1 else 550
    rows = [
        (1, (50, 700, first_right, 740), "The first paragraph"),
        (continuation_page, (continuation_left, 60, 550, 100), "continues here"),
        (following_page, (continuation_left, 120, 550, 160), "Another paragraph"),
    ]
    elements = [
        TextElement(
            id=cid,
            label=DocItemLabel.TEXT,
            page_no=page_no,
            text=text,
            cluster=Cluster(
                id=cid,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
            ),
        )
        for cid, (page_no, (left, top, right, bottom), text) in enumerate(rows)
    ]
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("input.pdf"),
            document_hash="0" * 64,
            valid=True,
            format=InputFormat.PDF,
        ),
        pages=[
            Page(page_no=page_no, size=Size(width=600, height=800))
            for page_no in range(1, following_page + 1)
        ],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)

    assert [item.text for item in doc.texts] == expected
