# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Ordering independent reading chains must not depend on input order."""

from itertools import permutations

import pytest
from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel

from docling.models.postprocessing.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
)


@pytest.mark.parametrize("input_order", list(permutations(range(3))))
def test_disjoint_graphic_head_order_is_independent_of_input(
    input_order: tuple[int, int, int],
) -> None:
    elements = [
        PageElement(
            cid=cid,
            page_no=1,
            page_size=Size(width=650, height=800),
            label=label,
            l=left,
            b=bottom,
            r=right,
            t=top,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        for cid, (label, left, bottom, right, top) in enumerate(
            [
                (DocItemLabel.TEXT, 50, 100, 200, 200),
                (DocItemLabel.TEXT, 250, 600, 400, 700),
                (DocItemLabel.PICTURE, 450, 500, 600, 600),
            ]
        )
    ]
    predictor = ReadingOrderPredictor()
    predictor.dilated_page_element = False

    result = predictor.predict_reading_order([elements[index] for index in input_order])

    assert [element.cid for element in result] == [0, 1, 2]
