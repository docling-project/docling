# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    Page,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.postprocessing.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)

# (label, page_no, text, (left, top, right, bottom)) in top-left page coordinates.
Fragment = tuple[DocItemLabel, int, str, tuple[float, float, float, float]]

SH = DocItemLabel.SECTION_HEADER


def _convert(fragments: list[Fragment], num_pages: int = 1) -> list[tuple[str, int]]:
    elements = []
    for cid, (label, page_no, text, (left, top, right, bottom)) in enumerate(fragments):
        cluster = Cluster(
            id=cid,
            label=label,
            bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        )
        elements.append(
            TextElement(
                id=cid, page_no=page_no, label=label, text=text, cluster=cluster
            )
        )
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    conv_res = ConversionResult(
        input=input_doc,
        pages=[
            Page(page_no=page_no, size=Size(width=612, height=792))
            for page_no in range(1, num_pages + 1)
        ],
        assembled=AssembledUnit(elements=elements, body=elements),
    )

    doc = ReadingOrderModel(ReadingOrderOptions())(conv_res)
    return [(item.text, len(item.prov)) for item, _ in doc.iterate_items()]


def test_wrapped_headings_in_one_column_are_merged() -> None:
    # Box positions from a single-column Seattle Municipal Code sample (issue #4016).
    items = _convert(
        [
            (SH, 1, "3.32.010 Seattle Public Utilities-Gen-", (54, 74.1, 300, 84.3)),
            (SH, 1, "eral Manager and Chief Executive Officer", (72, 88.1, 290, 98.3)),
            (DocItemLabel.TEXT, 1, "There is hereby created.", (54, 105, 500, 116)),
            (SH, 1, "3.30.250 Functional review of board", (54, 143.1, 260, 153.3)),
            (SH, 1, "operations-Abolition or continuation", (72, 157.1, 280, 167.3)),
            (SH, 1, "of Board.", (72, 171.1, 120, 181.3)),
        ]
    )

    assert items == [
        (
            "3.32.010 Seattle Public Utilities-General Manager and Chief Executive Officer",
            2,
        ),
        ("There is hereby created.", 1),
        (
            "3.30.250 Functional review of board operations-Abolition or continuation of Board.",
            3,
        ),
    ]


def test_heading_wrapped_across_pages_is_merged() -> None:
    items = _convert(
        [
            (SH, 1, "Chapter 3.32 - Seattle Public Utilities", (60, 63.4, 300, 74.5)),
            (DocItemLabel.TEXT, 1, "Section background paragraph.", (60, 90, 550, 670)),
            (SH, 1, "3.32.010 Seattle Public Utilities-Gen-", (60, 684.4, 300, 695.5)),
            (SH, 2, "eral Manager and Chief Executive Officer", (60, 60, 300, 71)),
            (DocItemLabel.TEXT, 2, "There is hereby created.", (60, 80, 550, 91)),
        ],
        num_pages=2,
    )

    assert items == [
        ("Chapter 3.32 - Seattle Public Utilities", 1),
        ("Section background paragraph.", 1),
        (
            "3.32.010 Seattle Public Utilities-General Manager and Chief Executive Officer",
            2,
        ),
        ("There is hereby created.", 1),
    ]


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        pytest.param(
            [
                (SH, 1, "Functional review of", (230, 74, 382, 84)),
                (SH, 1, "board operations and continuation", (180, 88, 432, 98)),
            ],
            "Functional review of board operations and continuation",
            id="centered-wider-second-line",
        ),
        pytest.param(
            [
                (SH, 1, "3.32.010 Seattle Public Utilities—", (54, 74, 300, 84)),
                (SH, 1, "General Manager", (72, 88, 300, 98)),
            ],
            "3.32.010 Seattle Public Utilities—General Manager",
            id="wrap-after-em-dash",
        ),
        pytest.param(
            [
                (SH, 1, "Article 12 of Chapter 3", (54, 74, 300, 84)),
                (SH, 1, "and related provisions", (54, 88, 300, 98)),
            ],
            "Article 12 of Chapter 3 and related provisions",
            id="wrap-after-digit",
        ),
    ],
)
def test_heading_wrap_variants_are_merged(
    fragments: list[Fragment], expected: str
) -> None:
    assert _convert(fragments) == [(expected, 2)]


@pytest.mark.parametrize("coord_origin", [CoordOrigin.TOPLEFT, CoordOrigin.BOTTOMLEFT])
def test_predict_merges_detects_wrap_in_either_coord_origin(
    coord_origin: CoordOrigin,
) -> None:
    page_size = Size(width=612, height=792)
    elements = []
    for cid, (text, bbox) in enumerate(
        [
            ("Functional review of board", BoundingBox(l=54, t=74, r=300, b=84)),
            ("operations of the board", BoundingBox(l=72, t=88, r=300, b=98)),
        ]
    ):
        if coord_origin == CoordOrigin.BOTTOMLEFT:
            bbox = bbox.to_bottom_left_origin(page_size.height)
        elements.append(
            PageElement(
                cid=cid,
                label=SH,
                text=text,
                page_no=1,
                page_size=page_size,
                l=bbox.l,
                t=bbox.t,
                r=bbox.r,
                b=bbox.b,
                coord_origin=bbox.coord_origin,
            )
        )

    assert ReadingOrderPredictor().predict_merges(sorted_elements=elements) == {0: [1]}


@pytest.mark.parametrize(
    ("fragments", "num_pages"),
    [
        pytest.param(
            [
                (SH, 1, "Definitions", (54, 74, 200, 84)),
                (SH, 1, "General provisions", (54, 88, 200, 98)),
            ],
            1,
            id="consecutive-headings-tight-spacing",
        ),
        pytest.param(
            [
                (SH, 1, "Chapter 3.32 Seattle Public Utilities", (54, 74, 300, 84)),
                (SH, 1, "3.32.010 General Manager", (54, 88, 300, 98)),
            ],
            1,
            id="chapter-then-numbered-section",
        ),
        pytest.param(
            [
                (SH, 1, "Scope of review", (54, 74, 300, 84)),
                (SH, 1, "operations of the board", (54, 120, 300, 130)),
            ],
            1,
            id="large-vertical-gap",
        ),
        pytest.param(
            [
                (SH, 1, "Scope and intent", (54, 740, 300, 750)),
                (SH, 2, "Applicability", (54, 60, 300, 70)),
            ],
            2,
            id="headings-at-page-break",
        ),
    ],
)
def test_separate_headings_are_not_merged(
    fragments: list[Fragment], num_pages: int
) -> None:
    items = _convert(fragments, num_pages=num_pages)

    assert items == [(text, 1) for _, _, text, _ in fragments]


def test_stacked_text_in_one_column_is_not_merged() -> None:
    # The same-column rule is for headings only; text blocks keep their own items.
    items = _convert(
        [
            (DocItemLabel.TEXT, 1, "first paragraph ends with", (54, 74, 300, 84)),
            (DocItemLabel.TEXT, 1, "second paragraph", (54, 88, 300, 98)),
            (SH, 1, "Heading", (54, 120, 300, 130)),
        ]
    )

    assert items == [
        ("first paragraph ends with", 1),
        ("second paragraph", 1),
        ("Heading", 1),
    ]
