# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, GroupLabel, Size
from docling_core.types.doc.document import DoclingDocument, ListItem

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


def convert_entries(
    entries: list[tuple[str, float, int, DocItemLabel]],
    *,
    boxes: list[BoundingBox] | None = None,
) -> DoclingDocument:
    elements = []
    page_positions: dict[int, int] = {}
    for index, (text, left, page_no, label) in enumerate(entries):
        top = 20 + 30 * page_positions.get(page_no, 0)
        page_positions[page_no] = page_positions.get(page_no, 0) + 1
        elements.append(
            TextElement(
                id=index,
                label=label,
                text=text,
                page_no=page_no,
                cluster=Cluster(
                    id=index,
                    label=label,
                    bbox=(
                        boxes[index]
                        if boxes is not None
                        else BoundingBox(l=left, t=top, r=450, b=top + 15)
                    ),
                ),
            )
        )
    input_doc = InputDocument.model_construct(
        file=PurePath("list.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    result = ConversionResult(
        input=input_doc,
        pages=[
            Page(page_no=page_no, size=Size(width=500, height=700))
            for page_no in sorted(page_positions)
        ],
        assembled=AssembledUnit(elements=elements, body=elements),
    )
    return ReadingOrderModel(ReadingOrderOptions())(result)


@pytest.mark.parametrize(
    ("indents", "expected_levels"),
    [
        ([118, 154, 190, 154, 118], [2, 4, 6, 4, 2]),
        ([118, 120, 117], [2, 2, 2]),
    ],
)
def test_pdf_list_indentation_controls_nesting(
    indents: list[float], expected_levels: list[int]
) -> None:
    doc = convert_entries(
        [
            (f"• Item {index}", left, 1, DocItemLabel.LIST_ITEM)
            for index, left in enumerate(indents)
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == expected_levels
    assert [item.text for item, _ in items] == [
        f"Item {index}" for index in range(len(indents))
    ]
    if len(indents) == 5:
        assert items[1][0].parent is not None
        nested_group = items[1][0].parent.resolve(doc)
        assert nested_group.label == GroupLabel.LIST
        assert nested_group.parent == items[0][0].get_ref()
        assert items[3][0].parent == items[1][0].parent
        assert items[4][0].parent == items[0][0].parent
        assert "    - Item 1\n        - Item 2" in doc.export_to_markdown()


def test_nested_pdf_list_continues_across_page_furniture() -> None:
    doc = convert_entries(
        [
            ("• Parent", 118, 1, DocItemLabel.LIST_ITEM),
            ("o First child", 154, 1, DocItemLabel.LIST_ITEM),
            ("1", 250, 1, DocItemLabel.PAGE_FOOTER),
            ("Page header", 100, 2, DocItemLabel.PAGE_HEADER),
            ("o Second child", 154, 2, DocItemLabel.LIST_ITEM),
            ("• Next parent", 118, 2, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4, 4, 2]
    assert items[1][0].parent == items[2][0].parent
    assert [item.text for item, _ in items] == [
        "Parent",
        "First child",
        "Second child",
        "Next parent",
    ]


def test_numbered_pdf_lists_keep_markers_when_nested() -> None:
    doc = convert_entries(
        [
            ("1. Parent", 118, 1, DocItemLabel.LIST_ITEM),
            ("a) Child", 154, 1, DocItemLabel.LIST_ITEM),
            ("2. Next parent", 118, 1, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4, 2]
    assert [item.marker for item, _ in items] == ["1.", "a)", "2."]
    assert all(item.enumerated for item, _ in items)


def test_body_paragraph_ends_a_nested_pdf_list() -> None:
    doc = convert_entries(
        [
            ("• Parent", 118, 1, DocItemLabel.LIST_ITEM),
            ("o Child", 154, 1, DocItemLabel.LIST_ITEM),
            ("A separate paragraph.", 118, 1, DocItemLabel.TEXT),
            ("o New list", 154, 1, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4, 2]
    assert items[2][0].parent != items[1][0].parent


def test_pdf_list_in_next_column_is_not_nested() -> None:
    doc = convert_entries(
        [
            ("• Left first", 50, 1, DocItemLabel.LIST_ITEM),
            ("• Left second", 50, 1, DocItemLabel.LIST_ITEM),
            ("• Right first", 300, 1, DocItemLabel.LIST_ITEM),
            ("• Right second", 300, 1, DocItemLabel.LIST_ITEM),
        ],
        boxes=[
            BoundingBox(l=50, t=20, r=220, b=35),
            BoundingBox(l=50, t=60, r=220, b=75),
            BoundingBox(l=300, t=20, r=470, b=35),
            BoundingBox(l=300, t=60, r=470, b=75),
        ],
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [item.text for item, _ in items] == [
        "Left first",
        "Left second",
        "Right first",
        "Right second",
    ]
    assert [level for _, level in items] == [2, 2, 2, 2]
