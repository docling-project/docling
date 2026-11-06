# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, GroupLabel, Size
from docling_core.types.doc.document import DoclingDocument, ListItem
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    AssembledUnit,
    BasePageElement,
    Cluster,
    FigureElement,
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
    cells: dict[int, list[TextCell]] | None = None,
) -> DoclingDocument:
    elements: list[BasePageElement] = []
    page_positions: dict[int, int] = {}
    for index, (text, left, page_no, label) in enumerate(entries):
        top = 20 + 30 * page_positions.get(page_no, 0)
        page_positions[page_no] = page_positions.get(page_no, 0) + 1
        cluster = Cluster(
            id=index,
            label=label,
            bbox=(
                boxes[index]
                if boxes is not None
                else BoundingBox(l=left, t=top, r=450, b=top + 15)
            ),
            cells=(cells or {}).get(index, []),
        )
        if label == DocItemLabel.PICTURE:
            elements.append(
                FigureElement(id=index, label=label, page_no=page_no, cluster=cluster)
            )
        else:
            elements.append(
                TextElement(
                    id=index,
                    label=label,
                    text=text,
                    page_no=page_no,
                    cluster=cluster,
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


@pytest.mark.parametrize("next_page", [1, 2])
def test_a_picture_stays_before_the_next_list(next_page: int) -> None:
    doc = convert_entries(
        [
            ("1. Parent", 118, 1, DocItemLabel.LIST_ITEM),
            ("o First child", 154, 1, DocItemLabel.LIST_ITEM),
            ("", 60, 1, DocItemLabel.PICTURE),
            ("o Separate item", 154, next_page, DocItemLabel.LIST_ITEM),
        ]
    )
    labels = [item.label for item, _ in doc.iterate_items()]
    assert labels == [
        DocItemLabel.LIST_ITEM,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.PICTURE,
        DocItemLabel.LIST_ITEM,
    ]
    items = [item for item, _ in doc.iterate_items() if isinstance(item, ListItem)]
    assert items[1].parent != items[2].parent


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


@pytest.mark.parametrize("opening_quote", ["«", "\u201c", '"'])
@pytest.mark.parametrize("markers", [("1.", "2.", "3."), ("(1)", "(2)", "(3)")])
def test_opening_quote_does_not_indent_numbered_siblings(
    opening_quote: str, markers: tuple[str, str, str]
) -> None:
    doc = convert_entries(
        [
            (
                f"{opening_quote}{markers[0]} First item.",
                113.4,
                1,
                DocItemLabel.LIST_ITEM,
            ),
            (f"{markers[1]} Second item.", 118.9, 1, DocItemLabel.LIST_ITEM),
            (
                f"{markers[2]} Third item.{opening_quote}",
                118.9,
                1,
                DocItemLabel.LIST_ITEM,
            ),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 2, 2]
    assert len({item.parent.cref for item, _ in items if item.parent is not None}) == 1
    assert items[0][0].orig.startswith(opening_quote)


def test_quoted_parent_still_has_an_indented_sublist() -> None:
    doc = convert_entries(
        [
            ("«1. Parent", 113.4, 1, DocItemLabel.LIST_ITEM),
            ("a) First child", 154, 1, DocItemLabel.LIST_ITEM),
            ("b) Second child", 154, 1, DocItemLabel.LIST_ITEM),
            ("2. Next parent»", 118.9, 1, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4, 4, 2]
    assert items[1][0].parent is not None
    assert items[1][0].parent.resolve(doc).parent == items[0][0].get_ref()
    assert items[0][0].parent == items[3][0].parent


@pytest.mark.parametrize("child_marker", ["a)", "1."])
def test_small_indent_with_a_new_numbering_style_still_nests(child_marker: str) -> None:
    doc = convert_entries(
        [
            ("«1. Parent", 113.4, 1, DocItemLabel.LIST_ITEM),
            (f"{child_marker} Child", 118.9, 1, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4]


def test_multiline_quoted_parent_does_not_absorb_a_real_indent() -> None:
    first_line = BoundingBox(l=113.4, t=20, r=450, b=30)
    doc = convert_entries(
        [
            (
                "«1. A parent that spans several lines.",
                113.4,
                1,
                DocItemLabel.LIST_ITEM,
            ),
            ("2. A nested item.", 149.4, 1, DocItemLabel.LIST_ITEM),
        ],
        boxes=[
            BoundingBox(l=113.4, t=20, r=450, b=75),
            BoundingBox(l=149.4, t=90, r=450, b=100),
        ],
        cells={
            0: [
                TextCell(
                    index=0,
                    text="«1. A parent",
                    orig="«1. A parent",
                    from_ocr=False,
                    rect=BoundingRectangle.from_bounding_box(first_line),
                )
            ]
        },
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4]


def test_nested_list_continues_after_page_footnotes() -> None:
    doc = convert_entries(
        [
            ("• Parent", 118, 1, DocItemLabel.LIST_ITEM),
            ("o First child", 154, 1, DocItemLabel.LIST_ITEM),
            ("1 A page footnote.", 100, 1, DocItemLabel.FOOTNOTE),
            ("2 Another footnote.", 100, 1, DocItemLabel.FOOTNOTE),
            ("1", 250, 1, DocItemLabel.PAGE_FOOTER),
            ("Page header", 100, 2, DocItemLabel.PAGE_HEADER),
            ("o Second child", 154, 2, DocItemLabel.LIST_ITEM),
            ("• Next parent", 118, 2, DocItemLabel.LIST_ITEM),
            ("A body paragraph ends the list.", 118, 2, DocItemLabel.TEXT),
            ("o A separate list", 154, 2, DocItemLabel.LIST_ITEM),
        ]
    )

    items = [
        (item, level)
        for item, level in doc.iterate_items()
        if isinstance(item, ListItem)
    ]
    assert [level for _, level in items] == [2, 4, 4, 2, 2]
    assert items[1][0].parent == items[2][0].parent
    assert items[4][0].parent != items[1][0].parent
    footnotes = [item for item in doc.texts if item.label == DocItemLabel.FOOTNOTE]
    assert [item.text for item in footnotes] == [
        "1 A page footnote.",
        "2 Another footnote.",
    ]
    assert all(item.prov[0].page_no == 1 for item in footnotes)
