# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A checkbox the layout boxed together with its label is keyed by that label."""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.items.form import FieldItem
from docling_core.types.doc.page import BoundingRectangle, PdfWidget, TextCell

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page

from .test_form_extraction import _materialize, _regions

# Tall enough for a note, a caption and three rows of PJ-1 page 4.
PAGE_HEIGHT = 120


def _block(
    cluster_id: int,
    label: DocItemLabel,
    bbox: BoundingBox,
    *lines: tuple[int, str, BoundingBox],
) -> Cluster:
    cells = [
        TextCell(
            index=index,
            rect=BoundingRectangle.from_bounding_box(box),
            text=text,
            orig=text,
            from_ocr=False,
        )
        for index, text, box in lines
    ]
    return Cluster(id=cluster_id, label=label, bbox=bbox, cells=cells)


def _checkbox(index: int, bbox: BoundingBox) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            bbox.to_bottom_left_origin(page_height=PAGE_HEIGHT)
        ),
        widget_text="/Off",
        widget_field_type="/Btn",
        widget_appearance_state="/Off",
    )


def _page(clusters: list[Cluster], widgets) -> Page:
    page = Page(page_no=1, size=Size(width=595, height=PAGE_HEIGHT))
    page.parsed_page = MagicMock(widgets=widgets, hyperlinks=[])
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    return page


def test_checkbox_in_its_label_block_is_keyed_by_the_whole_block() -> None:
    # PJ-1 page 4: a "Fotocópias de:" caption under a note, over rows of a
    # drawn square and a paragraph. The layout boxes each row as one
    # unselected checkbox, square included. The search would key every box
    # by the caption, which sits closer than the paragraph on the box's
    # right; each box is keyed by its own paragraph instead, which hosts the
    # item in place and is not emitted a second time.
    note = "Devem ser apresentados os documentos do requerente."
    note_bbox = BoundingBox(l=56.7, t=2.4, r=495.8, b=9.2)
    caption_bbox = BoundingBox(l=56.7, t=19.5, r=115.5, b=26.3)
    above = [
        _block(1, DocItemLabel.TEXT, note_bbox, (0, note, note_bbox)),
        _block(
            2,
            DocItemLabel.SECTION_HEADER,
            caption_bbox,
            (1, "Fotocópias de:", caption_bbox),
        ),
    ]
    rows = [
        _block(
            3,
            DocItemLabel.CHECKBOX_UNSELECTED,
            BoundingBox(l=55.5, t=32.4, r=538.7, b=64.5),
            (2, "Bilhete de identidade", BoundingBox(l=73.7, t=37.9, r=538.6, b=44.8)),
            (3, "do requerente e demais", BoundingBox(l=73.7, t=47.5, r=538.6, b=54.4)),
            (4, "pessoas.", BoundingBox(l=73.7, t=57.1, r=165.3, b=64.0)),
        ),
        _block(
            4,
            DocItemLabel.CHECKBOX_UNSELECTED,
            BoundingBox(l=55.2, t=67.9, r=513.3, b=80.7),
            (5, "Recibos de vencimento.", BoundingBox(l=73.7, t=73.2, r=513.3, b=80.1)),
        ),
        _block(
            5,
            DocItemLabel.CHECKBOX_UNSELECTED,
            BoundingBox(l=55.4, t=84.7, r=538.9, b=106.7),
            (6, "Declarações de IVA", BoundingBox(l=73.7, t=90.2, r=538.6, b=97.1)),
            (7, "dos dois trimestres.", BoundingBox(l=73.7, t=99.8, r=292.1, b=106.7)),
        ),
    ]
    widgets = [
        _checkbox(0, BoundingBox(l=58.8, t=34.8, r=66.8, b=42.8)),
        _checkbox(1, BoundingBox(l=58.8, t=70.2, r=66.8, b=78.2)),
        _checkbox(2, BoundingBox(l=58.8, t=87.2, r=66.8, b=95.2)),
    ]
    page = _page([*above, *rows], widgets)
    paragraphs = [" ".join(cell.text for cell in row.cells) for row in rows]

    doc = _materialize(page)

    assert [
        (source, [item.key_text for item in items]) for source, items in _regions(page)
    ] == [(row.id, [text]) for row, text in zip(rows, paragraphs)]
    order = []
    for item, _ in doc.iterate_items(with_groups=False):
        if isinstance(item, FieldItem):
            order.append(item.children[0].resolve(doc).text)
        elif item.label not in {DocItemLabel.FIELD_KEY, DocItemLabel.FIELD_VALUE}:
            order.append(item.text)  # the boxes' own checkbox items are empty
    assert [text for text in order if text] == [note, "Fotocópias de:", *paragraphs]


def _text_field(index: int, bbox: BoundingBox) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            bbox.to_bottom_left_origin(page_height=PAGE_HEIGHT)
        ),
        widget_text="",
        widget_field_type="/Tx",
    )


def _keys(page: Page) -> list[str]:
    return [item.key_text for _, items in _regions(page) for item in items]


def test_hosted_checkbox_key_is_located_by_its_text() -> None:
    # The block boxes the drawn square with its label, but the square is the
    # value's place: the key's location covers the label's text only.
    text_bbox = BoundingBox(l=73.7, t=37.9, r=300.0, b=44.8)
    block = _block(
        1,
        DocItemLabel.CHECKBOX_UNSELECTED,
        BoundingBox(l=55.5, t=32.4, r=300.0, b=46.0),
        (0, "Recibos de vencimento.", text_bbox),
    )
    page = _page([block], [_checkbox(0, BoundingBox(l=58.8, t=34.8, r=66.8, b=42.8))])

    doc = _materialize(page)

    (item,) = doc.field_items
    key = item.children[0].resolve(doc)
    assert key.text == "Recibos de vencimento."
    box = key.prov[0].bbox.to_top_left_origin(PAGE_HEIGHT)
    assert (box.l, box.t, box.r, box.b) == pytest.approx(
        (text_bbox.l, text_bbox.t, text_bbox.r, text_bbox.b)
    )


def test_block_label_also_keys_a_field_of_its_row() -> None:
    # "VISA Office" labels the checkbox in its block and the box on the same
    # row: the block's text keys the checkbox without taking the label from
    # the row's other field.
    label_bbox = BoundingBox(l=24, t=21, r=68, b=29)
    page = _page(
        [
            _block(
                1,
                DocItemLabel.CHECKBOX_UNSELECTED,
                BoundingBox(l=10, t=20, r=70, b=30),
                (0, "VISA Office", label_bbox),
            )
        ],
        [
            _checkbox(0, BoundingBox(l=12, t=21, r=20, b=29)),
            _text_field(1, BoundingBox(l=80, t=20, r=200, b=30)),
        ],
    )

    doc = _materialize(page)

    assert _keys(page) == ["VISA Office", "VISA Office"]
    assert all(item.label == DocItemLabel.FIELD_KEY for item in doc.texts if item.text)


def _keyed_at(page: Page) -> dict[tuple[int, int], str]:
    return {
        (round(item.values[0].bbox.l), round(item.values[0].bbox.t)): item.key_text
        for _, items in _regions(page)
        for item in items
    }


def _option(
    cluster_id: int, index: int, left: float, top: float, label: str, right: float
) -> tuple[Cluster, PdfWidget]:
    # A checkbox block, 8 pt tall, with the drawn square at its left.
    box = BoundingBox(l=left - 2, t=top - 1, r=right, b=top + 9)
    text_box = BoundingBox(l=left + 12, t=top, r=right - 4, b=top + 8)
    return (
        _block(
            cluster_id, DocItemLabel.CHECKBOX_UNSELECTED, box, (index, label, text_box)
        ),
        _checkbox(index, BoundingBox(l=left, t=top, r=left + 8, b=top + 8)),
    )


def test_checkbox_keyed_by_its_block_is_not_printed_a_second_time() -> None:
    # A heading, then three options. The first option's row also holds a
    # labelled field; the options are indented by different amounts. The
    # first option is keyed by its own block, not by the heading, so its
    # paragraph is hosted by the field and not left in the body as a second
    # checkbox.
    heading_bbox = BoundingBox(l=10, t=10, r=70, b=18)
    note_bbox = BoundingBox(l=72, t=20, r=98, b=28)
    note2_bbox = BoundingBox(l=72, t=44, r=98, b=52)
    zero, zero_box = _option(2, 1, 10, 20, "Choice 0 text", 68)
    one, one_box = _option(4, 3, 14, 32, "Choice 1 text", 102)
    two, two_box = _option(5, 4, 10, 44, "Choice 2 text", 68)
    page = _page(
        [
            _block(1, DocItemLabel.TEXT, heading_bbox, (0, "Documents", heading_bbox)),
            zero,
            _block(3, DocItemLabel.TEXT, note_bbox, (2, "Note 0:", note_bbox)),
            one,
            two,
            _block(6, DocItemLabel.TEXT, note2_bbox, (5, "Note 2:", note2_bbox)),
        ],
        [
            zero_box,
            _text_field(5, BoundingBox(l=102, t=20, r=158, b=28)),
            one_box,
            two_box,
            _text_field(6, BoundingBox(l=102, t=44, r=158, b=52)),
        ],
    )

    doc = _materialize(page)

    keyed = _keyed_at(page)
    assert keyed[(10, 20)] == "Choice 0 text"
    assert keyed[(14, 32)] == "Choice 1 text"
    assert keyed[(10, 44)] == "Choice 2 text"
    assert not [
        item
        for item in doc.texts
        if item.label
        in {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}
        and item.text
    ]
