# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A checkbox the layout boxed together with its label is keyed by that label."""

from unittest.mock import MagicMock

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.items.form import FieldItem
from docling_core.types.doc.page import BoundingRectangle, PdfWidget, TextCell

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page

from .test_form_extraction import _materialize

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
        (region.source_container_id, [item.key_text for item in region.items])
        for region in page.predictions.field_regions
    ] == [(row.id, [text]) for row, text in zip(rows, paragraphs)]
    order = []
    for item, _ in doc.iterate_items(with_groups=False):
        if isinstance(item, FieldItem):
            order.append(item.children[0].resolve(doc).text)
        elif item.label not in {DocItemLabel.FIELD_KEY, DocItemLabel.FIELD_VALUE}:
            order.append(item.text)  # the boxes' own checkbox items are empty
    assert [text for text in order if text] == [note, "Fotocópias de:", *paragraphs]


def test_two_checkboxes_in_one_block_are_left_to_the_search() -> None:
    # A Yes/No pair the layout boxed as one checkbox block: the block's text
    # keys neither box as a whole; each box reads its own option key.
    block = _block(
        1,
        DocItemLabel.CHECKBOX_UNSELECTED,
        BoundingBox(l=10, t=10, r=90, b=30),
        (0, "Sim", BoundingBox(l=24, t=11, r=40, b=19)),
        (1, "Não", BoundingBox(l=24, t=21, r=40, b=29)),
    )
    widgets = [
        _checkbox(0, BoundingBox(l=12, t=11, r=20, b=19)),
        _checkbox(1, BoundingBox(l=12, t=21, r=20, b=29)),
    ]
    page = _page([block], widgets)

    _materialize(page)

    assert [
        item.key_text
        for region in page.predictions.field_regions
        for item in region.items
    ] == ["Sim", "Não"]
