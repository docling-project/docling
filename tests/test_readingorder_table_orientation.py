# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    PictureItem,
    RichTableCell,
    Size,
    TableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import Orientation
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FigureElement,
    InputFormat,
    Page,
    Table,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _make_table(
    *,
    num_rows: int,
    num_cols: int,
    orientation: Orientation,
    table_cells: list[TableCell] | None = None,
    children: list[Cluster] | None = None,
) -> Table:
    return Table(
        label=DocItemLabel.TABLE,
        id=1,
        page_no=1,
        cluster=Cluster(
            id=1,
            label=DocItemLabel.TABLE,
            bbox=BoundingBox(l=0, t=0, r=10, b=10),
            children=children or [],
        ),
        otsl_seq=[],
        num_rows=num_rows,
        num_cols=num_cols,
        table_cells=table_cells or [],
        orientation=orientation,
    )


def _make_picture(
    *,
    element_id: int,
    bbox: BoundingBox,
    children: list[Cluster] | None = None,
) -> FigureElement:
    return FigureElement(
        label=DocItemLabel.PICTURE,
        id=element_id,
        page_no=1,
        cluster=Cluster(
            id=element_id,
            label=DocItemLabel.PICTURE,
            bbox=bbox,
            children=children or [],
        ),
    )


def test_structured_table_orientation_is_carried_to_table_data():
    cell = TableCell(
        text="cell",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    table = _make_table(
        num_rows=1,
        num_cols=1,
        table_cells=[cell],
        orientation=Orientation.ROT_90,
    )

    table_data = ReadingOrderModel._table_data_from_table(table)

    assert table_data.orientation == Orientation.ROT_90
    assert table_data.table_cells == [cell]


def test_empty_table_orientation_is_carried_to_table_data():
    table = _make_table(
        num_rows=0,
        num_cols=0,
        orientation=Orientation.ROT_180,
    )

    table_data = ReadingOrderModel._table_data_from_table(table)

    assert table_data.orientation == Orientation.ROT_180
    assert table_data.num_rows == 0
    assert table_data.num_cols == 0


def test_rich_cell_fallback_table_orientation_is_carried_to_table_data():
    child = Cluster(
        id=2,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
    )
    table = _make_table(
        num_rows=0,
        num_cols=0,
        children=[child],
        orientation=Orientation.ROT_270,
    )

    table_data = ReadingOrderModel._table_data_from_table(table)

    assert table_data.orientation == Orientation.ROT_270
    assert table_data.num_rows == 1
    assert table_data.num_cols == 1


def test_picture_inside_overlapping_table_cells_is_nested_in_rich_cell():
    cells = [
        TableCell(
            text="",
            bbox=None,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
        TableCell(
            text="",
            bbox=BoundingBox(l=4, t=0, r=6, b=2),
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
        TableCell(
            text="wrong overlapping cell",
            bbox=BoundingBox(l=0, t=2, r=8, b=8),
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
        ),
        TableCell(
            text="structure",
            bbox=BoundingBox(l=2, t=3, r=10, b=9),
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
        ),
    ]
    table = _make_table(
        num_rows=2,
        num_cols=2,
        table_cells=cells,
        orientation=Orientation.ROT_0,
    )
    child_bbox = BoundingBox(l=3, t=3, r=7, b=4)
    nested_picture = _make_picture(
        element_id=2,
        bbox=BoundingBox(l=2, t=2, r=8, b=8),
        children=[
            Cluster(
                id=4,
                label=DocItemLabel.TEXT,
                bbox=child_bbox,
                cells=[
                    TextCell(
                        rect=BoundingRectangle.from_bounding_box(child_bbox),
                        text="inside picture",
                        orig="inside picture",
                        from_ocr=False,
                    )
                ],
            )
        ],
    )
    # Inside the table box, but outside every predicted cell.
    unmatched_picture = _make_picture(
        element_id=3,
        bbox=BoundingBox(l=8, t=0, r=10, b=2),
    )
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("rich-table-picture.pdf"),
            document_hash="0" * 64,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=1, size=Size(width=100, height=100))],
        assembled=AssembledUnit(
            elements=[table, nested_picture, unmatched_picture],
        ),
    )
    model = ReadingOrderModel(options=ReadingOrderOptions())
    doc = model._readingorder_elements_to_docling_doc(
        conv_res,
        model._assembled_to_readingorder_elements(conv_res),
        el_to_captions_mapping={},
        el_to_footnotes_mapping={},
        el_merges_mapping={},
    )

    table_item = doc.tables[0]
    rich_cell = table_item.data.table_cells[3]
    assert isinstance(rich_cell, RichTableCell)
    group = rich_cell.ref.resolve(doc)
    children = [child.resolve(doc) for child in group.children]
    assert len(children) == 2
    assert isinstance(children[0], TextItem)
    assert children[0].text == "structure"
    assert isinstance(children[1], PictureItem)
    picture_children = [child.resolve(doc) for child in children[1].children]
    assert len(picture_children) == 1
    assert isinstance(picture_children[0], TextItem)
    assert picture_children[0].text == "inside picture"
    body_children = [child.resolve(doc) for child in doc.body.children]
    assert len(body_children) == 2
    assert isinstance(body_children[0], TableItem)
    assert isinstance(body_children[1], PictureItem)
    doc.validate_document()
