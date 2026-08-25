from pathlib import PurePath

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DocumentOrigin,
    GroupItem,
    PictureItem,
    ProvenanceItem,
    RichTableCell,
    Size,
    TableCell,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import DoclingDocument
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


def _text_cell(
    text: str,
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
) -> TextCell:
    left, top, right, bottom = bbox
    return TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
        text=text,
        orig=text,
        from_ocr=True,
    )


def _child(cid: int, bbox: tuple[float, float, float, float], text: str) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        cells=[_text_cell(text, bbox)],
    )


def _table(table_cells: list[TableCell], children: list[Cluster]) -> Table:
    return Table(
        label=DocItemLabel.TABLE,
        id=1,
        page_no=1,
        cluster=Cluster(
            id=1,
            label=DocItemLabel.TABLE,
            bbox=BoundingBox(l=0, t=0, r=100, b=100),
            children=children,
        ),
        otsl_seq=[],
        num_rows=1,
        num_cols=1,
        table_cells=table_cells,
    )


def _new_doc() -> DoclingDocument:
    doc = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            mimetype="application/pdf", filename="test.pdf", binary_hash=1
        ),
    )
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    return doc


def test_unmatched_table_children_skips_absorbed_cells():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    absorbed = _child(2, (0, 0, 10, 10), "cell")
    orphaned = _child(3, (50, 50, 60, 60), "Signature John Doe")
    table = _table([matched_cell], [absorbed, orphaned])

    unmatched = ReadingOrderModel._unmatched_table_children(table)

    assert unmatched == [orphaned]


def test_unmatched_table_children_skips_populated_cells_without_bounds():
    unbounded_cell = TableCell(
        text="cell",
        bbox=None,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    child = _child(2, (50, 50, 60, 60), "Signature John Doe")
    table = _table([unbounded_cell], [child])

    unmatched = ReadingOrderModel._unmatched_table_children(table)

    assert unmatched == []


def test_unmatched_table_children_splits_mixed_cluster_without_mutating_source():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    absorbed = _text_cell("cell", (0, 0, 10, 10))
    orphaned = _text_cell("Signature John Doe", (50, 50, 60, 60))
    child = Cluster(
        id=2,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=0, t=0, r=60, b=60),
        cells=[absorbed, orphaned],
    )
    table = _table([matched_cell], [child])

    unmatched = ReadingOrderModel._unmatched_table_children(table)

    assert len(unmatched) == 1
    recovered = unmatched[0]
    assert recovered is not child
    assert recovered.cells == [orphaned]
    assert recovered.bbox == BoundingBox(l=50, t=50, r=60, b=60)
    assert child.cells == [absorbed, orphaned]
    assert child.bbox == BoundingBox(l=0, t=0, r=60, b=60)


def test_recover_orphaned_table_text_disabled_by_default_leaves_doc_unchanged():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    orphaned = _child(3, (50, 50, 60, 60), "Signature John Doe")
    table = _table([matched_cell], [orphaned])

    doc = _new_doc()
    prov = ProvenanceItem(
        page_no=1,
        charspan=(0, 0),
        bbox=BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
    )
    doc.add_table(
        data=ReadingOrderModel._table_data_from_table(table),
        prov=prov,
    )

    model = ReadingOrderModel(options=ReadingOrderOptions())
    model._add_unmatched_table_text(table, doc)

    assert doc.groups == []
    assert doc.texts == []
    assert "Signature" not in doc.export_to_markdown()


def test_recover_orphaned_table_text_when_all_children_are_absorbed_adds_nothing():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    absorbed = _child(2, (0, 0, 10, 10), "cell")
    table = _table([matched_cell], [absorbed])

    doc = _new_doc()
    prov = ProvenanceItem(
        page_no=1,
        charspan=(0, 0),
        bbox=BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
    )
    doc.add_table(
        data=ReadingOrderModel._table_data_from_table(table),
        prov=prov,
    )

    model = ReadingOrderModel(
        options=ReadingOrderOptions(recover_orphaned_table_text=True)
    )
    model._add_unmatched_table_text(table, doc)

    assert doc.groups == []
    assert doc.texts == []
    assert [child.cref for child in doc.body.children] == ["#/tables/0"]


def test_v1_recovery_treats_any_positive_overlap_as_matched():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    partially_absorbed = _child(2, (8, 0, 18, 10), "cell")
    table = _table([matched_cell], [partially_absorbed])

    doc = _new_doc()
    prov = ProvenanceItem(
        page_no=1,
        charspan=(0, 0),
        bbox=BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
    )
    doc.add_table(
        data=ReadingOrderModel._table_data_from_table(table),
        prov=prov,
    )

    model = ReadingOrderModel(
        options=ReadingOrderOptions(recover_orphaned_table_text=True)
    )
    model._add_unmatched_table_text(table, doc)

    assert doc.groups == []
    assert doc.texts == []
    assert [child.cref for child in doc.body.children] == ["#/tables/0"]


def test_recover_orphaned_table_text_when_enabled_appends_body_text_after_table():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    orphaned = _child(3, (50, 50, 60, 60), "Signature John Doe")
    table = _table([matched_cell], [orphaned])

    doc = _new_doc()
    prov = ProvenanceItem(
        page_no=1,
        charspan=(0, 0),
        bbox=BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.BOTTOMLEFT),
    )
    doc.add_table(
        data=ReadingOrderModel._table_data_from_table(table),
        prov=prov,
    )

    model = ReadingOrderModel(
        options=ReadingOrderOptions(recover_orphaned_table_text=True)
    )
    model._add_unmatched_table_text(table, doc)

    assert len(doc.texts) == 1
    assert doc.texts[0].text == "Signature John Doe"
    assert doc.body.children[0].cref == "#/tables/0"
    assert doc.body.children[1].cref == "#/groups/0"
    assert "Signature John Doe" in doc.export_to_markdown()


def test_recovery_coexists_with_picture_nested_in_table_cell():
    matched_cell = TableCell(
        text="cell",
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    absorbed = _child(2, (0, 0, 10, 10), "cell")
    orphaned = _child(3, (50, 50, 60, 60), "Signature John Doe")
    table = _table([matched_cell], [absorbed, orphaned])
    picture = FigureElement(
        label=DocItemLabel.PICTURE,
        id=4,
        page_no=1,
        cluster=Cluster(
            id=4,
            label=DocItemLabel.PICTURE,
            bbox=BoundingBox(l=1, t=1, r=9, b=9),
        ),
    )
    conv_res = ConversionResult(
        input=InputDocument.model_construct(
            file=PurePath("table-picture.pdf"),
            document_hash="0" * 64,
            format=InputFormat.PDF,
        ),
        pages=[Page(page_no=1, size=Size(width=100, height=100))],
        assembled=AssembledUnit(elements=[table, picture]),
    )
    model = ReadingOrderModel(
        options=ReadingOrderOptions(recover_orphaned_table_text=True)
    )

    doc = model._readingorder_elements_to_docling_doc(
        conv_res,
        model._assembled_to_readingorder_elements(conv_res),
        el_to_captions_mapping={},
        el_to_footnotes_mapping={},
        el_merges_mapping={},
    )

    body_items = [child.resolve(doc) for child in doc.body.children]
    assert len(body_items) == 2
    assert isinstance(body_items[0], TableItem)
    assert isinstance(body_items[1], GroupItem)
    recovered_items = [child.resolve(doc) for child in body_items[1].children]
    assert len(recovered_items) == 1
    assert isinstance(recovered_items[0], TextItem)
    assert recovered_items[0].text == "Signature John Doe"

    rich_cell = body_items[0].data.table_cells[0]
    assert isinstance(rich_cell, RichTableCell)
    rich_group = rich_cell.ref.resolve(doc)
    assert isinstance(rich_group, GroupItem)
    rich_items = [child.resolve(doc) for child in rich_group.children]
    assert len(rich_items) == 2
    assert isinstance(rich_items[0], TextItem)
    assert rich_items[0].text == "cell"
    assert isinstance(rich_items[1], PictureItem)
    doc.validate_document()
