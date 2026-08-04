from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DocumentOrigin,
    ProvenanceItem,
    Size,
    TableCell,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster, Table
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _text_cell(text: str) -> TextCell:
    return TextCell(
        index=0,
        rect=BoundingRectangle(
            r_x0=0, r_y0=0, r_x1=1, r_y1=0, r_x2=1, r_y2=1, r_x3=0, r_y3=1
        ),
        text=text,
        orig=text,
        from_ocr=True,
    )


def _child(cid: int, bbox: tuple, text: str) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        cells=[_text_cell(text)],
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
