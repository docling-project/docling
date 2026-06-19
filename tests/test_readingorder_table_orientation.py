from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.document import Orientation

from docling.datamodel.base_models import Cluster, Table
from docling.models.stages.reading_order.readingorder_model import ReadingOrderModel


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
