from unittest.mock import MagicMock

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
    _parse_otsl_output,
)


def test_options_kind():
    opts = GraniteVisionTableStructureOptions()
    assert opts.kind == "granite_vision_table"


def test_parse_simple_table():
    """2x2 table with column headers."""
    text = "<ched>Name</ched><ched>Value</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert otsl_seq == ["ched", "ched", "nl", "fcel", "fcel", "nl"]
    assert num_rows == 2
    assert num_cols == 2
    assert len(cells) == 4

    header_cells = [c for c in cells if c.column_header]
    assert len(header_cells) == 2
    assert header_cells[0].text == "Name"
    assert header_cells[1].text == "Value"

    data_cells = [c for c in cells if not c.column_header]
    assert data_cells[0].text == "Foo"
    assert data_cells[0].start_row_offset_idx == 1
    assert data_cells[0].start_col_offset_idx == 0
    assert data_cells[1].text == "42"
    assert data_cells[1].start_col_offset_idx == 1


def test_parse_empty_cell():
    """Empty cell produces empty text, still in grid."""
    text = "<ched>A</ched><ched>B</ched><nl><fcel>x</fcel><ecel></ecel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert num_rows == 2
    assert num_cols == 2
    assert len(cells) == 4
    empty = [c for c in cells if c.start_row_offset_idx == 1 and c.start_col_offset_idx == 1]
    assert len(empty) == 1
    assert empty[0].text == ""


def test_parse_colspan():
    """lcel produces colspan=2 on the preceding fcel."""
    text = "<fcel>Merged</fcel><lcel><nl><fcel>A</fcel><fcel>B</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    assert num_cols == 2
    merged = [c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0]
    assert len(merged) == 1
    assert merged[0].col_span == 2
    assert merged[0].end_col_offset_idx == 2


def test_parse_rowspan():
    """ucel produces rowspan=2 on the preceding fcel above it."""
    text = "<fcel>Tall</fcel><fcel>A</fcel><nl><ucel><fcel>B</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output(text)

    tall = [c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0]
    assert len(tall) == 1
    assert tall[0].row_span == 2
    assert tall[0].end_row_offset_idx == 2


def test_parse_row_header():
    """rhed token produces row_header=True."""
    text = "<rhed>Section</rhed><fcel>Data</fcel><nl>"
    _, cells, _, _ = _parse_otsl_output(text)

    rhed_cells = [c for c in cells if c.row_header]
    assert len(rhed_cells) == 1
    assert rhed_cells[0].text == "Section"


def test_parse_no_bbox():
    """All cells must have bbox=None."""
    text = "<ched>X</ched><nl><fcel>Y</fcel><nl>"
    _, cells, _, _ = _parse_otsl_output(text)
    assert all(c.bbox is None for c in cells)


def test_parse_empty_string():
    """Empty or whitespace-only string returns empty table."""
    otsl_seq, cells, num_rows, num_cols = _parse_otsl_output("")
    assert otsl_seq == []
    assert cells == []
    assert num_rows == 0
    assert num_cols == 0


def test_get_options_type():
    assert GraniteVisionTableStructureModel.get_options_type() is GraniteVisionTableStructureOptions


def test_model_disabled_skips_pages():
    """When enabled=False, predict_tables returns empty prediction without running inference."""
    from unittest.mock import MagicMock
    from docling_core.types.doc import DocItemLabel

    model = GraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(),
    )

    # Give the page a cluster labelled TABLE so the cluster filter passes —
    # the skip must be triggered by enabled=False, not by an empty cluster list.
    cluster = MagicMock()
    cluster.label = DocItemLabel.TABLE

    page = MagicMock()
    page._backend.is_valid.return_value = True
    page.predictions.layout.clusters = [cluster]
    page.predictions.tablestructure = None
    page.size = MagicMock()

    results = model.predict_tables(MagicMock(), [page])
    assert len(results) == 1
    assert results[0].table_map == {}


def test_model_invalid_backend_returns_empty_prediction():
    """Pages with invalid backends return an empty TableStructurePrediction."""
    from unittest.mock import MagicMock

    model = GraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(),
    )

    page = MagicMock()
    page._backend.is_valid.return_value = False
    page.predictions.tablestructure = None

    results = model.predict_tables(MagicMock(), [page])
    assert len(results) == 1
    assert results[0].table_map == {}
