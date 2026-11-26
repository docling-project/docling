# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest
import torch
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
)
from docling.utils.otsl import parse_otsl_output

pytestmark = pytest.mark.ml_vlm


def test_options_kind():
    opts = GraniteVisionTableStructureOptions()
    assert opts.kind == "granite_vision_table"


def test_parse_simple_table():
    """2x2 table with column headers."""
    text = "<ched>Name</ched><ched>Value</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"
    otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

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
    _otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

    assert num_rows == 2
    assert num_cols == 2
    assert len(cells) == 4
    empty = [
        c for c in cells if c.start_row_offset_idx == 1 and c.start_col_offset_idx == 1
    ]
    assert len(empty) == 1
    assert empty[0].text == ""


def test_parse_literal_less_than_in_cell_text():
    text = "<fcel>卖出<fcel><-10%<nl><fcel>A < B<fcel>kept<nl>"

    _, cells, num_rows, num_cols = parse_otsl_output(text)

    assert (num_rows, num_cols) == (2, 2)
    assert [cell.text for cell in cells] == ["卖出", "<-10%", "A < B", "kept"]


def test_parse_colspan():
    """lcel produces colspan=2 on the preceding fcel."""
    text = "<fcel>Merged</fcel><lcel><nl><fcel>A</fcel><fcel>B</fcel><nl>"
    _otsl_seq, cells, _num_rows, num_cols = parse_otsl_output(text)

    assert num_cols == 2
    merged = [
        c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0
    ]
    assert len(merged) == 1
    assert merged[0].col_span == 2
    assert merged[0].end_col_offset_idx == 2


def test_parse_rowspan():
    """ucel produces rowspan=2 on the preceding fcel above it."""
    text = "<fcel>Tall</fcel><fcel>A</fcel><nl><ucel><fcel>B</fcel><nl>"
    _otsl_seq, cells, _num_rows, _num_cols = parse_otsl_output(text)

    tall = [
        c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0
    ]
    assert len(tall) == 1
    assert tall[0].row_span == 2
    assert tall[0].end_row_offset_idx == 2


def test_parse_multiple_rowspan():
    """Multiple ucel produces rowspan=2 on the corresponding fcel above it."""
    text = (
        "<fcel>Regular</fcel><fcel>Merged</fcel><fcel>Merged</fcel><nl>"
        "<fcel>Regular</fcel><ucel><ucel><nl>"
    )
    _otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

    assert num_cols == 3
    assert num_rows == 2
    assert len(cells) == 4
    for cell in cells:
        if cell.text == "Merged":
            assert cell.row_span == 2
            assert cell.col_span == 1
            assert cell.start_row_offset_idx == 0
            assert cell.end_row_offset_idx == 2
            assert cell.start_col_offset_idx + 1 == cell.end_col_offset_idx
        else:
            assert cell.text == "Regular"
            assert cell.row_span == 1
            assert cell.col_span == 1
            assert cell.start_row_offset_idx + 1 == cell.end_row_offset_idx
            assert cell.start_col_offset_idx + 1 == cell.end_col_offset_idx


def test_parse_xcel():
    """xcel is merged with adjacent ucel and lcel."""
    text = (
        "<fcel>Regular</fcel><fcel>Regular</fcel><fcel>Regular</fcel><fcel>Regular</fcel><nl>"
        "<fcel>Regular</fcel><fcel>Merged</fcel><lcel><fcel>Regular</fcel><nl>"
        "<fcel>Regular</fcel><ucel><xcel><fcel>Regular</fcel><nl>"
        "<fcel>Regular</fcel><fcel>Regular</fcel><fcel>Regular</fcel><fcel>Regular</fcel><nl>"
    )
    _otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

    assert num_cols == 4
    assert num_rows == 4
    assert len(cells) == 13
    for cell in cells:
        if cell.text == "Merged":
            assert cell.row_span == 2
            assert cell.col_span == 2
            assert cell.start_col_offset_idx == 1
            assert cell.end_col_offset_idx == 3
            assert cell.start_row_offset_idx == 1
            assert cell.end_row_offset_idx == 3
        else:
            assert cell.text == "Regular"
            assert cell.row_span == 1
            assert cell.col_span == 1
            assert cell.start_row_offset_idx + 1 == cell.end_row_offset_idx
            assert cell.start_col_offset_idx + 1 == cell.end_col_offset_idx


def test_parse_row_header():
    """rhed token produces row_header=True."""
    text = "<rhed>Section</rhed><fcel>Data</fcel><nl>"
    _, cells, _, _ = parse_otsl_output(text)

    rhed_cells = [c for c in cells if c.row_header]
    assert len(rhed_cells) == 1
    assert rhed_cells[0].text == "Section"


def test_parse_no_bbox():
    """All cells must have bbox=None."""
    text = "<ched>X</ched><nl><fcel>Y</fcel><nl>"
    _, cells, _, _ = parse_otsl_output(text)
    assert all(c.bbox is None for c in cells)


def test_parse_empty_string():
    """Empty or whitespace-only string returns empty table."""
    otsl_seq, cells, num_rows, num_cols = parse_otsl_output("")
    assert otsl_seq == []
    assert cells == []
    assert num_rows == 0
    assert num_cols == 0


def test_get_options_type():
    assert (
        GraniteVisionTableStructureModel.get_options_type()
        is GraniteVisionTableStructureOptions
    )


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


def test_parse_xcel_2d_merge():
    """xcel produces both rowspan=2 and colspan=2 on the origin cell."""
    text = "<fcel>Big</fcel><lcel><nl><ucel><xcel><nl>"
    _, cells, num_rows, num_cols = parse_otsl_output(text)

    assert num_rows == 2
    assert num_cols == 2
    origin = [
        c for c in cells if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0
    ]
    assert len(origin) == 1
    assert origin[0].col_span == 2
    assert origin[0].row_span == 2
    assert origin[0].end_col_offset_idx == 2
    assert origin[0].end_row_offset_idx == 2


def test_parse_srow():
    """srow token produces row_section=True."""
    text = "<srow>Category</srow><fcel>Data</fcel><nl>"
    _, cells, _, _ = parse_otsl_output(text)

    srow_cells = [c for c in cells if c.row_section]
    assert len(srow_cells) == 1
    assert srow_cells[0].text == "Category"


def test_parse_ecel_self_closing():
    """<ecel/> self-closing form produces an empty cell."""
    text = "<fcel>A</fcel><ecel/><nl>"
    _, cells, _, _ = parse_otsl_output(text)

    empty = [c for c in cells if c.start_col_offset_idx == 1]
    assert len(empty) == 1
    assert empty[0].text == ""


def test_factory_registration():
    """GraniteVisionTableStructureModel must be discoverable via the table structure factory."""
    from docling.models.factories.table_factory import TableStructureFactory
    from docling.models.stages.table_structure.table_structure_model_granite_vision import (
        GraniteVisionTableStructureModel,
    )

    factory = TableStructureFactory()
    factory.load_from_plugins()
    registered = list(factory.classes.values())
    assert GraniteVisionTableStructureModel in registered


class _RecordingProcessor:
    """Stands in for the GV processor and records the images it receives."""

    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def apply_chat_template(self, conversation, **kwargs) -> str:
        return "<tables_otsl>"

    def __call__(self, *, text, images, **kwargs):
        self.images = images
        return _Inputs({"input_ids": torch.zeros((len(images), 1), dtype=torch.long)})

    def decode(self, ids, **kwargs) -> str:
        return "<fcel>a<nl>"


class _Inputs(dict):
    def to(self, device):
        return self


class _EchoModel:
    def generate(self, *, input_ids, **kwargs):
        return input_ids


def test_table_crop_uses_configured_scale():
    """Table crops are rendered at the option scale, not at a fixed 72 dpi."""
    buf = BytesIO()
    Image.new("RGB", (200, 100), "white").save(buf, format="PNG")
    buf.seek(0)
    in_doc = InputDocument(
        path_or_stream=buf,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename="page.png",
    )
    page = Page(page_no=1, size=Size(width=200, height=100))
    page._backend = in_doc._backend.load_page(0)
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TABLE,
                bbox=BoundingBox(
                    l=5, t=20, r=105, b=70, coord_origin=CoordOrigin.TOPLEFT
                ),
            )
        ]
    )

    model = GraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=GraniteVisionTableStructureOptions(scale=3.0),
        accelerator_options=AcceleratorOptions(),
    )
    processor = _RecordingProcessor()
    model.enabled = True
    model.device = "cpu"
    model._processor = processor
    model._model = _EchoModel()
    model._model_max_length = 16

    model.predict_tables(ConversionResult(input=in_doc), [page])

    # 100 x 50 pt table at scale 3
    assert [im.size for im in processor.images] == [(300, 150)]
