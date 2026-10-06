# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A runaway TableFormer structure must not take the table's text with it.

TableFormer's structure decoder is autoregressive with a fixed step budget. On
some crops of dense tables it never emits a row break and repeats a single cell
token until the budget is exhausted (issue #3002). The cell matcher then
produces a one-row grid, nearly every text cell is dropped from the table, and
the reading-order stage discards the table's nested text clusters because the
table "has a grid". Both table stages now recognise such a structure, discard
it, and keep the text through the empty-table path.

These tests drive the stages on a synthetic page and stand in only for the
model inference call. The end-to-end check against a real conversion lives in
``test_table_structure_degenerate_pipeline.py``.
"""

import logging
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T  # type: ignore[import-untyped]
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument, _DummyBackend
from docling.models.base_table_model import (
    degenerate_structure_reason,
    keep_table_text_as_child,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.stages.table_structure.table_structure_model_v2 import (
    TableStructureModelV2,
)

PAGE_WIDTH, PAGE_HEIGHT = 600, 800
TABLE_BBOX = BoundingBox(l=100, t=100, r=400, b=300)
V2_MAX_LENGTH = TableStructureModelV2.MAX_LENGTH


def _text_cell(index: int, bbox: BoundingBox, text: str) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=False,
    )


# A 2x2 table whose text lines sit well inside its four cells.
TABLE_CELLS = [
    _text_cell(0, BoundingBox(l=110, t=110, r=240, b=140), "Model"),
    _text_cell(1, BoundingBox(l=260, t=110, r=390, b=140), "Score"),
    _text_cell(2, BoundingBox(l=110, t=210, r=240, b=240), "OTSL"),
    _text_cell(3, BoundingBox(l=260, t=210, r=390, b=240), "0.93"),
]
TABLE_TEXT = [cell.text for cell in TABLE_CELLS]
# Whitespace inside the table, text below it, and a line that only overlaps
# the table box by a quarter of its own area.
BLANK_CELL = _text_cell(4, BoundingBox(l=120, t=160, r=200, b=180), "   ")
CAPTION_CELL = _text_cell(5, BoundingBox(l=100, t=320, r=400, b=350), "Caption")
STRADDLING_CELL = _text_cell(6, BoundingBox(l=350, t=280, r=450, b=320), "Edge")

COMPLETE_SEQ = ["fcel", "fcel", "nl", "fcel", "fcel", "nl"]
# Normalised xyxy boxes TableFormerV2 would predict for the four cells.
GRID_BBOXES = [
    [0.02, 0.05, 0.48, 0.2],
    [0.52, 0.05, 0.98, 0.2],
    [0.02, 0.55, 0.48, 0.7],
    [0.52, 0.55, 0.98, 0.7],
]


def _table_cluster(
    cells: list[TextCell] | None = None,
    children: list[Cluster] | None = None,
) -> Cluster:
    return Cluster(
        id=3,
        label=DocItemLabel.TABLE,
        bbox=TABLE_BBOX,
        confidence=0.8,
        cells=cells or [],
        children=children or [],
    )


def _page(cells: list[TextCell], clusters: list[Cluster]) -> Page:
    full = BoundingBox(l=0, t=0, r=PAGE_WIDTH, b=PAGE_HEIGHT)
    geometry = PdfPageGeometry(
        angle=0,
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        rect=BoundingRectangle.from_bounding_box(full),
        art_bbox=full,
        bleed_bbox=full,
        crop_bbox=full,
        media_bbox=full,
        trim_bbox=full,
    )
    page = Page(page_no=1, size=Size(width=PAGE_WIDTH, height=PAGE_HEIGHT))
    page.parsed_page = SegmentedPdfPage(
        dimension=geometry,
        textline_cells=cells,
        char_cells=[],
        word_cells=[],
        has_chars=False,
        has_words=False,
        has_lines=True,
    )
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    return page


class _PageBackend:
    """The slice of PdfPageBackend the table stages use while predicting."""

    def __init__(self, segmented_page: SegmentedPdfPage | None):
        self._segmented_page = segmented_page

    def is_valid(self) -> bool:
        return True

    def get_segmented_page(self) -> SegmentedPdfPage | None:
        return self._segmented_page

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        return ""


def _stage_page(table: Cluster, scale: float) -> Page:
    """A page the table stages can run on without a PDF behind it."""
    page = _page([*TABLE_CELLS, CAPTION_CELL], [table])
    page._backend = _PageBackend(page.parsed_page)  # type: ignore[assignment]
    page._image_cache[scale] = Image.new(
        "RGB", (int(PAGE_WIDTH * scale), int(PAGE_HEIGHT * scale)), "white"
    )
    return page


@pytest.fixture
def conv_res(tmp_path: Path) -> ConversionResult:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    return ConversionResult(
        input=InputDocument(
            path_or_stream=pdf_path, format=InputFormat.PDF, backend=_DummyBackend
        )
    )


@pytest.mark.parametrize(
    ("otsl_seq", "expected"),
    [
        ([], None),
        (["fcel", "fcel", "nl", "fcel", "ecel", "nl"], None),
        # Long but complete: ends with a row break.
        ((["fcel"] * 33 + ["nl"]) * 30 + ["fcel"] * 2, "limit"),
        ((["fcel"] * 33 + ["nl"]) * 30 + ["fcel", "nl"], None),
        (["ecel"] * 1023, "row break"),
        (["lcel"] * 1023, "row break"),
    ],
)
def test_degenerate_structure_reason(otsl_seq, expected):
    reason = degenerate_structure_reason(otsl_seq, max_steps=1024)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


def test_keep_table_text_as_child_gathers_the_text_inside_the_table():
    table = _table_cluster()
    paragraph = Cluster(
        id=7,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=100, t=400, r=400, b=500),
        children=[
            Cluster(
                id=12,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=100, t=400, r=400, b=450),
            )
        ],
    )
    page = _page(
        [*TABLE_CELLS, BLANK_CELL, CAPTION_CELL, STRADDLING_CELL],
        [table, paragraph],
    )

    keep_table_text_as_child(table, page)

    (child,) = table.children
    assert child.label == DocItemLabel.TEXT
    # The id must not collide with any cluster or nested cluster on the page.
    assert child.id == 13
    assert child.bbox == TABLE_BBOX
    assert child.confidence == table.confidence
    assert [cell.text for cell in child.cells] == TABLE_TEXT


def test_keep_table_text_as_child_leaves_existing_nested_clusters_alone():
    nested = Cluster(
        id=4,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=110, t=110, r=240, b=140),
        cells=[TABLE_CELLS[0]],
    )
    table = _table_cluster(children=[nested])
    page = _page(TABLE_CELLS, [table])

    keep_table_text_as_child(table, page)

    assert table.children == [nested]


def test_keep_table_text_as_child_without_text_in_the_table():
    table = _table_cluster()
    page = _page([BLANK_CELL, CAPTION_CELL, STRADDLING_CELL], [table])

    keep_table_text_as_child(table, page)

    assert table.children == []


class _FakeTokenizer:
    """Decodes ids one token at a time, like the TableFormerV2 tokenizer."""

    _tokens = ["<start>", "<end>", "<pad>", "<fcel>", "<ecel>", "<nl>", "<lcel>"]

    def encode_tags(self, tags: list[str]) -> list[int]:
        return [self._tokens.index(f"<{tag}>") for tag in tags]

    def decode(self, ids: list[int]) -> str:
        return self._tokens[ids[0]]


class _FakeTableFormer:
    """Stand in for TableFormerV2.generate with a fixed structure prediction."""

    def __init__(self, otsl_seq: list[str], bboxes: list[list[float]] | None):
        self.otsl_seq = otsl_seq
        self.bboxes = bboxes

    def generate(self, image_tensor, tokenizer: _FakeTokenizer, max_length: int):
        ids = [0, *tokenizer.encode_tags(self.otsl_seq), 1]
        return {
            "generated_ids": torch.tensor([ids]),
            "predicted_bboxes": (
                torch.tensor([self.bboxes]) if self.bboxes is not None else None
            ),
        }


def _v2_model(
    otsl_seq: list[str], bboxes: list[list[float]] | None = None
) -> TableStructureModelV2:
    model = object.__new__(TableStructureModelV2)
    model.enabled = True
    model.do_cell_matching = True
    model.scale = 2.0
    model.device = "cpu"
    model.transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    model.tokenizer = _FakeTokenizer()
    model.model = _FakeTableFormer(otsl_seq, bboxes)  # type: ignore[assignment]
    return model


RUNAWAY_V2_SEQUENCES = [
    pytest.param(["fcel"] * (V2_MAX_LENGTH - 1), "no row break", id="no-row-break"),
    pytest.param(
        ["fcel", "fcel", "nl"] * ((V2_MAX_LENGTH - 2) // 3) + ["fcel"],
        "decode limit",
        id="cut-off",
    ),
]


@pytest.mark.parametrize(("otsl_seq", "reason"), RUNAWAY_V2_SEQUENCES)
def test_v2_predict_tables_discards_runaway_structure_and_keeps_text(
    conv_res, caplog, otsl_seq, reason
):
    table = _table_cluster(cells=TABLE_CELLS)
    model = _v2_model(otsl_seq)
    page = _stage_page(table, model.scale)

    with caplog.at_level(logging.WARNING, logger="docling"):
        (prediction,) = model.predict_tables(conv_res, [page])

    tbl = prediction.table_map[table.id]
    # The empty-table path: no grid, the raw prediction kept for inspection.
    assert (tbl.num_rows, tbl.num_cols, tbl.table_cells) == (0, 0, [])
    assert tbl.otsl_seq == otsl_seq
    (child,) = table.children
    assert [cell.text for cell in child.cells] == TABLE_TEXT
    assert any(reason in record.getMessage() for record in caplog.records)


def test_v2_predict_tables_keeps_a_complete_structure(conv_res, caplog):
    table = _table_cluster(cells=TABLE_CELLS)
    model = _v2_model(COMPLETE_SEQ, GRID_BBOXES)
    page = _stage_page(table, model.scale)

    with caplog.at_level(logging.WARNING, logger="docling"):
        (prediction,) = model.predict_tables(conv_res, [page])

    tbl = prediction.table_map[table.id]
    assert (tbl.num_rows, tbl.num_cols) == (2, 2)
    assert [cell.text for cell in tbl.table_cells] == TABLE_TEXT
    assert table.children == []
    assert not caplog.records


@pytest.mark.parametrize(
    ("otsl_seq", "bboxes", "expected_shape", "expected_text"),
    [
        pytest.param(["ecel"] * (V2_MAX_LENGTH - 1), None, (0, 0), [], id="runaway"),
        pytest.param(COMPLETE_SEQ, GRID_BBOXES, (2, 2), TABLE_TEXT, id="complete"),
    ],
)
def test_v2_image_prediction_applies_the_same_guard(
    caplog, otsl_seq, bboxes, expected_shape, expected_text
):
    table = _table_cluster(cells=TABLE_CELLS)
    model = _v2_model(otsl_seq, bboxes)
    crop = Image.new("RGB", (600, 400), "white")

    with caplog.at_level(logging.WARNING, logger="docling"):
        tbl = model._do_prediction_on_image_to_table(
            table_image=crop, table_cluster=table, page_no=1
        )

    assert (tbl.num_rows, tbl.num_cols) == expected_shape
    assert [cell.text for cell in tbl.table_cells] == expected_text
    assert bool(caplog.records) == (expected_shape == (0, 0))


class _FakePredictor:
    """Stand in for TFPredictor.multi_table_predict with a fixed prediction.

    ``cells`` are the text lines the cell matcher placed into the grid, laid
    out two per row. For the runaway structure observed in issue #3002 the
    one-row grid only has room for the first line; the rest is already gone.
    """

    def __init__(self, scale: float, rs_seq: list[str], shape: tuple[int, int], cells):
        self.scale = scale
        self.rs_seq = rs_seq
        self.shape = shape
        self.cells = cells

    def _response(self, index: int, cell: TextCell) -> dict:
        bbox = cell.rect.to_bounding_box().scaled(self.scale)
        row, col = divmod(index, 2)
        return {
            "bbox": {**bbox.model_dump(), "token": cell.text},
            "row_span": 1,
            "col_span": 1,
            "start_row_offset_idx": row,
            "end_row_offset_idx": row + 1,
            "start_col_offset_idx": col,
            "end_col_offset_idx": col + 1,
            "column_header": False,
            "row_header": False,
            "row_section": False,
        }

    def multi_table_predict(self, page_input, table_bboxes, do_matching=True):
        num_rows, num_cols = self.shape
        return [
            {
                "tf_responses": [
                    self._response(index, cell) for index, cell in enumerate(self.cells)
                ],
                "predict_details": {
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "prediction": {"rs_seq": self.rs_seq},
                },
            }
            for _ in table_bboxes
        ]


def _v1_model(rs_seq: list[str], shape: tuple[int, int], cells) -> TableStructureModel:
    model = object.__new__(TableStructureModel)
    model.enabled = True
    model.do_cell_matching = True
    model.scale = 2.0
    model.tm_config = {"predict": {"max_steps": 1024}}
    model.tf_predictor = _FakePredictor(model.scale, rs_seq, shape, cells)  # type: ignore[assignment]
    return model


def test_v1_predict_tables_discards_runaway_structure_and_keeps_text(conv_res, caplog):
    model = _v1_model(["ecel"] * 1023, (1, 6), TABLE_CELLS[:1])
    table = _table_cluster(cells=TABLE_CELLS)
    page = _stage_page(table, model.scale)

    with caplog.at_level(logging.WARNING, logger="docling"):
        (prediction,) = model.predict_tables(conv_res, [page])

    tbl = prediction.table_map[table.id]
    assert (tbl.num_rows, tbl.num_cols, tbl.table_cells) == (0, 0, [])
    (child,) = table.children
    assert [cell.text for cell in child.cells] == TABLE_TEXT
    assert any("no row break" in record.getMessage() for record in caplog.records)


def test_v1_predict_tables_keeps_a_complete_structure(conv_res, caplog):
    model = _v1_model(COMPLETE_SEQ, (2, 2), TABLE_CELLS)
    table = _table_cluster(cells=TABLE_CELLS)
    page = _stage_page(table, model.scale)

    with caplog.at_level(logging.WARNING, logger="docling"):
        (prediction,) = model.predict_tables(conv_res, [page])

    tbl = prediction.table_map[table.id]
    assert (tbl.num_rows, tbl.num_cols) == (2, 2)
    assert [cell.text for cell in tbl.table_cells] == TABLE_TEXT
    assert table.children == []
    assert not caplog.records
