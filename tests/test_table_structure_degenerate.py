# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A runaway TableFormer structure must not take the table's text with it.

TableFormer's structure decoder is autoregressive with a 1024-step budget. On
some crops of dense tables it never emits a row break and repeats a single cell
token until the budget is exhausted (issue #3002). The cell matcher then
produces a one-row grid, nearly every text cell is dropped from the table, and
the reading-order stage discards the table's nested text clusters because the
table "has a grid". The table stage now recognises such a structure, discards
it, and keeps the text through the empty-table path.
"""

import logging
from pathlib import Path

import pytest
from docling_core.types.doc.document import TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_table_model import degenerate_structure_reason

PDF_PATH = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")


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


def _runaway_predict(self, page_input, table_bboxes, do_matching=True):
    """Stand in for TFPredictor.multi_table_predict with the observed failure."""
    return [
        {
            "tf_responses": [],
            "predict_details": {
                "num_rows": 1,
                "num_cols": 6,
                "prediction": {"rs_seq": ["ecel"] * 1023},
            },
        }
        for _ in table_bboxes
    ]


@pytest.mark.ml_pdf_model
def test_runaway_structure_keeps_table_text(monkeypatch, caplog):
    from docling_ibm_models.tableformer.data_management.tf_predictor import (
        TFPredictor,
    )

    monkeypatch.setattr(TFPredictor, "multi_table_predict", _runaway_predict)

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.accelerator_options.device = AcceleratorDevice.CPU
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    with caplog.at_level(logging.WARNING, logger="docling"):
        doc = converter.convert(PDF_PATH).document

    tables = [item for item, _ in doc.iterate_items() if isinstance(item, TableItem)]
    assert len(tables) == 1
    table = tables[0]
    # The discarded grid is replaced by the empty-table path: one rich cell
    # whose group holds the nested text.
    assert (table.data.num_rows, table.data.num_cols) == (1, 1)
    assert table.children, "the table's text clusters must be kept as children"

    text = doc.export_to_markdown()
    assert "OTSL" in text and "HTML" in text
    assert any("no row break" in rec.getMessage() for rec in caplog.records)
