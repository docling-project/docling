# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""End-to-end check that a runaway TableFormer structure keeps the table text.

The stage-level tests live in ``test_table_structure_degenerate.py``. This
module runs the full PDF pipeline on a real page and only replaces the
TableFormer inference call with the failure observed in issue #3002, so it
exercises the layout stage, the table stage's guard, the reading-order
empty-table path, and the Markdown export together.
"""

import logging
from pathlib import Path

import pytest
from docling_core.types.doc.document import TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model

PDF_PATH = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")


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
