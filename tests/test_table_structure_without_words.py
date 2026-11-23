# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""TableFormer must not decode the structure of a table that has no words.

With cell matching, TableFormer makes the table cells only from the words that
it matches to the predicted structure. Thus a table with no words, for example
the empty ruled grid of a form, always gives an empty table. The structure
decode of such a table can continue to the step limit and take seconds on CPU.
"""

from pathlib import Path

import pytest
from docling_core.types.doc.document import TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model

HEADER = (b"Month", b"Revenue", b"Cost", b"Margin")
COLUMN_WIDTH, ROW_HEIGHT, LEFT = 115, 24, 72


def _ruled_grid(top: int, columns: int = 4, rows: int = 5) -> bytes:
    right = LEFT + columns * COLUMN_WIDTH
    bottom = top - rows * ROW_HEIGHT
    lines = [
        (LEFT, top - row * ROW_HEIGHT, right, top - row * ROW_HEIGHT)
        for row in range(rows + 1)
    ] + [
        (LEFT + col * COLUMN_WIDTH, top, LEFT + col * COLUMN_WIDTH, bottom)
        for col in range(columns + 1)
    ]
    return b"".join(b"%d %d m %d %d l S\n" % line for line in lines)


def _text(x: int, baseline: int, text: bytes) -> bytes:
    return b"BT /F1 11 Tf 1 0 0 1 %d %d Tm (%s) Tj ET\n" % (x, baseline, text)


def _build_pdf() -> bytes:
    """One page with two ruled 4x5 grids in Helvetica.

    The first grid has no text. The second grid has a header row.
    """
    content = (
        _text(LEFT, 720, b"Table 1. Income")
        + _ruled_grid(top=700)
        + _text(LEFT, 540, b"Table 2. Costs")
        + _ruled_grid(top=520)
        + b"".join(
            _text(LEFT + 8 + col * COLUMN_WIDTH, 520 - 16, word)
            for col, word in enumerate(HEADER)
        )
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    startxref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        startxref,
    )
    return bytes(out)


@pytest.mark.parametrize(
    ("do_cell_matching", "decoded_token_counts"),
    [
        # Only the grid with the header row has words to match.
        (True, [len(HEADER)]),
        # Without cell matching, the cells come from the predicted structure.
        (False, [len(HEADER), 0]),
    ],
)
def test_table_without_words_is_decoded_only_without_cell_matching(
    monkeypatch, tmp_path: Path, do_cell_matching, decoded_token_counts
):
    from docling_ibm_models.tableformer.data_management.tf_predictor import (
        TFPredictor,
    )

    decode = TFPredictor.multi_table_predict
    token_counts: list[int] = []

    def counted_decode(self, page_input, table_bboxes, do_matching=True):
        token_counts.append(len(page_input["tokens"]))
        return decode(self, page_input, table_bboxes, do_matching=do_matching)

    monkeypatch.setattr(TFPredictor, "multi_table_predict", counted_decode)

    pdf_path = tmp_path / "ruled_grids.pdf"
    pdf_path.write_bytes(_build_pdf())
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.accelerator_options.device = AcceleratorDevice.CPU
    options.table_structure_options = TableStructureOptions(
        do_cell_matching=do_cell_matching
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(pdf_path).document

    tables = [item for item, _ in doc.iterate_items() if isinstance(item, TableItem)]
    assert len(tables) == 2, "the layout model must find both grids"
    assert sorted(token_counts, reverse=True) == decoded_token_counts
