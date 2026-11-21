# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model

# First two pages of the synthetic 400-row table from issue #5513.
PDF_PATH = Path("tests/data/table_structure/sources/tableformer_ruled.pdf")


@pytest.mark.parametrize("mode", [TableFormerMode.FAST, TableFormerMode.ACCURATE])
def test_tableformer_keeps_ruled_pdf_rows(mode: TableFormerMode) -> None:
    options = PdfPipelineOptions(
        do_ocr=False,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=2
        ),
        table_structure_options=TableStructureOptions(mode=mode),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(PDF_PATH, page_range=(1, 2)).document

    first, second = doc.tables
    assert first.data.grid[45][0].text == "Item 45"
    assert second.data.grid[45][0].text == "Item 93"
    assert [(table.data.num_rows, table.data.num_cols) for table in doc.tables] == [
        (48, 6),
        (48, 6),
    ]
    rows = [row for table in doc.tables for row in table.data.grid]
    assert [row[0].text for row in rows] == ["Col1"] + [
        f"Item {row}" for row in range(1, 96)
    ]
    if mode == TableFormerMode.ACCURATE:
        assert [[cell.text for cell in row] for row in rows] == [
            [f"Col{col}" for col in range(1, 7)],
            *[
                [f"Item {row}", *(f"R{row}C{col}" for col in range(2, 7))]
                for row in range(1, 96)
            ],
        ]
    markdown = doc.export_to_markdown()
    assert "| Item 45 " in markdown
    assert "| Item 46 " in markdown
