from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    TableStructureV2Options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.table_structure.table_topology import (
    validate_table_topology,
)

pytestmark = pytest.mark.ml_pdf_model

SOURCE = (
    Path(__file__).parent
    / "data"
    / "pdf"
    / ("merged_span_detectable_workflow_test.pdf")
)


def _iter_pages(conv):
    pages = getattr(conv, "pages", {})
    if isinstance(pages, dict):
        return pages.items()

    return enumerate(pages, start=1)


def _cell_text(cell) -> str:
    text = getattr(cell, "text", "") or getattr(cell, "token", "") or ""
    return " ".join(str(text).split())


def _find_cell(table, text: str):
    matches = [cell for cell in table.table_cells if _cell_text(cell) == text]
    assert len(matches) == 1, (
        f"Expected exactly one cell with text={text!r}, found {len(matches)}"
    )
    return matches[0]


def _assert_cell_rect(
    table,
    text: str,
    *,
    rows: tuple[int, int],
    cols: tuple[int, int],
):
    cell = _find_cell(table, text)
    assert (cell.start_row_offset_idx, cell.end_row_offset_idx) == rows
    assert (cell.start_col_offset_idx, cell.end_col_offset_idx) == cols
    assert cell.row_span == rows[1] - rows[0]
    assert cell.col_span == cols[1] - cols[0]


def _convert_first_table(version: str, do_cell_matching: bool):
    pipeline_options = PdfPipelineOptions()

    if version == "v2":
        pipeline_options.table_structure_options = TableStructureV2Options()
    else:
        pipeline_options.table_structure_options = TableStructureOptions()
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.table_structure_options.do_cell_matching = do_cell_matching
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_page_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv = converter.convert(SOURCE)

    for _, page in _iter_pages(conv):
        predictions = getattr(page, "predictions", None)
        table_structure = (
            getattr(predictions, "tablestructure", None) if predictions else None
        )
        if not table_structure:
            continue

        for table in table_structure.table_map.values():
            return table

    raise AssertionError("No table-structure prediction found in regression PDF.")


@pytest.mark.parametrize(
    ("version", "do_cell_matching"),
    [
        ("v1", True),
        ("v1", False),
        ("v2", True),
        ("v2", False),
    ],
)
def test_3698_table_structure_reconciliation_matrix(version, do_cell_matching):
    table = _convert_first_table(version, do_cell_matching)

    assert table.num_rows == 10
    assert table.num_cols == 5

    diagnostics = validate_table_topology(
        table.table_cells,
        table.num_rows,
        table.num_cols,
    )
    assert diagnostics.valid

    texts = {_cell_text(cell) for cell in table.table_cells}

    assert "Owner Window" not in texts
    assert "Mira 09:00 - 10:30" not in texts
    assert "Owner Mira" not in texts
    assert "Window 09:00 - 10:30" not in texts

    _assert_cell_rect(table, "Owner", rows=(2, 3), cols=(1, 2))
    _assert_cell_rect(table, "Mira", rows=(3, 4), cols=(1, 2))
    _assert_cell_rect(table, "Window", rows=(2, 3), cols=(2, 3))
    _assert_cell_rect(table, "09:00 - 10:30", rows=(3, 4), cols=(2, 3))

    assert _find_cell(table, "Owner").column_header
    assert _find_cell(table, "Window").column_header
    assert _find_cell(table, "Task").column_header
    assert _find_cell(table, "Risk").column_header

    _assert_cell_rect(table, "South", rows=(4, 7), cols=(0, 1))
    _assert_cell_rect(table, "API contract", rows=(4, 6), cols=(3, 4))
    _assert_cell_rect(table, "West", rows=(7, 10), cols=(0, 1))
    _assert_cell_rect(table, "Data migration", rows=(7, 9), cols=(3, 4))

    _assert_cell_rect(table, "11:00 - 13:00", rows=(8, 9), cols=(2, 3))
