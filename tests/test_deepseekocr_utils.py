# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for the DeepSeek-OCR table markup parsing.

These operate on plain strings: no model, no network, no reference test data.
"""

from docling_core.types.doc import Size, TableData

from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown

PAGE_SIZE = Size(width=880, height=1200)


def parse_table(table_html: str) -> TableData:
    """Parse one annotated table and return the table data of the document."""
    source = f"table[[100,100,900,300]]\n{table_html}"
    doc = parse_deepseekocr_markdown(source, PAGE_SIZE, page_no=1)
    assert len(doc.tables) == 1

    return doc.tables[0].data


def test_rowspan_shifted_row_keeps_its_last_cell():
    """A rowspan shifts the next row to the right, and the shift keeps its cells."""
    data = parse_table(
        '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
        "<tr><td>C</td><td>D</td></tr></table>"
    )

    assert data.num_cols == 3
    assert [
        (cell.text, cell.start_row_offset_idx, cell.start_col_offset_idx)
        for cell in data.table_cells
    ] == [("A", 0, 0), ("B", 0, 1), ("C", 1, 1), ("D", 1, 2)]


def test_span_that_is_not_a_number_keeps_the_table():
    """A colspan that is not a number counts as one column, as in the HTML backend."""
    data = parse_table(
        '<table><tr><th colspan="2x">Header</th></tr>'
        "<tr><td>a</td><td>b</td></tr></table>"
    )

    assert data.num_cols == 2
    assert [cell.text for cell in data.table_cells] == ["Header", "a", "b"]
