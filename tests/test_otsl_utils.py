# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.utils.otsl import parse_otsl_output


def test_closed_cell_does_not_swallow_preceding_open_cells():
    """An unclosed cell followed later by a closed one keeps every row and cell."""
    text = "<otsl><fcel>a<fcel>b<nl><fcel>c</fcel><fcel>d</fcel><nl></otsl>"
    otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

    assert otsl_seq == ["fcel", "fcel", "nl", "fcel", "fcel", "nl"]
    assert (num_rows, num_cols) == (2, 2)
    assert [
        (c.text, c.start_row_offset_idx, c.start_col_offset_idx) for c in cells
    ] == [("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1)]


def test_closed_cells_still_parse():
    text = "<ched>Name</ched><ched>Value</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"
    _otsl_seq, cells, num_rows, num_cols = parse_otsl_output(text)

    assert (num_rows, num_cols) == (2, 2)
    assert [c.text for c in cells] == ["Name", "Value", "Foo", "42"]
