# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


def test_latex_table_parsing():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{tabular}{cc}
    Header1 & Header2 \\\\
    Row1Col1 & Row1Col2 \\\\
    Row2Col1 & \\%Escaped
    \\end{tabular}
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 3
    assert table.data.num_cols == 2

    # Check content
    cells = [c.text.strip() for c in table.data.table_cells]
    assert "Header1" in cells
    assert "row1col1" not in cells  # Case sensitivity check (should preserve)
    assert "Row1Col1" in cells
    assert "%Escaped" in cells  # Should be unescaped or at least cleanly parsed


def test_latex_table_environment():
    """Test table environment (wrapper around tabular)"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{table}
    \\begin{tabular}{cc}
    A & B \\\\
    C & D
    \\end{tabular}
    \\caption{Sample table}
    \\end{table}
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) >= 1


def test_latex_empty_table():
    """Test table with no parseable content"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{tabular}{cc}
    \\end{tabular}
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()
    assert doc is not None


def test_latex_starred_table_and_figure():
    """Test starred table* and figure* environments"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{table*}
    \\begin{tabular}{c}
    Wide table
    \\end{tabular}
    \\end{table*}
    \\begin{figure*}
    \\includegraphics{wide.png}
    \\end{figure*}
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) >= 1
    assert len(doc.pictures) >= 1


def test_latex_table_wrappers_are_tables():
    """tabular*, tabularx and longtable are parsed as tables, columns intact."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular*}{\textwidth}{lr}
    Key & Value \\
    Left & Right \\
    \end{tabular*}
    \begin{tabularx}{\textwidth}{lX}
    Name & Description \\
    Alpha & First row \\
    \end{tabularx}
    \begin{longtable}[c]{cc}
    Item & Qty \\
    \endhead
    Widgets & 3 \\
    \end{longtable}
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) == 3
    # The table parser still appends an empty row for the whitespace after the
    # final row separator, for every environment; only the rows with content
    # are compared here.
    rows = [
        [
            [cell.text.strip() for cell in row]
            for row in table.data.grid
            if any(cell.text.strip() for cell in row)
        ]
        for table in doc.tables
    ]
    assert rows == [
        [["Key", "Value"], ["Left", "Right"]],
        [["Name", "Description"], ["Alpha", "First row"]],
        [["Item", "Qty"], ["Widgets", "3"]],
    ]


def test_latex_multicolumn_table():
    """Test \\multicolumn in a tabular environment produces correct column span."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{ccc}
    \multicolumn{2}{c}{Merged Header} & Right \\
    A & B & C \\
    \end{tabular}
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) >= 1
    table = doc.tables[0]

    # The table should have 2 rows and 3 columns ( hopefullyyy )
    assert table.data.num_rows >= 1
    assert table.data.num_cols >= 2
    cells = [c.text.strip() for c in table.data.table_cells]
    assert any("Merged Header" in c for c in cells)


def test_latex_multirow_table():
    """Test \\multirow in a tabular environment produces correct row span."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{cc}
    \multirow{2}{*}{Tall Cell} & Top \\
    & Bottom \\
    \end{tabular}
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) >= 1
    cells = [c.text.strip() for c in doc.tables[0].data.table_cells]
    assert any("Tall Cell" in c for c in cells)


def test_latex_table_formatting_in_cells():
    """Test that LaTeX formatting commands in multicolumn/multirow cells
    produce clean text, not raw LaTeX syntax (issue #3199)."""
    latex_content = rb"""
    \documentclass{article}
    \usepackage{multirow}
    \begin{document}
    \begin{tabular}{ccc}
    \multicolumn{2}{c}{\textbf{Bold Header}} & Plain \\
    \multicolumn{2}{c}{\textit{Italic Header}} & Other \\
    \multicolumn{2}{c}{\tiny Small Text} & More \\
    \multicolumn{2}{c}{\textbf{\textit{Both}}} & End \\
    \multirow{2}{*}{\textbf{Bold Cell}} & A & B \\
    & C & D \\
    \end{tabular}
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.tables) >= 1
    cells = [c.text.strip() for c in doc.tables[0].data.table_cells]

    # Formatting macros should be stripped, leaving only text content
    assert any("Bold Header" in c for c in cells), f"cells: {cells}"
    assert not any("\\textbf" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Italic Header" in c for c in cells), f"cells: {cells}"
    assert not any("\\textit" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Small Text" in c for c in cells), f"cells: {cells}"
    assert not any("\\tiny" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Both" in c for c in cells), f"cells: {cells}"
    assert any("Bold Cell" in c for c in cells), f"cells: {cells}"


def _convert_table(latex_content: bytes) -> "DoclingDocument":
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    return backend.convert()


def test_latex_multirow_args_in_real_document():
    r"""\multirow arguments must be parsed even when the tabular does not
    start at source position 0, and the consumed brace groups must not
    leak into cells.

    pylatexenc reports document-global node positions while
    latex_verbatim() is relative to the node; slicing the argument source
    with the global position silently missed the arguments whenever the
    document had a preamble, mangling cells into "2*A" and losing the row
    span. The brace groups re-yielded by the walker also used to become
    phantom cells and the trailing whitespace produced an empty row.
    """
    doc = _convert_table(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{|l|l|}
    \hline
    \multirow{2}{*}{A} & B \\
    C & D \\
    \hline
    \end{tabular}
    \end{document}
    """
    )

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 2
    assert table.data.num_cols == 2

    cells = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c
        for c in table.data.table_cells
    }
    assert cells[(0, 0)].text == "A"
    assert cells[(0, 1)].text == "B"
    assert cells[(1, 0)].text == "C"
    assert cells[(1, 1)].text == "D"
    # the row span of \multirow{2} is now actually applied
    assert cells[(0, 0)].end_row_offset_idx == 2


def test_latex_multicolumn_without_phantom_cells():
    """The & following a consumed \\multicolumn must not emit an empty
    phantom cell, and the table must not gain a trailing empty row."""
    doc = _convert_table(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{ccc}
    \multicolumn{2}{c}{Wide} & Right \\
    a & b & c \\
    \end{tabular}
    \end{document}
    """
    )

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 2
    assert table.data.num_cols == 3

    cells = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c
        for c in table.data.table_cells
    }
    assert cells[(0, 0)].text == "Wide"
    assert cells[(0, 0)].end_col_offset_idx == 2
    assert cells[(0, 2)].text == "Right"
    assert cells[(1, 0)].text == "a"
    assert cells[(1, 1)].text == "b"
    assert cells[(1, 2)].text == "c"
    # no phantom cell between the multicolumn and Right
    assert (0, 1) not in cells
