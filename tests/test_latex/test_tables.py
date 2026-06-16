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
LATEX_DATA_DIR = Path("./tests/data/latex/")


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
