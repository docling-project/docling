from docling.datamodel.base_models import DocItemLabel

from ._utils import make_backend


def test_latex_table_parsing():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{tabular}{cc}
    Header1 & Header2 \\\\
    Row1Col1 & Row1Col2 \\\\
    Row2Col1 & \\%Escaped
    \\end{tabular}
    \\end{document}
    """
    ).convert()

    table = doc.tables[0]
    cells = [c.text.strip() for c in table.data.table_cells]
    assert len(doc.tables) == 1
    assert table.data.num_rows == 3
    assert table.data.num_cols == 2
    assert "Header1" in cells
    assert "row1col1" not in cells
    assert "Row1Col1" in cells
    assert "%Escaped" in cells


def test_latex_table_environment():
    doc = make_backend(
        b"""
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
    ).convert()
    assert len(doc.tables) >= 1


def test_latex_empty_table():
    assert (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{tabular}{cc}
    \\end{tabular}
    \\end{document}
    """
        ).convert()
        is not None
    )


def test_latex_starred_table_and_figure():
    doc = make_backend(
        b"""
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
    ).convert()

    assert len(doc.tables) >= 1
    assert len(doc.pictures) >= 1


def test_latex_multicolumn_table():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{ccc}
    \multicolumn{2}{c}{Merged Header} & Right \\
    A & B & C \\
    \end{tabular}
    \end{document}
    """
    ).convert()

    table = doc.tables[0]
    cells = [c.text.strip() for c in table.data.table_cells]
    assert len(doc.tables) >= 1
    assert table.data.num_rows >= 1
    assert table.data.num_cols >= 2
    assert any("Merged Header" in c for c in cells)


def test_latex_multirow_table():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{tabular}{cc}
    \multirow{2}{*}{Tall Cell} & Top \\
    & Bottom \\
    \end{tabular}
    \end{document}
    """
    ).convert()

    cells = [c.text.strip() for c in doc.tables[0].data.table_cells]
    assert len(doc.tables) >= 1
    assert any("Tall Cell" in c for c in cells)


def test_latex_table_formatting_in_cells():
    doc = make_backend(
        rb"""
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
    ).convert()

    cells = [c.text.strip() for c in doc.tables[0].data.table_cells]
    assert len(doc.tables) >= 1
    assert any("Bold Header" in c for c in cells), f"cells: {cells}"
    assert not any("\\textbf" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Italic Header" in c for c in cells), f"cells: {cells}"
    assert not any("\\textit" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Small Text" in c for c in cells), f"cells: {cells}"
    assert not any("\\tiny" in c for c in cells), f"raw LaTeX in cells: {cells}"
    assert any("Both" in c for c in cells), f"cells: {cells}"
    assert any("Bold Cell" in c for c in cells), f"cells: {cells}"
