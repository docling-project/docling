from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def test_latex_basic_conversion():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    Hello World.
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

    assert len(doc.texts) > 0
    # Check structure
    headers = [t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    paragraphs = [t for t in doc.texts if t.label != DocItemLabel.SECTION_HEADER]

    assert len(headers) == 1
    assert headers[0].text == "Introduction"
    assert "Hello World" in paragraphs[0].text


def test_latex_preamble_filter():
    latex_content = b"""
    \\documentclass{article}
    \\usepackage{test}
    \\title{Ignored Title}
    \\begin{document}
    Real Content
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

    # Title in preamble should be ignored by the backend (unless we explicitly parse it, which current logic doesn't for simplistic Document extraction)
    # The current logic filters for 'document' environment, so "Real Content" should be there, "Ignored Title" should not (if inside structure but outside document env)

    full_text = doc.export_to_markdown()
    assert "Real Content" in full_text
    assert "Ignored Title" not in full_text
    assert "usepackage" not in full_text


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


def test_latex_math_parsing():
    # Test align environment (starred and unstarred) and inline/display math
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Inline math: $E=mc^2$.
    Display math:
    $$
    x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
    $$
    Aligned equations:
    \begin{align}
    a &= b + c \\
    d &= e + f
    \end{align}
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

    formulas = [t for t in doc.texts if t.label == DocItemLabel.FORMULA]
    assert len(formulas) >= 3  # Inline, Display, Align

    md = doc.export_to_markdown()
    # Check delimiters
    assert "$E=mc^2$" in md or r"\( E=mc^2 \)" in md or "E=mc^2" in md
    assert r"\frac" in md
    assert r"\begin{align}" in md  # Should preserve align tag for proper rendering


def test_latex_escaped_chars():
    # Test correct handling of escaped chars to ensure text isn't split
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    value is 23\\% which is high.
    Costs \\$100.
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

    text_items = [
        t.text
        for t in doc.texts
        if t.label == DocItemLabel.TEXT or t.label == DocItemLabel.PARAGRAPH
    ]
    full_text = " ".join(text_items)

    # "23%" should be together, not "23" and "%" split
    assert "23%" in full_text or "23\\%" in full_text
    # Should not have loose "%" newline
    assert "which is high" in full_text
    assert "$100" in full_text or "\\$100" in full_text


def test_latex_unknown_macro_fallback():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\unknownmacro{Known Content}
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

    md = doc.export_to_markdown()
    assert "Known Content" in md
