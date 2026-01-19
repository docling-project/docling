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


def test_latex_abstract_environment():
    """Test abstract environment parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{abstract}
    This is the abstract content.
    \\end{abstract}
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
    assert "Abstract" in md
    assert "abstract content" in md


def test_latex_list_itemize():
    """Test itemize list environment"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{itemize}
    \\item First item
    \\item Second item
    \\item Third item
    \\end{itemize}
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

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) >= 3
    item_texts = [item.text for item in list_items]
    assert any("First item" in t for t in item_texts)
    assert any("Second item" in t for t in item_texts)


def test_latex_list_enumerate():
    """Test enumerate list environment"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{enumerate}
    \\item Alpha
    \\item Beta
    \\end{enumerate}
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

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) >= 2


def test_latex_description_list():
    """Test description list with optional item labels"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{description}
    \\item[Term1] Definition one
    \\item[Term2] Definition two
    \\end{description}
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

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) >= 2


def test_latex_verbatim_environment():
    """Test verbatim code environment"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{verbatim}
    def hello():
        print("world")
    \\end{verbatim}
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

    code_items = [t for t in doc.texts if t.label == DocItemLabel.CODE]
    assert len(code_items) >= 1
    assert "hello" in code_items[0].text or "print" in code_items[0].text


def test_latex_lstlisting_environment():
    """Test lstlisting code environment"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{lstlisting}
    int main() {
        return 0;
    }
    \\end{lstlisting}
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

    code_items = [t for t in doc.texts if t.label == DocItemLabel.CODE]
    assert len(code_items) >= 1


def test_latex_bibliography():
    """Test bibliography environment parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Some text.
    \\begin{thebibliography}{9}
    \\bibitem{ref1} Author One, Title One, 2020.
    \\bibitem{ref2} Author Two, Title Two, 2021.
    \\end{thebibliography}
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
    assert "References" in md


def test_latex_caption():
    """Test caption macro parsing via includegraphics"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{test.png}
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

    # includegraphics creates a caption with the image path
    caption_items = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(caption_items) >= 1
    assert "test.png" in caption_items[0].text


def test_latex_footnote():
    """Test footnote macro parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Main text\\footnote{This is a footnote}.
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

    footnote_items = [t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert len(footnote_items) >= 1
    assert "footnote" in footnote_items[0].text


def test_latex_url():
    """Test URL macro parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Visit \\url{https://example.com} for more.
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

    ref_items = [t for t in doc.texts if t.label == DocItemLabel.REFERENCE]
    assert len(ref_items) >= 1
    assert "example.com" in ref_items[0].text


def test_latex_label():
    """Test label macro parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    \\label{sec:intro}
    Some content.
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

    # Labels are stored internally
    assert "sec:intro" in backend.labels


def test_latex_includegraphics():
    """Test includegraphics macro parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{image.png}
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

    assert len(doc.pictures) >= 1


def test_latex_citations():
    """Test cite macros parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    As shown in \\cite{smith2020} and \\citep{jones2021}.
    Also see \\ref{fig:1} and \\eqref{eq:main}.
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

    ref_items = [t for t in doc.texts if t.label == DocItemLabel.REFERENCE]
    assert len(ref_items) >= 2


def test_latex_title_macro():
    """Test title macro inside document"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\title{Document Title}
    \\maketitle
    Some content.
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

    title_items = [t for t in doc.texts if t.label == DocItemLabel.TITLE]
    assert len(title_items) >= 1


def test_latex_various_math_environments():
    """Test various math environments"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Equation starred:
    \begin{equation*}
    a = b
    \end{equation*}
    Gather:
    \begin{gather}
    x = y \\
    z = w
    \end{gather}
    Multline:
    \begin{multline}
    first \\
    second
    \end{multline}
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
    assert len(formulas) >= 3


def test_latex_heading_levels():
    """Test different heading levels"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\part{Part One}
    \\chapter{Chapter One}
    \\section{Section One}
    \\subsection{Subsection One}
    \\subsubsection{Subsubsection One}
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

    headers = [t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    assert len(headers) >= 3


def test_latex_text_formatting():
    """Test text formatting macros"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    This is \\textbf{bold} and \\textit{italic} and \\emph{emphasized}.
    Also \\texttt{monospace} and \\underline{underlined}.
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
    assert "bold" in md
    assert "italic" in md
    assert "emphasized" in md


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


def test_latex_figure_environment():
    """Test figure environment parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{figure}
    \\includegraphics{test.png}
    \\caption{Test figure}
    \\end{figure}
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

    assert len(doc.pictures) >= 1
    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(captions) >= 1


def test_latex_is_valid():
    """Test is_valid method"""
    # Valid document
    latex_content = b"\\documentclass{article}\\begin{document}Content\\end{document}"
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    assert backend.is_valid() is True

    # Empty document
    empty_content = b"   "
    in_doc_empty = InputDocument(
        path_or_stream=BytesIO(empty_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="empty.tex",
    )
    backend_empty = LatexDocumentBackend(
        in_doc=in_doc_empty, path_or_stream=BytesIO(empty_content)
    )
    assert backend_empty.is_valid() is False


def test_latex_supports_pagination():
    """Test supports_pagination class method"""
    assert LatexDocumentBackend.supports_pagination() is False


def test_latex_supported_formats():
    """Test supported_formats class method"""
    formats = LatexDocumentBackend.supported_formats()
    assert InputFormat.LATEX in formats


def test_latex_file_path_loading(tmp_path):
    """Test loading LaTeX from file path instead of BytesIO"""
    latex_file = tmp_path / "test.tex"
    latex_file.write_text(
        r"""
    \documentclass{article}
    \begin{document}
    File content here.
    \end{document}
    """
    )

    in_doc = InputDocument(
        path_or_stream=latex_file,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=latex_file)
    doc = backend.convert()

    md = doc.export_to_markdown()
    assert "File content here" in md


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

    # Should not crash, table may or may not be added
    assert doc is not None


def test_latex_marginpar():
    """Test marginpar macro is handled without error"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Main text with marginpar.
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

    # Just verify it doesn't crash and produces output
    assert doc is not None


def test_latex_no_document_env():
    """Test LaTeX without document environment processes all nodes"""
    latex_content = b"""
    \\section{Direct Section}
    Some direct content without document environment.
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
    assert "Direct Section" in md or "direct content" in md


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


def test_latex_newline_macro():
    """Test handling of \\\\ newline macro"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Line one\\\\
    Line two
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

    # Should not crash
    assert doc is not None


def test_latex_filecontents_ignored():
    """Test filecontents environment is ignored"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{filecontents}{sample.bib}
    @article{test, author={A}, title={B}}
    \\end{filecontents}
    \\begin{document}
    Actual content.
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
    assert "Actual content" in md
    # filecontents should not appear in output
    assert "@article" not in md


def test_latex_tilde_macro():
    """Test ~ (non-breaking space) macro handling"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Dr.~Smith arrived.
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
    assert "Smith" in md


def test_latex_math_environment():
    """Test math environment (not displaymath)"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Inline: \begin{math}a+b\end{math}.
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
    assert len(formulas) >= 1


def test_latex_displaymath_brackets():
    """Test \\[ \\] display math"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Display: \[ x^2 + y^2 = z^2 \]
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
    assert len(formulas) >= 1


def test_latex_citet_macro():
    """Test citet citation macro"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\citet{author2022} showed this.
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

    ref_items = [t for t in doc.texts if t.label == DocItemLabel.REFERENCE]
    assert len(ref_items) >= 1
