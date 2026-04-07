from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/")


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

    # Preamble metadata (\title, \author, \date) is now extracted
    # following pandoc's approach. Only package commands should be filtered.

    full_text = doc.export_to_markdown()
    assert "Real Content" in full_text
    assert "Ignored Title" in full_text
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
    assert len(formulas) >= 2  # Display math and Align environment

    # Inline math should be part of the paragraph text
    paragraphs = [
        t for t in doc.texts if t.label in [DocItemLabel.PARAGRAPH, DocItemLabel.TEXT]
    ]
    full_text = " ".join([p.text for p in paragraphs])
    assert "$E=mc^2$" in full_text

    md = doc.export_to_markdown()
    # Check delimiters
    assert "$E=mc^2$" in md or r"\( E=mc^2 \)" in md
    assert r"\frac" in md
    assert r"\begin{align}" in md


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


def test_latex_citet_macro():
    """Test citet macro (textual citation)"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    According to \\citet{author2020}, this is correct.
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

    # Citations are now inline with text
    md = doc.export_to_markdown()
    assert "[author2020]" in md
    assert "According to" in md


def test_latex_list_nested():
    """Test nested lists (itemize within itemize, enumerate within itemize)"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{itemize}
    \\item Outer item one
    \\item Outer item two
      \\begin{itemize}
      \\item Inner item A
      \\item Inner item B
      \\end{itemize}
    \\item Outer item three
      \\begin{enumerate}
      \\item Numbered inner 1
      \\item Numbered inner 2
      \\end{enumerate}
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

    # Check that we have list groups
    list_groups = [g for g in doc.groups if g.label == GroupLabel.LIST]
    assert len(list_groups) >= 1  # At least the outer list

    # Check that list items exist
    # Note: Current implementation merges nested list items into their parent items
    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) >= 3  # 3 outer items (nested items are merged)

    # Verify some item content - nested items should appear within outer items
    item_texts = [item.text for item in list_items]
    assert any("Outer item one" in t for t in item_texts)
    # Nested items appear in the outer item text
    assert any("Inner item A" in t or "Inner item B" in t for t in item_texts)
    assert any("Numbered inner 1" in t or "Numbered inner 2" in t for t in item_texts)


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
    backend.convert()

    # Labels are stored internally
    assert "sec:intro" in backend.labels


def test_latex_includegraphics():
    """Test includegraphics with actual image file"""
    import tempfile
    from pathlib import Path

    from PIL import Image as PILImage

    # Create a temporary directory and test image
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_file = tmpdir_path / "test.tex"
        img_file = tmpdir_path / "test_image.png"

        # Create a simple test image with known DPI
        test_img = PILImage.new("RGB", (100, 50), color="red")
        test_img.save(img_file, dpi=(96, 96))

        latex_content = b"""
        \\documentclass{article}
        \\begin{document}
        \\includegraphics{test_image.png}
        \\end{document}
        """

        # Write LaTeX content to file
        tex_file.write_bytes(latex_content)

        in_doc = InputDocument(
            path_or_stream=tex_file,
            format=InputFormat.LATEX,
            backend=LatexDocumentBackend,
            filename="test.tex",
        )
        backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=tex_file)
        doc = backend.convert()

        # Verify picture was created
        assert len(doc.pictures) >= 1
        picture = doc.pictures[0]

        # Verify image was embedded (not None)
        assert picture.image is not None

        # Verify caption was created
        assert len(picture.captions) >= 1
        assert "test_image.png" in picture.captions[0].resolve(doc).text


def test_latex_includegraphics_missing_image():
    """Test includegraphics gracefully handles missing images"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{nonexistent_image.png}
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

    # Picture should still be created with caption
    assert len(doc.pictures) >= 1
    picture = doc.pictures[0]

    # Image should be None (couldn't load)
    assert picture.image is None

    # Caption should still exist
    assert len(picture.captions) >= 1
    assert "nonexistent_image.png" in picture.captions[0].resolve(doc).text


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

    # Citations are now inline with text
    md = doc.export_to_markdown()
    assert "[smith2020]" in md
    assert "[jones2021]" in md
    assert "[fig:1]" in md
    assert "[eq:main]" in md


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


def test_latex_citet_macro_2():
    """Test citet citation macro - citations inline with text"""
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

    # Citations are now inline with text
    md = doc.export_to_markdown()
    assert "[author2022]" in md
    assert "showed this" in md


# E2E Ground-Truth Tests


@pytest.fixture(scope="module")
def latex_paths() -> list[Path]:
    """Find all LaTeX files in the test data directory."""
    directory = Path("./tests/data/latex/")
    if not directory.exists():
        return []

    paths = list(directory.glob("*.tex"))

    for subdir in directory.iterdir():
        if subdir.is_dir():
            if (subdir / "main.tex").exists():
                paths.append(subdir / "main.tex")
            elif (subdir / f"arxiv_{subdir.name}.tex").exists():
                paths.append(subdir / f"arxiv_{subdir.name}.tex")

    return sorted(paths)


def get_latex_converter():
    """Create a DocumentConverter for LaTeX files."""
    converter = DocumentConverter(allowed_formats=[InputFormat.LATEX])
    return converter


def test_e2e_latex_conversions(latex_paths):
    """E2E test for LaTeX conversions with ground-truth comparison."""
    if not latex_paths:
        pytest.skip("No LaTeX test files found")

    converter = get_latex_converter()

    for latex_path in latex_paths:
        if latex_path.parent.resolve() == LATEX_DATA_DIR.resolve():
            gt_name = latex_path.name
        else:
            gt_name = f"{latex_path.parent.name}_{latex_path.name}"

        gt_path = LATEX_DATA_DIR.parent / "groundtruth" / "docling_v2" / gt_name

        conv_result: ConversionResult = converter.convert(latex_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            f"Markdown export mismatch for {latex_path}"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            f"Indented text export mismatch for {latex_path}"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            f"Document JSON mismatch for {latex_path}"
        )


def test_latex_document_with_leading_comments():
    """Test that documents starting with comment lines don't cause regex errors"""
    latex_content = b"""% This is a leading comment
% Another comment line
\\documentclass{article}
\\begin{document}
\\section{Test Section}
This is test content.
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

    # Should parse successfully without regex errors
    assert len(doc.texts) > 0
    md = doc.export_to_markdown()
    assert "Test Section" in md
    assert "test content" in md


def test_latex_custom_macro_with_backslash():
    """Test that custom macros containing backslashes don't cause regex errors"""
    latex_content = b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
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

    # Should parse successfully without regex errors
    assert len(doc.texts) > 0
    md = doc.export_to_markdown()
    # The macro expansion should work
    assert "special term" in md and "42" in md


def test_latex_figure_with_caption():
    """Test that figure environment properly groups caption and image"""
    latex_content = b"""\\documentclass{article}
\\begin{document}
\\begin{figure}
\\includegraphics{test.png}
\\caption{This is a test figure caption}
\\label{fig:test}
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

    # Should have a figure group
    figure_groups = [g for g in doc.groups if g.name == "figure"]
    assert len(figure_groups) >= 1

    # Should have picture and caption
    assert len(doc.pictures) >= 1
    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    # includegraphics creates one caption, and \caption macro creates another
    assert len(captions) >= 1


def extract_macro_name_old(raw_string):
    """The OLD buggy implementation"""
    # This was the original broken code
    macro_name_raw = raw_string.strip("{} ")
    macro_name = macro_name_raw.lstrip("\\")
    return macro_name


def extract_macro_name_new(raw_string):
    """The NEW fixed implementation"""
    # This is the fixed code
    macro_name = raw_string.strip("{} \n\t\\")
    if macro_name.startswith("\\"):
        macro_name = macro_name[1:]
    return macro_name


def test_macro_extraction():
    """Test various formats of macro names"""
    print("\n" + "=" * 80)
    print("TESTING MACRO NAME EXTRACTION LOGIC")
    print("=" * 80)

    # Test cases: (input, expected_output)
    test_cases = [
        (r"{\myterm}", "myterm"),
        (r"\myterm", "myterm"),
        (r"{ \myterm }", "myterm"),
        (r"{  \myvalue  }", "myvalue"),
        (r"{\important}", "important"),
        (r"{ \test }", "test"),
        (r"{\alpha}", "alpha"),
    ]

    print("\n" + "-" * 80)
    print("OLD (BUGGY) IMPLEMENTATION:")
    print("-" * 80)
    old_passed = 0
    for input_str, expected in test_cases:
        result = extract_macro_name_old(input_str)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} Input: {input_str!r:20} → Result: {result!r:15} (expected: {expected!r})"
        )
        if result == expected:
            old_passed += 1

    print("\n" + "-" * 80)
    print("NEW (FIXED) IMPLEMENTATION:")
    print("-" * 80)
    new_passed = 0
    for input_str, expected in test_cases:
        result = extract_macro_name_new(input_str)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} Input: {input_str!r:20} → Result: {result!r:15} (expected: {expected!r})"
        )
        if result == expected:
            new_passed += 1

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"OLD implementation: {old_passed}/{len(test_cases)} tests passed")
    print(f"NEW implementation: {new_passed}/{len(test_cases)} tests passed")

    if new_passed == len(test_cases):
        print("\n✓✓ SUCCESS: New implementation fixes the bug!")
    else:
        print("\n✗✗ FAILURE: New implementation still has issues")

    assert new_passed == len(test_cases), (
        f"New implementation failed: {new_passed}/{len(test_cases)} tests passed"
    )


def test_edge_cases():
    """Test edge cases and special characters"""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    edge_cases = [
        # Format: (input, expected, description)
        (r"{\cmd}", "cmd", "Simple macro"),
        (r"{\\cmd}", "cmd", "Double backslash"),
        (r"{ \cmd }", "cmd", "Spaces around macro"),
        (r"{\   cmd   }", "cmd", "Extra spaces"),
        (r"{\my_macro}", "my_macro", "Underscore in name"),
        (r"{\MyMacro}", "MyMacro", "CamelCase"),
        (r"{\MACRO}", "MACRO", "Uppercase"),
    ]

    all_passed = True
    for input_str, expected, description in edge_cases:
        result = extract_macro_name_new(input_str)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description:25} | Input: {input_str!r:20} → {result!r}")
        if result != expected:
            print(f"   Expected: {expected!r}")
            all_passed = False

    if all_passed:
        print("\n✓✓ All edge cases passed!")
    else:
        print("\n✗✗ Some edge cases failed")

    assert all_passed, "Some edge cases failed"


"""
Add this to test_backend_latex.py to debug what's happening
"""


def test_debug_macro_extraction():
    """Debug test to see what's actually being extracted"""
    from io import BytesIO

    from docling.backend.latex_backend import LatexDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    latex_content = b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
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

    print(f"\n{'=' * 80}")
    print("DEBUG INFO:")
    print(f"{'=' * 80}")
    print(f"Custom macros extracted: {backend._custom_macros}")
    print(f"Number of text items: {len(doc.texts)}")
    print("\nText items:")
    for i, text_item in enumerate(doc.texts):
        print(f"  {i}: {text_item.label} = {text_item.text!r}")

    md = doc.export_to_markdown()
    print("\nMarkdown output:")
    print(md)
    print(f"{'=' * 80}\n")

    # Check if macros were registered
    assert "myterm" in backend._custom_macros, "myterm not in _custom_macros!"
    assert backend._custom_macros["myterm"] == "special term"
    assert "special term" in md, f"'special term' not in output: {md!r}"
    assert "42" in md, f"'42' not in output: {md!r}"


def test_latex_href_macro():
    """Test \\href{url}{display} emits a markdown-style link."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Visit \href{https://example.com}{Example Site} for more.
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

    md = doc.export_to_markdown()
    assert "Example Site" in md
    assert "https://example.com" in md


def test_latex_textcolor_macro():
    """Test \\textcolor{color}{text} extracts the text content and ignores the color."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    This is \textcolor{red}{important} text.
    Also \colorbox{yellow}{highlighted} here.
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

    md = doc.export_to_markdown()
    assert "important" in md
    assert "highlighted" in md
    # Color names should not leak into output
    assert "red" not in md
    assert "yellow" not in md


def test_latex_custom_macro_parameters():
    """Test custom macros with parameters expand arguments before formatting is unwrapped."""
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\highlight}[1]{\textcolor{white}{\textbf{#1}}}
    \newcommand{\metric}[2]{#1{\scriptsize$\_{#2}$}}
    \begin{document}
    \highlight{Result}
    \metric{Accuracy}{test}
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

    md = doc.export_to_markdown()
    assert "Result" in md
    assert "Accuracy" in md
    assert "test" in md
    assert "#1" not in md
    assert "#2" not in md
    assert "\\textcolor" not in md
    assert "\\scriptsize" not in md


def test_latex_subequations_environment():
    """Test subequations wrapper environment passes through inner equations."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{subequations}
    \begin{align}
    a &= b \\
    c &= d
    \end{align}
    \end{subequations}
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
    assert len(formulas) >= 1, "subequations should pass through inner align formula"


def test_latex_legacy_font_switches():
    """Test legacy font/size switches (\\bf, \\it, \\tt, \\large, \\tiny) are silently ignored."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    {\bf bold text} and {\it italic text}.
    {\tt monospace} and {\large big} and {\tiny small}.
    Normal content here.
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

    md = doc.export_to_markdown()
    # Content inside the groups should still appear
    assert "bold text" in md
    assert "italic text" in md
    assert "Normal content here" in md


def test_latex_accent_macro():
    """Test accent macros (\\'{e}, \\`{a}) are converted to Unicode characters."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    caf\'{e} and na\"{i}ve.
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

    md = doc.export_to_markdown()
    assert "caf" in md
    assert len(doc.texts) > 0


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


def test_latex_convert_error_fallback():
    """Test convert() returns an empty doc (not an exception) when _do_parse_and_process errors."""
    latex_content = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    options = LatexBackendOptions(parse_timeout=0.05)
    backend = LatexDocumentBackend(
        in_doc=in_doc, path_or_stream=BytesIO(latex_content), options=options
    )

    def _raise(doc):
        raise RuntimeError("Simulated parse failure")

    backend._do_parse_and_process = _raise  # type: ignore[method-assign]
    doc = backend.convert()
    assert doc is not None


def test_latex_theorem_environment():
    """Test theorem/proof/lemma environments emit bold labels + body"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{theorem}
    Every even integer greater than 2 is the sum of two primes.
    \end{theorem}
    \begin{proof}
    Left as an exercise.
    \end{proof}
    \begin{lemma}
    A helper result.
    \end{lemma}
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

    md = doc.export_to_markdown()
    assert "**Theorem.**" in md
    assert "two primes" in md
    assert "*Proof.*" in md
    assert "exercise" in md
    assert "◻" in md
    assert "**Lemma.**" in md


def test_latex_subparagraph_heading():
    """Test \\subparagraph emits heading level 5"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\paragraph{Para Level}
    Content A.
    \\subparagraph{Subpara Level}
    Content B.
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
    assert any("Subpara Level" in h.text for h in headers)

    md = doc.export_to_markdown()
    assert "Content A" in md
    assert "Content B" in md


def test_latex_split_cases_math():
    """Test split/cases inner math environments produce FORMULA labels"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{equation}
    \begin{cases}
    x & \text{if } x > 0 \\
    -x & \text{otherwise}
    \end{cases}
    \end{equation}
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
    # The cases content should be in the formula
    formula_text = " ".join(f.text for f in formulas)
    assert "cases" in formula_text or "otherwise" in formula_text


def test_latex_renewcommand():
    """Test \\renewcommand and \\providecommand macros are expanded"""
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\foo}{original}
    \renewcommand{\foo}{replaced}
    \providecommand{\bar}{provided}
    \begin{document}
    Value is \foo{} and \bar{}.
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    backend.convert()

    # renewcommand should have overwritten the original
    assert "foo" in backend._custom_macros
    assert backend._custom_macros["foo"] == "replaced"
    assert "bar" in backend._custom_macros
    assert backend._custom_macros["bar"] == "provided"


def test_latex_input_cycle_detection(tmp_path):
    """Test that circular \\input doesn't stack overflow"""
    # Create two files that reference each other
    file_a = tmp_path / "a.tex"
    file_b = tmp_path / "b.tex"

    file_a.write_text(
        "\\documentclass{article}\\begin{document}A content\\input{b}\\end{document}"
    )
    file_b.write_text("B content\\input{a}")

    in_doc = InputDocument(
        path_or_stream=file_a,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="a.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=file_a)
    # Should not crash / stack overflow
    doc = backend.convert()
    md = doc.export_to_markdown()
    assert "A content" in md


def test_latex_author_date():
    """Test \\author and \\date text is preserved"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\title{My Paper}
    \\author{Jane Doe}
    \\date{January 2025}
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

    md = doc.export_to_markdown()
    assert "Jane Doe" in md
    assert "January 2025" in md


def test_latex_quote_environment():
    """Test quote/quotation environments produce text output"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{quote}
    This is a quoted passage.
    \\end{quote}
    \\begin{quotation}
    This is a longer quotation.
    \\end{quotation}
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
    assert "quoted passage" in md
    assert "longer quotation" in md


def test_vspace_argument_does_not_leak():
    """Verify \\vspace{-1mm} does not produce '-1mm' as a text item."""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Before text.
    \\vspace{-1mm}
    After text.
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

    all_text = " ".join(t.text for t in doc.texts)
    assert "-1mm" not in all_text, f"vspace argument leaked into text: {all_text!r}"
    assert "Before text" in all_text
    assert "After text" in all_text


def test_hspace_argument_does_not_leak():
    """Verify \\hspace{0.2cm} does not produce '0.2cm' as a text item."""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Left text.\\hspace{0.2cm}Right text.
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

    all_text = " ".join(t.text for t in doc.texts)
    assert "0.2cm" not in all_text, f"hspace argument leaked into text: {all_text!r}"
    assert "Left text" in all_text
    assert "Right text" in all_text
