# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re
from io import BytesIO

from docling_core.types.doc import ContentLayer, DocItemLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


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
    items = [item.text for item in doc.texts if item.label == DocItemLabel.LIST_ITEM]
    assert len(items) == 2
    assert items[0].startswith("[ref1] Author One")
    assert items[1].startswith("[ref2] Author Two")
    assert not any(item == "9" for item in items)


def test_latex_layout_assignments_and_bibliography_style_are_not_body_text():
    latex_content = b"""\\documentclass{article}
\\begin{document}
Visible text. \\vfuzz=2pt
\\hfuzz = 1.5pt
\\bibliographystyle{amsplain}
\\begin{thebibliography}{10}
\\bibitem{ref} Author, Title.
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

    texts = [item.text for item in doc.texts]
    assert any("Visible text." in text for text in texts)
    assert any("[ref] Author, Title." in text for text in texts)
    assert not any(
        "=2pt" in text or "1.5pt" in text or "amsplain" in text for text in texts
    )


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


def test_latex_nested_formatting_macros():
    """Nested formatting macros must be fully unwrapped without leaking raw commands."""
    # Each case is a single-paragraph document so we can assert on the exact text node.
    cases = [
        # (description, latex_body, expected_text)
        (
            "textsc inline stays in paragraph",
            b"\\documentclass{article}\\begin{document}Some \\textsc{small caps} text.\\end{document}",
            "Some small caps text.",
        ),
        (
            "textcolor+textbf inline stays in paragraph",
            b"\\documentclass{article}\\begin{document}Some \\textcolor{blue}{\\textbf{nested bold}} and more.\\end{document}",
            "Some nested bold and more.",
        ),
        (
            "deep nesting textcolor+textbf+textsc inline",
            b"\\documentclass{article}\\begin{document}Text \\textcolor{blue}{\\textbf{\\textsc{inline}}} here.\\end{document}",
            "Text inline here.",
        ),
        (
            "textbf wrapping textsc inline",
            b"\\documentclass{article}\\begin{document}Plain \\textbf{\\textsc{bold sc}} text.\\end{document}",
            "Plain bold sc text.",
        ),
        (
            "color name must not appear in heading",
            b"\\documentclass{article}\\begin{document}\\section{\\textcolor{blue}{\\textbf{\\textsc{[SEP]}}}}\\end{document}",
            "[SEP]",
        ),
    ]

    for description, latex_content, expected in cases:
        in_doc = InputDocument(
            path_or_stream=BytesIO(latex_content),
            format=InputFormat.LATEX,
            backend=LatexDocumentBackend,
            filename="test.tex",
        )
        backend = LatexDocumentBackend(
            in_doc=in_doc, path_or_stream=BytesIO(latex_content)
        )
        doc = backend.convert()

        all_text = " ".join(
            t.text for t in doc.texts if t.content_layer == ContentLayer.BODY
        )
        assert expected in all_text, (
            f"[{description}] expected {expected!r} in output, got: {all_text!r}"
        )

        leaked = re.findall(r"\\[a-zA-Z]+", all_text)
        assert not leaked, f"[{description}] leaked raw LaTeX commands: {leaked}"
