from docling_core.types.doc import DocItemLabel

from ._utils import make_backend


def test_latex_abstract_environment():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{abstract}
    This is the abstract content.
    \\end{abstract}
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Abstract" in md
    assert "abstract content" in md


def test_latex_verbatim_environment():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{verbatim}
    def hello():
        print("world")
    \\end{verbatim}
    \\end{document}
    """
    ).convert()

    code_items = [t for t in doc.texts if t.label == DocItemLabel.CODE]
    assert len(code_items) >= 1
    assert "hello" in code_items[0].text or "print" in code_items[0].text


def test_latex_lstlisting_environment():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{lstlisting}
    int main() {
        return 0;
    }
    \\end{lstlisting}
    \\end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.CODE]) >= 1


def test_latex_bibliography():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    Some text.
    \\begin{thebibliography}{9}
    \\bibitem{ref1} Author One, Title One, 2020.
    \\bibitem{ref2} Author Two, Title Two, 2021.
    \\end{thebibliography}
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "References" in md


def test_latex_heading_levels():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\part{Part One}
    \\chapter{Chapter One}
    \\section{Section One}
    \\subsection{Subsection One}
    \\subsubsection{Subsubsection One}
    \\end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]) >= 3


def test_latex_theorem_environment():
    md = (
        make_backend(
            rb"""
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
        )
        .convert()
        .export_to_markdown()
    )
    assert "**Theorem.**" in md
    assert "two primes" in md
    assert "*Proof.*" in md
    assert "exercise" in md
    assert "◻" in md
    assert "**Lemma.**" in md


def test_latex_subparagraph_heading():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\paragraph{Para Level}
    Content A.
    \\subparagraph{Subpara Level}
    Content B.
    \\end{document}
    """
    ).convert()

    headers = [t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    md = doc.export_to_markdown()
    assert any("Subpara Level" in h.text for h in headers)
    assert "Content A" in md
    assert "Content B" in md


def test_latex_quote_environment():
    md = (
        make_backend(
            b"""
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
        )
        .convert()
        .export_to_markdown()
    )
    assert "quoted passage" in md
    assert "longer quotation" in md
