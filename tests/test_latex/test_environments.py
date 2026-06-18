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
