# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export
from .latex_serializer import SourceLaTeXInlineSerializer

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


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

    assert any(formula.text == "E=mc^2" for formula in formulas)
    dclg = doc.export_to_doclang()
    assert "<formula>E=mc^2</formula>" in dclg
    assert "$E=mc^2$" not in dclg

    md = doc.export_to_markdown()
    # Check delimiters
    assert "$E=mc^2$" in md or r"\( E=mc^2 \)" in md
    assert r"\frac" in md
    assert r"\begin{align}" in md


def test_inline_math_preserves_text_runs_and_math_commands():
    latex_content = rb"""\documentclass{article}
\begin{document}
The long-sought Higgs boson(s) $h$ of electroweak symmetry breaking
in particle physics may soon be observed through ($h\rightarrow\gamma\gamma$).
The cost is \$100. A group {$x$} and \textbf{bold $y$} follow.
\end{document}"""
    stream = BytesIO(latex_content)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="inline.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()

    dclg = doc.export_to_doclang()
    assert "The long-sought Higgs boson(s) <formula>h</formula>" in dclg
    assert "<formula>h</formula>" in dclg
    assert r"<formula>h\rightarrow\gamma\gamma</formula>" in dclg
    assert "electroweak symmetry breaking in particle physics" in dclg
    assert "<formula>x</formula>" in dclg
    assert "<formula>y</formula>" in dclg
    assert "$100" in dclg
    assert "<formula>100</formula>" not in dclg

    from docling_core.transforms.serializer.latex import LaTeXDocSerializer, LaTeXParams

    parsed_latex = (
        LaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(include_annotations=False, enable_chart_tables=False),
        )
        .serialize()
        .text
    )
    assert r"$h\rightarrow\gamma\gamma$" in parsed_latex
    assert r"\$h\$" not in parsed_latex
    assert r"\textbackslash{}rightarrow" not in parsed_latex


def test_source_linebreaks_are_spaces_in_document_text():
    from docling_core.transforms.serializer.latex import LaTeXDocSerializer, LaTeXParams

    latex_content = (
        b"\\documentclass{article}\\begin{document}first\n\t$h$\n\tlast\\end{document}"
    )
    stream = BytesIO(latex_content)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="linebreaks.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()
    parsed_latex = (
        LaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(include_annotations=False, enable_chart_tables=False),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )

    assert "first $h$ last" in parsed_latex
    for item in doc.texts:
        if item.label not in {DocItemLabel.FORMULA, DocItemLabel.CODE}:
            assert "\n" not in item.text
            assert "\t" not in item.text
            assert "\n" not in item.orig
            assert "\t" not in item.orig


def test_inline_math_source_linebreak_does_not_wrap_formula_content():
    latex_content = rb"""\documentclass{article}
\begin{document}
The interval is $70 < Q <
115$ GeV.
\end{document}"""
    stream = BytesIO(latex_content)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="inline-linebreak.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()

    formulas = [item for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert [formula.text for formula in formulas] == ["70 < Q < 115"]
    assert "<formula><![CDATA[70 < Q < 115]]></formula>" in doc.export_to_doclang()


def test_def_redefined_accent_does_not_consume_math_delimiter():
    latex_content = rb"""\documentclass{article}
\def\v{\vert}
\begin{document}
Let $x=\v$ and $y$ follow.
\end{document}"""
    stream = BytesIO(latex_content)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="redefined-accent.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()

    formulas = [item for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert [formula.text for formula in formulas] == [r"x=\vert", "y"]
    assert "and" in doc.export_to_doclang()


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


def test_latex_preserves_numbered_and_aligned_math_environments():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{eqnarray}
    a&=&b\\
    c&=&d
    \end{eqnarray}
    \begin{equation}
    x=y\tag{1.2a}
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
    formulas = [
        item.text
        for item in backend.convert().texts
        if item.label == DocItemLabel.FORMULA
    ]

    assert len(formulas) == 2
    assert formulas[0].startswith(r"\begin{eqnarray}")
    assert formulas[0].endswith(r"\end{eqnarray}")
    assert formulas[1].startswith(r"\begin{equation}")
    assert r"\tag{1.2a}" in formulas[1]


def test_latex_shortcuts_do_not_change_macro_definitions():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \newcommand{\be}{\begin{equation}}
    \newcommand{\ee}{\end{equation}}
    \be x=y \ee
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    processed = backend._preprocess_custom_macros(backend.latex_text)
    assert r"\newcommand" not in processed
    assert backend._custom_macros["be"] == r"\begin{equation}"
    assert backend._custom_macros["ee"] == r"\end{equation}"
    assert r"\begin{equation} x=y \end{equation}" in processed
    formulas = [
        item.text
        for item in backend.convert().texts
        if item.label == DocItemLabel.FORMULA
    ]
    assert len(formulas) == 1
    assert formulas[0].startswith(r"\begin{equation}")


def test_latex_expands_nested_zero_argument_macros():
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\Real}{\mathbb{R}}
    \newcommand{\Rd}{\Real^d}
    \begin{document}
    The space $\Rd$ is used here.
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
    text = " ".join(
        item.text for item in doc.texts if item.content_layer == ContentLayer.BODY
    )
    assert r"\mathbb{R}^d" in text
    assert r"\Real" not in text


def test_latex_math_environment():
    """Test that the math environment stays inside its paragraph."""
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
    assert any(formula.text == "a+b" for formula in formulas)
    dclg = doc.export_to_doclang()
    assert "<formula>a+b</formula>" in dclg
    assert "Inline: " in dclg


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
