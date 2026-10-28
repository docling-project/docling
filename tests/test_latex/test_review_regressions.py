# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re
from io import BytesIO
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest
from doclang.validation import validate
from docling_core.types.doc import ContentLayer, DocItemLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def _convert(source: str):
    stream = BytesIO(source.encode())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="review.tex",
    )
    return LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()


def test_commented_definition_does_not_hide_body():
    doc = _convert(
        r"""\documentclass{article}
% \newcommand{\unused}{
\begin{document}
Important content.
\end{document}
"""
    )
    assert "Important content." in doc.export_to_markdown()


def test_comment_inside_definition_does_not_hide_body():
    doc = _convert(
        r"""\documentclass{article}
\newcommand{\term}{First% {
Second}
\begin{document}
Important content. \term
\end{document}
"""
    )
    assert doc.export_to_markdown() == "Important content. FirstSecond"


@pytest.mark.parametrize("comment", ["% comment\n", "% comment\n% more\n"])
def test_comment_inside_definition_preserves_control_word_boundary(comment: str):
    doc = _convert(
        "\\documentclass{article}\n\\newcommand{\\brand}{Docling}\n"
        "\\newcommand{\\term}[1]{\\brand" + comment + "rocks #1}\n\\begin{document}\n"
        "Important content. \\term{now}\n\\end{document}\n"
    )
    assert doc.export_to_markdown() == "Important content. Doclingrocks now"


def test_comment_inside_math_definition_preserves_control_word_boundary():
    doc = _convert(
        r"""\documentclass{article}
\newcommand{\expression}{\alpha% comment
beta}
\begin{document}
$\expression$
\end{document}
"""
    )
    formulas = [item.text for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formulas == [r"\alpha beta"]


def test_comment_before_sqrt_argument_keeps_argument():
    doc = _convert(
        r"""\documentclass{article}
\newcommand{\expression}{\sqrt% comment
x}
\begin{document}
$\expression$
\end{document}
"""
    )
    equivalent = _convert(
        r"\documentclass{article}\newcommand{\expression}{\sqrt x}"
        r"\begin{document}$\expression$\end{document}"
    )
    formulas = [item.text for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formulas == [r"\sqrt x"]
    assert doc.export_to_markdown() == equivalent.export_to_markdown() == r"$\sqrt x$"


def test_comment_before_frac_arguments_keeps_arguments():
    doc = _convert(
        r"""\documentclass{article}
\newcommand{\expression}{\frac% comment
ab}
\begin{document}
$\expression$
\end{document}
"""
    )
    equivalent = _convert(
        r"\documentclass{article}\newcommand{\expression}{\frac ab}"
        r"\begin{document}$\expression$\end{document}"
    )
    formulas = [item.text for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formulas == [r"\frac ab"]
    assert doc.export_to_markdown() == equivalent.export_to_markdown() == r"$\frac ab$"


def test_default_doclang_export_validates_with_source_commands(tmp_path):
    doc = _convert(
        r"\documentclass{article}\newcommand{\term}{Defined}"
        r"\begin{document}Body.\end{document}"
    )
    path = tmp_path / "source-commands.dclg"
    path.write_text(doc.export_to_doclang(), encoding="utf-8")
    validate(path, allow_empty_namespace=True, xsd_only=True)


def test_body_footnotes_keep_source_order():
    doc = _convert(
        r"""\documentclass{article}
\begin{document}
First\footnote{note one} second\footnote{note two} third.
\end{document}
"""
    )
    assert doc.export_to_markdown() == (
        "First\n\nnote one\n\nsecond\n\nnote two\n\nthird."
    )


@pytest.mark.parametrize(
    ("math", "expected"),
    [
        (r"\begin{math}a+" + "\n b" + r"\end{math}", "a+ b"),
        (r"\ensuremath{c+" + "\n d}", "c+ d"),
    ],
)
def test_multiline_inline_formula_has_plain_doclang_content(math: str, expected: str):
    doc = _convert(
        "\\documentclass{article}\\begin{document}Before "
        + math
        + " after.\\end{document}"
    )
    root = ET.fromstring(doc.export_to_doclang())
    formulas = [element for element in root.iter() if element.tag == "formula"]
    assert len(formulas) == 1
    assert formulas[0].text == expected
    assert not list(formulas[0])


def test_shortcut_name_defined_as_beta_keeps_math_meaning():
    doc = _convert(
        r"\documentclass{article}\newcommand{\be}{\beta}"
        r"\begin{document}Before $\be+1$ after.\end{document}"
    )
    formulas = [item.text for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formulas == [r"\beta+1"]
    assert "Before" in doc.export_to_markdown()
    assert "after." in doc.export_to_markdown()


@pytest.mark.parametrize(
    ("opening", "closing_name", "closing", "expected"),
    [
        (
            r"\begin{eqnarray}",
            "eeq",
            r"\end{eqnarray}",
            r"\begin{eqnarray} x&=&y \end{eqnarray}",
        ),
        (
            r"\begin{eqnarray}",
            "eneq",
            r"\end{eqnarray}",
            r"\begin{eqnarray} x&=&y \end{eqnarray}",
        ),
        ("$$", "eeq", "$$", "x&=&y"),
    ],
)
def test_source_defined_display_shortcuts(
    opening: str, closing_name: str, closing: str, expected: str
):
    doc = _convert(
        r"\documentclass{article}\newcommand{\beq}{"
        + opening
        + "}\\newcommand{\\"
        + closing_name
        + r"}{"
        + closing
        + "}\\begin{document}\\beq x&=&y \\"
        + closing_name
        + r"\end{document}"
    )
    formulas = [item.text for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formulas == [expected]


def test_many_unused_macros_do_not_slow_formula_expansion():
    names = (
        "unused"
        + chr(97 + index // 676)
        + chr(97 + (index // 26) % 26)
        + chr(97 + index % 26)
        for index in range(700)
    )
    definitions = "".join("\\newcommand{\\" + name + "}{x}" for name in names)
    source = (
        r"\documentclass{article}\newcommand{\used}{1}"
        + definitions
        + r"\begin{document}$x+\used$\end{document}"
    )
    stream = BytesIO(source.encode())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="many-macros.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream)
    backend._preprocess_custom_macros(backend.latex_text)
    assert len(backend._custom_macros) == 701

    with patch.object(re, "sub", wraps=re.sub) as substitute:
        assert backend._expand_macros(r"x+\used") == "x+1"
    assert substitute.call_count < 10


def test_verbatim_definition_is_preserved():
    doc = _convert(
        r"""\documentclass{article}
\begin{document}
\begin{verbatim}
\newcommand{\foo}{First}
\end{verbatim}
\end{document}
"""
    )
    assert r"\newcommand{\foo}{First}" in doc.export_to_markdown()


def test_body_definitions_are_supplementary():
    declarations = [
        r"\newcommand{\term}{Defined}",
        r"\newenvironment{custom}{Start}{End}",
        r"\newtheorem{lemma}{Lemma}",
        r"\newlength{\customlength}",
        r"\newcounter{customcounter}",
        r"\newif\ifcustom",
        r"\newtheoremstyle{customstyle}{1pt}{2pt}{}{}{}{.}{1em}{}",
    ]
    doc = _convert(
        "\\documentclass{article}\\begin{document}\n"
        + "\n".join(declarations)
        + "\nVisible body.\\end{document}"
    )

    source_commands = [
        item.text for item in doc.texts if item.content_layer == ContentLayer.FURNITURE
    ]
    assert not (set(declarations) - set(source_commands)), set(declarations) - set(
        source_commands
    )
    markdown = doc.export_to_markdown()
    assert "Visible body." in markdown
    assert all(command not in markdown for command in declarations)


def test_math_environment_inside_formatting():
    doc = _convert(
        r"""\documentclass{article}
\begin{document}
\textbf{Before \begin{math}x\end{math} after.}
\end{document}
"""
    )
    markdown = doc.export_to_markdown()
    assert "Before" in markdown
    assert "$x$" in markdown
    assert "after." in markdown


def test_custom_macro_arguments_with_inline_math():
    doc = _convert(
        r"""\documentclass{article}
\newcommand{\pair}[2]{#1 and #2}
\begin{document}
\textbf{\pair{A}{B} $x$}
\end{document}
"""
    )
    markdown = doc.export_to_markdown()
    assert "A and B" in markdown
    assert "$x$" in markdown
    assert "#1" not in markdown
