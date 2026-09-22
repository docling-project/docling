# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression cases for whitespace after inline LaTeX macros (#4339)."""

from io import BytesIO

import pytest
from docling_core.types.doc import DocItemLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


@pytest.mark.parametrize(
    ("body", "expected_paragraphs"),
    [
        (
            r"A \textit{it} text. B \textbf{bf} more."
            + "\n\n"
            + r"C \emph{em}x and \textit{it}.",
            ["A it text. B bf more.", "C emx and it."],
        ),
        (
            r"Some background with \textbf{bold} and \textit{italic} text.",
            ["Some background with bold and italic text."],
        ),
        (
            r"A \textbf{bold}\textit{italic} text.",
            ["A bolditalic text."],
        ),
        (
            "First.\n\n"
            + r"Second \textit{emphasis} continues."
            + "\n\nThird.",
            ["First.", "Second emphasis continues.", "Third."],
        ),
    ],
)
def test_latex_preserves_whitespace_across_paragraphs_and_macros(
    body: str, expected_paragraphs: list[str]
) -> None:
    source = (
        r"\documentclass{article}"
        + "\n"
        + r"\begin{document}"
        + "\n"
        + body
        + "\n"
        + r"\end{document}"
        + "\n"
    ).encode("utf-8")
    in_doc = InputDocument(
        path_or_stream=BytesIO(source),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="inline_whitespace.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(source))
    doc = backend.convert()

    text_items = [
        item.text
        for item in doc.texts
        if item.label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH)
    ]
    assert text_items == expected_paragraphs
