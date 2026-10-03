# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


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
    assert [(item.text, item.enumerated) for item in list_items] == [
        ("Alpha", True),
        ("Beta", True),
    ]
    assert doc.export_to_markdown() == "1. Alpha\n2. Beta"


def test_latex_enumerate_numbering_with_nested_content():
    """Each enumerate item takes one number, whatever it contains.

    Nested lists and further paragraphs of an item are added inside the item,
    so they do not take a position in the numbering of the outer list.
    """
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item First
      \begin{itemize}
      \item Detail
      \end{itemize}
      Rest of the first item.
    \item Second

    Second paragraph of the second item.
    \item Third
      \begin{enumerate}
      \item Sub one
      \item Sub two
      \end{enumerate}
    \item Fourth
    \end{enumerate}
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

    outer = doc.groups[0]
    outer_items = [child.resolve(doc) for child in outer.children]
    assert [(item.label, item.text) for item in outer_items] == [
        (DocItemLabel.LIST_ITEM, "First"),
        (DocItemLabel.LIST_ITEM, "Second"),
        (DocItemLabel.LIST_ITEM, "Third"),
        (DocItemLabel.LIST_ITEM, "Fourth"),
    ]
    assert all(item.enumerated for item in outer_items)

    md = doc.export_to_markdown()
    assert "\n2. Second" in md
    assert "\n3. Third\n    1. Sub one\n    2. Sub two\n4. Fourth" in md
    assert "Rest of the first item." in md
    assert "Second paragraph of the second item." in md


@pytest.mark.parametrize(
    "content",
    [
        rb"\[x=1\]",
        rb"\begin{equation}x=1\end{equation}",
        rb"\href{https://example.com}{a link}",
        rb"\begin{quote}A quote.\end{quote}",
    ],
    ids=["display_math", "equation", "href", "quote"],
)
def test_latex_enumerate_item_with_block_content(content: bytes):
    """Content that flushes the text buffer stays inside its item.

    Added as siblings in the list, the content and the text after it would each
    take a number, e.g. ``1. First``, the formula, ``3. continuation``,
    ``4. Second``.
    """
    latex_content = (
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item First """
        + content
        + rb""" continuation
    \item Second
    \end{enumerate}
    \end{document}
    """
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    outer = doc.groups[0]
    outer_items = [child.resolve(doc) for child in outer.children]
    assert [(item.label, item.text) for item in outer_items] == [
        (DocItemLabel.LIST_ITEM, "First"),
        (DocItemLabel.LIST_ITEM, "Second"),
    ]
    assert all(item.enumerated for item in outer_items)

    md = doc.export_to_markdown()
    assert md.startswith("1. First\n")
    assert md.endswith("\n2. Second")
    assert "continuation" in md


@pytest.mark.parametrize(
    ("start", "child_labels"),
    [
        (rb"\footnote{A note.}", [DocItemLabel.FOOTNOTE]),
        (rb"\newline", []),
    ],
    ids=["footnote", "newline"],
)
def test_latex_enumerate_item_starting_with_footnote_or_newline(
    start: bytes, child_labels: list[DocItemLabel]
):
    """The text after a leading footnote or line break is the text of the item.

    A footnote stays attached to the item; a line break before any content has
    nothing to break.
    """
    latex_content = (
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item """
        + start
        + rb""" First
    \item Second
    \end{enumerate}
    \end{document}
    """
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    outer = doc.groups[0]
    outer_items = [child.resolve(doc) for child in outer.children]
    assert [item.text for item in outer_items] == ["First", "Second"]
    assert all(item.enumerated for item in outer_items)
    assert [
        child.resolve(doc).label for child in outer_items[0].children
    ] == child_labels

    md = doc.export_to_markdown()
    assert md.startswith("1. First\n")
    assert md.endswith("\n2. Second")


def test_latex_enumerate_empty_item_keeps_numbering():
    """An empty ``\\item`` still takes a number, as in the typeset document."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item One
    \item
    \item \label{it:three}
    \item Four
    \end{enumerate}
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

    outer = doc.groups[0]
    outer_items = [child.resolve(doc) for child in outer.children]
    assert [item.text for item in outer_items] == ["One", "", "", "Four"]
    assert all(item.enumerated for item in outer_items)
    assert doc.export_to_markdown().endswith("\n4. Four")


@pytest.mark.parametrize(
    ("leading", "expected_md"),
    [
        (rb"% The steps.", "1. One\n2. Two"),
        (rb"\setlength{\itemsep}{0pt}", "1. One\n2. Two"),
        (rb"\label{list:steps}", "1. One\n2. Two"),
        (rb"Stray text.", "1. Stray text.\n2. One\n3. Two"),
    ],
    ids=["comment", "setlength", "label", "stray_text"],
)
def test_latex_enumerate_content_before_first_item(leading: bytes, expected_md: str):
    """Content before the first ``\\item`` does not add an item to the list.

    Stray text there, a LaTeX error, is kept as a numbered item.
    """
    latex_content = (
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    """
        + leading
        + rb"""
    \item One
    \item Two
    \end{enumerate}
    \end{document}
    """
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert doc.export_to_markdown() == expected_md


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
