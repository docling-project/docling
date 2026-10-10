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


@pytest.mark.parametrize(
    "start",
    [
        rb"\begin{quote}First\end{quote}",
        rb"{\begin{quote}First\end{quote}}",
        rb"\begin{center}First\end{center}",
    ],
    ids=["quote", "quote_in_group", "center"],
)
def test_latex_enumerate_item_starting_with_nested_text(start: bytes):
    """Text nested in an environment or group at the start of an item is its text.

    Added as a child of an item without text, it would be exported as ``1. ``
    followed by ``First`` on its own line, which a CommonMark parser reads as an
    empty item and a paragraph that takes in ``2. Second``.
    """
    latex_content = (
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item """
        + start
        + rb"""
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

    assert doc.export_to_markdown() == "1. First\n2. Second"


@pytest.mark.parametrize(
    "item",
    [
        b"\\item\n\nFirst\n\nContinuation",
        b"\\item \\begin{quote}\n\nFirst\n\nContinuation\\end{quote}",
        b"\\item {\n\nFirst\n\nContinuation}",
    ],
    ids=["paragraphs", "quote", "group"],
)
def test_latex_enumerate_item_starting_with_paragraph_break(item: bytes):
    """The first paragraph after a leading blank line is the text of the item."""
    latex_content = (
        b"\\documentclass{article}\n\\begin{document}\n\\begin{enumerate}\n"
        + item
        + b"\n\\item Second\n\\end{enumerate}\n\\end{document}\n"
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
    first, second = (child.resolve(doc) for child in outer.children)
    assert (first.text, second.text) == ("First", "Second")
    assert [child.resolve(doc).text for child in first.children] == ["Continuation"]
    assert doc.export_to_markdown() == "1. First\nContinuation\n2. Second"


@pytest.mark.parametrize(
    "item",
    [
        rb"\item \[x=1\] First",
        rb"\item \begin{equation}x=1\end{equation} First",
    ],
    ids=["display_math", "equation"],
)
def test_latex_enumerate_item_starting_with_formula(item: bytes):
    """A formula at the start of an item is on the line of the item, with its text.

    As a block child of an item without text, it was exported as ``1. `` followed
    by the formula on its own line, which a CommonMark parser reads as an empty
    item; ``2. Second`` then loses its numbering.
    """
    latex_content = (
        b"\\documentclass{article}\n\\begin{document}\n\\begin{enumerate}\n"
        + item
        + b"\n\\item Second\n\\end{enumerate}\n\\end{document}\n"
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
    first, second = (child.resolve(doc) for child in outer.children)
    assert (first.text, second.text) == ("", "Second")
    (line,) = (child.resolve(doc) for child in first.children)
    assert line.label == GroupLabel.INLINE
    assert [
        (child.resolve(doc).label, child.resolve(doc).text) for child in line.children
    ] == [(DocItemLabel.FORMULA, "x=1"), (DocItemLabel.TEXT, "First")]
    assert doc.export_to_markdown() == "1. $x=1$ First\n2. Second"


@pytest.mark.parametrize(
    ("item", "children", "expected_md"),
    [
        (
            rb"\item \[x=1\] \begin{itemize}\item In\end{itemize} After",
            [GroupLabel.INLINE, GroupLabel.LIST, DocItemLabel.TEXT],
            "1. $x=1$\n    - In\nAfter\n2. Second",
        ),
        (
            rb"\item \[x=1\]\footnote{Note.} First",
            [GroupLabel.INLINE, DocItemLabel.FOOTNOTE],
            "1. $x=1$ First\nNote.\n2. Second",
        ),
    ],
    ids=["nested_list", "footnote"],
)
def test_latex_enumerate_line_of_item_starting_with_formula(
    item: bytes, children: list, expected_md: str
):
    """The line of an item ends at other content; a footnote does not end it."""
    latex_content = (
        b"\\documentclass{article}\n\\begin{document}\n\\begin{enumerate}\n"
        + item
        + b"\n\\item Second\n\\end{enumerate}\n\\end{document}\n"
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    first = doc.groups[0].children[0].resolve(doc)
    assert [child.resolve(doc).label for child in first.children] == children
    assert doc.export_to_markdown() == expected_md


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
    assert len(list_items) == 2
    assert list_items[0].text == "Term1: Definition one"
    assert list_items[1].text == "Term2: Definition two"


def test_latex_description_list_edge_cases():
    """Test description list with formatted terms, special symbols, and spacing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{description}
    \\item[\\textbf{Term}] Bold term definition
    \\item[--] Dash bullet
    \\item Plain item
    \\item[] Empty term definition
    \\item[Spaced]   Multiple spaces after term
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
    assert [item.text for item in list_items] == [
        "Term: Bold term definition",
        "Dash bullet",
        "Plain item",
        "Empty term definition",
        "Spaced: Multiple spaces after term",
    ]


@pytest.mark.parametrize(
    ("env_name", "items", "expected"),
    [
        ("compactitem", "\\item Alpha\n\\item Beta\n", ["Alpha", "Beta"]),
        (
            "compactdesc",
            "\\item[One] Alpha\n\\item[Two] Beta\n",
            ["One: Alpha", "Two: Beta"],
        ),
    ],
)
def test_latex_items_outside_list_environments(env_name, items, expected):
    """Each item starts a new text item, also outside itemize/enumerate/description.

    These environments reach the inline text path as one node list, so the
    text buffer must be flushed at every item, not only at the end.
    """
    latex_content = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        f"\\begin{{{env_name}}}\n"
        f"{items}"
        f"\\end{{{env_name}}}\n"
        "\\end{document}\n"
    ).encode()
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert [t.text for t in doc.texts] == expected


def test_latex_item_term_without_text():
    """An item term that converts to empty text is dropped, not exported as LaTeX"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{itemize}
    \\item[\\textbullet] Bullet macro
    \\item[\\quad] Quad marker
    \\item[\\hspace{1em}] Spaced marker
    \\end{itemize}
    \\begin{tabular}{l}
    \\begin{itemize}\\item[\\textbullet] Cell item\\end{itemize}
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

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert [item.text for item in list_items] == [
        "Bullet macro",
        "Quad marker",
        "Spaced marker",
    ]
    assert len(doc.tables) == 1
    cells = [c.text.strip() for c in doc.tables[0].data.table_cells]
    assert cells == ["Cell item"]


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
