# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    OrigListItemMarkerMode,
)
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


def test_latex_enumerate_custom_labels_stay_ordered():
    """Test labelled items keep an enumerate an ordered list"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{enumerate}
    \item[(a)] First
    \item[(b)] Second
    \item Third
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

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert [(item.marker, item.enumerated) for item in list_items] == [
        ("(a)", True),
        ("(b)", True),
        ("", True),
    ]
    # by default the Markdown export shows a custom marker as "- (a)"; with the
    # original markers kept, the list is numbered
    serializer = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(orig_list_item_marker_mode=OrigListItemMarkerMode.ALWAYS),
    )
    assert serializer.serialize().text == "1. (a) First\n2. (b) Second\n3. Third"


def test_latex_description_list():
    """Test description list with optional item labels"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{description}
    \\item[Term1] Definition one
    \\item[\\textbf{Term2}] Definition two
    \\item[Term3:] Definition with a colon in the term
    \\item Definition without a term
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
    assert [(item.marker, item.text) for item in list_items] == [
        ("Term1:", "Definition one"),
        ("Term2:", "Definition two"),
        ("Term3:", "Definition with a colon in the term"),
        ("", "Definition without a term"),
    ]
    assert doc.export_to_markdown() == (
        "- Term1: Definition one\n"
        "- Term2: Definition two\n"
        "- Term3: Definition with a colon in the term\n"
        "- Definition without a term"
    )


@pytest.mark.parametrize("env", ["itemize", "enumerate"])
def test_latex_list_custom_item_label(env: str):
    """Test the custom label of a list item is its marker, an empty one is ignored"""
    latex_content = (
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{ENV}
    \item[(a)] First
    \item[] Second
    \item Third
    \end{ENV}
    \end{document}
    """
    ).replace(b"ENV", env.encode())
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert [(item.marker, item.text) for item in list_items] == [
        ("(a)", "First"),
        ("", "Second"),
        ("", "Third"),
    ]
    assert all(item.enumerated == (env == "enumerate") for item in list_items)


def test_latex_description_term_before_block_definition():
    """Test a term stays first when its definition starts with block content"""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{description}
    \item[Energy] \[E=mc^2\]
    \item[Units] \begin{itemize}\item joule\end{itemize} or electronvolt
    \item[Speed] Distance per time.
    \item[Pending]
    \end{description}
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

    list_group = next(g for g in doc.groups if g.label == GroupLabel.LIST)
    energy, units, speed, pending = (ref.resolve(doc) for ref in list_group.children)

    assert (energy.marker, energy.text) == ("Energy:", "")
    (formula,) = (ref.resolve(doc) for ref in energy.children)
    assert (formula.label, formula.text) == (DocItemLabel.FORMULA, "E=mc^2")

    # the text after the nested list still belongs to the definition
    assert units.marker == "Units:"
    nested, rest = (ref.resolve(doc) for ref in units.children)
    assert nested.label == GroupLabel.LIST
    assert [ref.resolve(doc).text for ref in nested.children] == ["joule"]
    assert (rest.label, rest.text) == (DocItemLabel.TEXT, "or electronvolt")

    assert (speed.marker, speed.text, speed.children) == (
        "Speed:",
        "Distance per time.",
        [],
    )

    # a term without a definition keeps its item
    assert (pending.marker, pending.text, pending.children) == ("Pending:", "", [])


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
