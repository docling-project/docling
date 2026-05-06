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
