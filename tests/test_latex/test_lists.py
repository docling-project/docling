from docling_core.types.doc import DocItemLabel, GroupLabel

from ._utils import make_backend


def test_latex_list_itemize():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{itemize}
    \\item First item
    \\item Second item
    \\item Third item
    \\end{itemize}
    \\end{document}
    """
    ).convert()

    item_texts = [
        item.text for item in doc.texts if item.label == DocItemLabel.LIST_ITEM
    ]
    assert len(item_texts) >= 3
    assert any("First item" in t for t in item_texts)
    assert any("Second item" in t for t in item_texts)


def test_latex_list_enumerate():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{enumerate}
    \\item Alpha
    \\item Beta
    \\end{enumerate}
    \\end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]) >= 2


def test_latex_description_list():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{description}
    \\item[Term1] Definition one
    \\item[Term2] Definition two
    \\end{description}
    \\end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]) >= 2


def test_latex_list_nested():
    doc = make_backend(
        b"""
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
    ).convert()

    list_groups = [g for g in doc.groups if g.label == GroupLabel.LIST]
    item_texts = [
        item.text for item in doc.texts if item.label == DocItemLabel.LIST_ITEM
    ]
    assert len(list_groups) >= 1
    assert len(item_texts) >= 3
    assert any("Outer item one" in t for t in item_texts)
    assert any("Inner item A" in t or "Inner item B" in t for t in item_texts)
    assert any("Numbered inner 1" in t or "Numbered inner 2" in t for t in item_texts)
