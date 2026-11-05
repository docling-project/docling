# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import FormulaItem, Size

from docling.utils.nemotron_parse_utils import parse_nemotron_parse_v2


@pytest.mark.parametrize(
    "opening, closing",
    [("$", "$"), ("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]"), ("", "")],
)
def test_formula_export_has_one_delimiter_pair(opening: str, closing: str) -> None:
    latex = r"\left\{x_1 * y_2 : x > 0\right\}"
    raw = f"{opening}{latex}{closing}"
    source = f"<x_0.2><y_0.2>{raw}<x_0.8><y_0.4><class_Formula>"
    size = Size(width=600, height=800)
    doc = parse_nemotron_parse_v2(source, size, size, page_no=1)
    formula = doc.texts[0]
    assert isinstance(formula, FormulaItem)
    assert formula.text == latex
    assert formula.orig == raw
    assert formula.prov
    assert doc.export_to_markdown() == f"$${latex}$$"


def test_multiline_formula_keeps_latex_and_line_breaks() -> None:
    latex = "\\begin{aligned}\nx &= y \\\\\nz &= w\n\\end{aligned}"
    source = f"<x_0.2><y_0.2>$$\n{latex}\n$$<x_0.8><y_0.4><class_Formula>"
    size = Size(width=600, height=800)
    doc = parse_nemotron_parse_v2(source, size, size, page_no=1)
    assert doc.texts[0].text == latex
    assert doc.export_to_markdown() == f"$${latex}$$"


def test_formula_segments_lose_every_inline_delimiter() -> None:
    raw = r"(6) \(x_1=a\), \(x_2=b\)"
    source = f"<x_0.2><y_0.2>{raw}<x_0.8><y_0.4><class_Formula>"
    size = Size(width=600, height=800)
    doc = parse_nemotron_parse_v2(source, size, size, page_no=1)
    assert doc.texts[0].text == "(6) x_1=a, x_2=b"
    assert doc.texts[0].orig == raw
