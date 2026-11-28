# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the text of PowerPoint runs set in a symbol font."""

from pathlib import Path

from docling_core.types.doc import ContentLayer
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches

from .test_backend_pptx import get_converter


def _add_runs(paragraph, runs: list[tuple[str, str | None]]) -> None:
    """Add runs of text to a paragraph, each with an optional a:sym font."""
    for text, sym_font in runs:
        run = paragraph.add_run()
        run.text = text
        if sym_font is not None:
            rpr = run._r.get_or_add_rPr()
            rpr.append(rpr.makeelement(qn("a:sym"), {"typeface": sym_font}))


def test_pptx_symbol_font_characters_are_mapped_to_unicode(tmp_path: Path):
    """A character inserted from the Symbol font is output as its Unicode symbol.

    PowerPoint stores it as a private-use code point (U+F000-U+F0FF) in a run
    whose ``a:sym`` font is Symbol, so an arrow or a Greek letter came out as
    an unreadable private-use character.
    """
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    _add_runs(
        box.text_frame.paragraphs[0],
        [("A ", None), ("\uf0ae", "Symbol"), (" b", None)],
    )
    _add_runs(
        box.text_frame.add_paragraph(), [("\uf061", "Symbol"), (" = 0.05", "Symbol")]
    )
    table = slide.shapes.add_table(1, 1, Inches(1), Inches(3), Inches(4), Inches(1))
    _add_runs(
        table.table.cell(0, 0).text_frame.paragraphs[0],
        [("\uf062", "Symbol"), (" risk", None)],
    )
    _add_runs(
        slide.notes_slide.notes_text_frame.paragraphs[0],
        [("\uf065", "Symbol"), (" is the empty string", None)],
    )
    pptx_path = tmp_path / "symbol_font.pptx"
    prs.save(pptx_path)

    doc = get_converter().convert(pptx_path).document

    body = [item.text for item in doc.texts if item.content_layer == ContentLayer.BODY]
    assert body == ["A \u2192 b", "\u03b1 = 0.05"]
    assert doc.tables[0].data.table_cells[0].text == "\u03b2 risk"
    notes = [
        item.text for item in doc.texts if item.content_layer == ContentLayer.NOTES
    ]
    assert notes == ["\u03b5 is the empty string"]
