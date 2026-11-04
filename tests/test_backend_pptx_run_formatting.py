# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for character formatting and links on PowerPoint text runs."""

from pathlib import Path

from docling_core.types.doc import ContentLayer, Script, TextItem
from pptx import Presentation
from pptx.util import Inches

from docling.datamodel.document import DoclingDocument

from .test_backend_pptx import get_converter


def _text_items(doc: DoclingDocument) -> dict[str, TextItem]:
    return {
        item.text: item
        for item, _ in doc.iterate_items(included_content_layers=set(ContentLayer))
        if isinstance(item, TextItem)
    }


def _convert(prs, path: Path) -> DoclingDocument:
    prs.save(path)
    return get_converter().convert(path).document


def test_pptx_mixed_runs_keep_format_and_link(tmp_path: Path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
    paragraph = box.text_frame.paragraphs[0]
    paragraph.add_run().text = "Install "
    bold = paragraph.add_run()
    bold.text = "Docling"
    bold.font.bold = True
    bold.font.italic = True
    paragraph.add_run().text = " library, see "
    link = paragraph.add_run()
    link.text = "the docs"
    link.hyperlink.address = "https://example.com/docs"

    doc = _convert(prs, tmp_path / "mixed_runs.pptx")
    items = _text_items(doc)

    assert items["Docling"].formatting is not None
    assert items["Docling"].formatting.bold
    assert items["Docling"].formatting.italic
    assert str(items["the docs"].hyperlink) == "https://example.com/docs"
    assert items["Install"].formatting is None
    assert items["Install"].hyperlink is None
    assert (
        doc.export_to_markdown()
        == "Install ***Docling*** library, see [the docs](https://example.com/docs)"
    )


def test_pptx_uniform_paragraph_stays_one_item(tmp_path: Path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
    paragraph = box.text_frame.paragraphs[0]
    for part in ("Key ", "Benefits:"):
        run = paragraph.add_run()
        run.text = part
        run.font.bold = True
        run.font.underline = True

    doc = _convert(prs, tmp_path / "uniform.pptx")
    items = _text_items(doc)

    assert list(items) == ["Key Benefits:"]
    assert items["Key Benefits:"].formatting.bold
    assert items["Key Benefits:"].formatting.underline


def test_pptx_list_item_runs_and_superscript(tmp_path: Path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    body = slide.placeholders[1]
    paragraph = body.text_frame.paragraphs[0]
    paragraph.add_run().text = "E = mc"
    sup = paragraph.add_run()
    sup.text = "2"
    sup._r.get_or_add_rPr().set("baseline", "30000")

    doc = _convert(prs, tmp_path / "list_runs.pptx")
    items = _text_items(doc)

    assert items["2"].formatting.script == Script.SUPER
    assert items["E = mc"].formatting is None
