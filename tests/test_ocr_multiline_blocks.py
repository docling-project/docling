# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import DocItemLabel, Size

from docling.utils.deepseekocr_utils import (
    parse_deepseekocr_markdown,
    parse_unlimited_ocr_markdown,
)


@pytest.mark.parametrize("parser", ["deepseek", "unlimited"])
def test_text_block_keeps_continuation_lines_and_following_table(parser: str) -> None:
    def marker(label: str, top: int) -> str:
        if parser == "deepseek":
            return f"{label}[[100,{top},900,{top + 100}]]\n"
        return f"<|det|>{label} [100,{top},900,{top + 100}]<|/det|>"

    source = (
        marker("text", 100)
        + "First line\nSecond line\n\nThird paragraph\n"
        + marker("table", 400)
        + "<table><tr><td>Total</td></tr></table>\n"
        + marker("table_caption", 600)
        + "Caption first line\nCaption second line\n"
        + marker("text", 800)
        + "Final block"
    )
    parse = (
        parse_deepseekocr_markdown
        if parser == "deepseek"
        else parse_unlimited_ocr_markdown
    )
    doc = parse(source, Size(width=600, height=800), page_no=2)
    text = next(item for item in doc.texts if item.label == DocItemLabel.TEXT)
    assert text.text == "First line\nSecond line\n\nThird paragraph"
    assert text.prov[0].page_no == 2
    assert len(doc.tables) == 1
    assert doc.tables[0].data.grid[0][0].text == "Total"
    caption = doc.tables[0].captions[0].resolve(doc)
    assert caption.text == "Caption first line\nCaption second line"
    assert doc.texts[-1].text == "Final block"
    assert "Second line" in doc.export_to_markdown()


def test_caption_does_not_claim_table_markup_already_collected() -> None:
    source = (
        "table[[100,100,900,300]]\n"
        "table_caption[[100,400,900,500]]\nCaption first\nCaption second\n\n"
        "<table><tr><td>Total</td></tr></table>\n\n"
        "text[[100,600,900,700]]\nFinal block"
    )
    doc = parse_deepseekocr_markdown(source, Size(width=600, height=800), page_no=1)
    assert doc.tables[0].data.grid[0][0].text == "Total"
    assert (
        doc.tables[0].captions[0].resolve(doc).text == "Caption first\nCaption second"
    )
    assert all("<table>" not in item.text for item in doc.texts)
    assert doc.texts[-1].text == "Final block"
