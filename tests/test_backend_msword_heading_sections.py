# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Section containment after headings that use Word list numbering."""

from pathlib import Path

import pytest
from docling_core.types.doc import ListGroup, SectionHeaderItem
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from .test_backend_msword_lists import _convert_docx


def _set_numbering(properties, num_id: int, level: int) -> None:
    num_pr = properties.get_or_add_numPr()
    num_pr.get_or_add_numId().val = num_id
    num_pr.get_or_add_ilvl().val = level


@pytest.mark.parametrize("numbering_source", ["style", "paragraph", "none"])
@pytest.mark.parametrize(
    "following_block", ["paragraph", "list", "table", "direct-list"]
)
def test_numbered_heading_keeps_section_content(
    tmp_path: Path, numbering_source: str, following_block: str
) -> None:
    """Numbering a heading must not make the next block close a list."""
    source = Document()
    numbering = source.part.numbering_part.element
    for child in list(numbering):
        numbering.remove(child)
    numbering.extend(
        parse_xml(
            f"<w:numbering {nsdecls('w')}>"
            '<w:abstractNum w:abstractNumId="10">'
            '<w:multiLevelType w:val="multilevel"/>'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/>'
            '<w:numFmt w:val="decimal"/><w:pStyle w:val="Heading1"/>'
            '<w:lvlText w:val="%1."/></w:lvl>'
            '<w:lvl w:ilvl="1"><w:start w:val="1"/>'
            '<w:numFmt w:val="decimal"/><w:pStyle w:val="Heading2"/>'
            '<w:lvlText w:val="%1.%2"/></w:lvl></w:abstractNum>'
            '<w:abstractNum w:abstractNumId="11">'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/>'
            '<w:numFmt w:val="bullet"/><w:lvlText w:val="-"/>'
            "</w:lvl></w:abstractNum>"
            '<w:num w:numId="1"><w:abstractNumId w:val="10"/></w:num>'
            '<w:num w:numId="2"><w:abstractNumId w:val="11"/></w:num>'
            "</w:numbering>"
        )
    )
    if numbering_source == "style":
        for level in (1, 2):
            _set_numbering(
                source.styles[f"Heading {level}"].element.get_or_add_pPr(),
                1,
                level - 1,
            )

    def add_heading(text: str, level: int) -> None:
        paragraph = source.add_heading(text, level=level)
        if numbering_source == "paragraph":
            _set_numbering(paragraph._p.get_or_add_pPr(), 1, level - 1)

    add_heading("Scope", 1)
    if following_block in {"paragraph", "list"}:
        source.add_paragraph("Scope text.")
    if following_block in {"list", "direct-list"}:
        paragraph = source.add_paragraph("pumps", style="List Paragraph")
        _set_numbering(paragraph._p.get_or_add_pPr(), 2, 0)
    elif following_block == "table":
        table = source.add_table(rows=2, cols=2)
        for row, values in zip(table.rows, [("Size", "Limit"), ("DN50", "16 bar")]):
            for cell, value in zip(row.cells, values):
                cell.text = value
        source.add_paragraph("Scope text.")
    add_heading("Limits", 2)
    source.add_paragraph("Limits text.")
    add_heading("Terms", 1)
    source.add_paragraph("Terms text.")

    doc = _convert_docx(source, tmp_path)
    headings = [item for item in doc.texts if isinstance(item, SectionHeaderItem)]
    scope, limits, terms = headings
    prefix = ("1 ", "1.1 ", "2 ") if numbering_source != "none" else ("", "", "")
    assert [item.text for item in headings] == [
        f"{prefix[0]}Scope",
        f"{prefix[1]}Limits",
        f"{prefix[2]}Terms",
    ]
    by_text = {item.text: item for item in doc.texts}
    if following_block != "direct-list":
        assert by_text["Scope text."].parent == scope.get_ref()
    assert limits.parent == scope.get_ref()
    assert by_text["Limits text."].parent == limits.get_ref()
    assert terms.parent == scope.parent
    assert by_text["Terms text."].parent == terms.get_ref()
    if following_block in {"list", "direct-list"}:
        group = by_text["pumps"].parent.resolve(doc)
        assert isinstance(group, ListGroup)
        assert group.parent == scope.get_ref()
    elif following_block == "table":
        assert doc.tables[0].parent == scope.get_ref()
