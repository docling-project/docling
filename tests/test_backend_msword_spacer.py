# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for Word list handling around blank spacer paragraphs.

Kept separate from ``test_backend_msword.py`` so that file stays under the
repository's per-file line limit.
"""

from docling_core.types.doc import ListItem
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


def _make_multilevel_numbering(doc, abstract_id: str, num_id: str, levels: int = 3):
    """Register a multi-level decimal numbering definition in a docx."""
    numbering = doc.part.numbering_part.element

    abstract_num = OxmlElement("w:abstractNum")
    abstract_num.set(qn("w:abstractNumId"), abstract_id)
    for ilvl in range(levels):
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), str(ilvl))
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        lvl.append(start)
        numfmt = OxmlElement("w:numFmt")
        numfmt.set(qn("w:val"), "decimal")
        lvl.append(numfmt)
        lvltext = OxmlElement("w:lvlText")
        lvltext.set(qn("w:val"), ".".join(f"%{i + 1}" for i in range(ilvl + 1)))
        lvl.append(lvltext)
        abstract_num.append(lvl)
    numbering.append(abstract_num)

    num_elem = OxmlElement("w:num")
    num_elem.set(qn("w:numId"), num_id)
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), abstract_id)
    num_elem.append(abstract_ref)
    numbering.append(num_elem)


def _add_numbered_paragraph(doc, text: str, num_id: str, ilvl: int):
    paragraph = doc.add_paragraph(text)
    num_pr = OxmlElement("w:numPr")
    ilvl_elem = OxmlElement("w:ilvl")
    ilvl_elem.set(qn("w:val"), str(ilvl))
    num_pr.append(ilvl_elem)
    num_id_elem = OxmlElement("w:numId")
    num_id_elem.set(qn("w:val"), num_id)
    num_pr.append(num_id_elem)
    paragraph._p.get_or_add_pPr().append(num_pr)
    return paragraph


def test_empty_paragraph_between_list_items_keeps_body_text_in_place(tmp_path):
    """A blank spacer paragraph must not strand the body text after it.

    Authors commonly press Enter between list items for vertical spacing. Such
    an empty paragraph closes the list without clearing the cached list group,
    so the next item used to re-open that group -- which sits before the body
    text in between -- and the text ended up after the whole list instead of
    where the author put it.
    """

    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    def build(with_spacer: bool) -> list[str]:
        doc = Document()
        _make_multilevel_numbering(doc, abstract_id="700", num_id="701")

        _add_numbered_paragraph(doc, "First section", "701", 0)
        _add_numbered_paragraph(doc, "Sub one", "701", 1)
        if with_spacer:
            doc.add_paragraph("")
        doc.add_paragraph("Prose that belongs under Sub one.")
        _add_numbered_paragraph(doc, "Sub two", "701", 1)
        _add_numbered_paragraph(doc, "Second section", "701", 0)

        name = "with_spacer" if with_spacer else "without_spacer"
        docx_path = tmp_path / f"{name}.docx"
        doc.save(str(docx_path))
        markdown = converter.convert(docx_path).document.export_to_markdown()
        return [line for line in markdown.splitlines() if line.strip()]

    lines = build(with_spacer=True)
    # The spacer changes nothing about where the content ends up.
    assert lines == build(with_spacer=False)

    prose = lines.index("Prose that belongs under Sub one.")
    sub_one = next(i for i, line in enumerate(lines) if line.endswith("Sub one"))
    sub_two = next(i for i, line in enumerate(lines) if line.endswith("Sub two"))
    assert sub_one < prose < sub_two


def test_spacers_before_a_resumed_list_item_in_a_table_cell_keep_the_table(tmp_path):
    """Spacers before a resumed list item in a table cell keep the whole table."""
    doc = Document()
    table = doc.add_table(rows=3, cols=1)
    table.cell(0, 0).paragraphs[0].text = "row zero"
    cell = table.cell(1, 0)
    cell.paragraphs[0].text = "steps"
    for text in ("first", "second", "", "", "third"):
        cell.add_paragraph(text, style="List Number" if text else None)
    table.cell(2, 0).paragraphs[0].text = "row two"
    doc.add_paragraph("after the table")
    docx_path = tmp_path / "resumed_in_cell.docx"
    doc.save(str(docx_path))

    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])
    converted = converter.convert(docx_path).document

    table_item = converted.tables[0]
    rows = {
        table_cell.start_row_offset_idx for table_cell in table_item.data.table_cells
    }
    assert rows == {0, 1, 2}
    items = {item.text: item for item in converted.texts if isinstance(item, ListItem)}
    markers = [items[text].marker for text in ("first", "second", "third")]
    assert markers == ["1.", "2.", "3."]
    list_group = items["first"].parent.resolve(converted)
    assert items["third"].parent == list_group.get_ref()
    assert list_group.parent.resolve(converted).parent == table_item.get_ref()
    assert all(item.text.strip() for item in converted.texts)


def test_text_after_a_resumed_list_keeps_only_its_own_comment(tmp_path):
    """Comments on spacers before a resumed list item do not move to later text."""
    doc = Document()
    doc.add_paragraph("first", style="List Number")
    spacer_runs = [doc.add_paragraph().add_run("") for _ in range(2)]
    doc.add_paragraph("second", style="List Number")
    note = doc.add_paragraph("note")
    for run in spacer_runs:
        doc.add_comment(run, text="on a spacer")
    doc.add_comment(note.runs, text="on the note")
    docx_path = tmp_path / "commented_spacers.docx"
    doc.save(str(docx_path))

    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])
    converted = converter.convert(docx_path).document

    note_item = next(item for item in converted.texts if item.text == "note")
    comment_texts = [
        ref.resolve(converted).children[0].resolve(converted).text
        for ref in note_item.comments
    ]
    assert len(comment_texts) == 1
    assert comment_texts[0].endswith("on the note")
