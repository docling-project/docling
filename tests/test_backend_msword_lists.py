"""Tests for Word list numbering behaviour.

Kept separate from ``test_backend_msword.py`` so that file stays under the
repository's per-file line limit.
"""

from docling_core.types.doc import ListItem
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def test_ordered_list_resumes_numbering_after_intervening_list(tmp_path):
    """An ordered list interrupted by a bullet list must keep counting.

    Word numbers continuously per ``w:numId``, so items sharing one numId are a
    single list even when other content (including a list with a different
    numId) sits between them. Docling used to reset the counter whenever the
    numId changed, so the resumed items restarted at 1.
    """

    doc = Document()
    numbering = doc.part.numbering_part.element

    def add_numbering(abstract_id: str, num_id: str, num_fmt: str, lvl_text: str):
        abstract_num = OxmlElement("w:abstractNum")
        abstract_num.set(qn("w:abstractNumId"), abstract_id)
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        lvl.append(start)
        fmt = OxmlElement("w:numFmt")
        fmt.set(qn("w:val"), num_fmt)
        lvl.append(fmt)
        text = OxmlElement("w:lvlText")
        text.set(qn("w:val"), lvl_text)
        lvl.append(text)
        abstract_num.append(lvl)
        numbering.append(abstract_num)

        num = OxmlElement("w:num")
        num.set(qn("w:numId"), num_id)
        ref = OxmlElement("w:abstractNumId")
        ref.set(qn("w:val"), abstract_id)
        num.append(ref)
        numbering.append(num)

    add_numbering("300", "301", "decimal", "%1.")
    add_numbering("400", "401", "bullet", "•")

    def add_item(text: str, num_id: str):
        paragraph = doc.add_paragraph(text, style="List Paragraph")
        num_pr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        num_pr.append(ilvl)
        num_id_elem = OxmlElement("w:numId")
        num_id_elem.set(qn("w:val"), num_id)
        num_pr.append(num_id_elem)
        paragraph._element.get_or_add_pPr().append(num_pr)

    add_item("First ordered item", "301")
    add_item("Second ordered item", "301")
    add_item("bullet one", "401")
    add_item("bullet two", "401")
    add_item("Third ordered item", "301")

    docx_path = tmp_path / "resumed_ordered_list.docx"
    doc.save(str(docx_path))

    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename=docx_path.name,
    )
    converted = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=docx_path).convert()

    markers = [
        (item.text, item.marker)
        for item, _ in converted.iterate_items()
        if isinstance(item, ListItem)
    ]

    assert markers == [
        ("First ordered item", "1."),
        ("Second ordered item", "2."),
        ("bullet one", ""),
        ("bullet two", ""),
        ("Third ordered item", "3."),
    ]
