# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression test for issue #4185: DOCX list starting at indent level > 0.

A numbered Word list whose items sit at ``w:ilvl`` 1, 2, 1 (i.e. it never
touches indent level 0) used to drop the item that returns to the *starting*
level: the backend logged
"Parent element of the list item is not a ListGroup. The list item will be ignored."
and discarded the paragraph. Only a list that starts at level 0 was handled.

Kept in a standalone file so tests stay consistent with the other Word list
files (``test_backend_msword_lists.py``, ``test_backend_msword_spacer.py``).
"""

from io import BytesIO

from docling_core.types.doc import DoclingDocument, DocumentOrigin, ListGroup, ListItem
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
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


def _build_docx_with_levels(levels: list[int], abstract_id: str, num_id: str):
    doc = Document()
    _make_multilevel_numbering(doc, abstract_id=abstract_id, num_id=num_id)
    names = ["Item A", "Item B", "Item C", "Item D"]
    for text, ilvl in zip(names, levels):
        _add_numbered_paragraph(doc, text, num_id, ilvl)
    return doc


def test_list_returning_to_starting_level_above_zero_keeps_items(tmp_path):
    """A list at levels 1, 2, 1 must not drop the item returning to level 1.

    Regression for #4185: a numbered list that starts at ``w:ilvl`` 1 (never
    touching level 0) used to lose the third item, because the level-1 slot
    between the list base and the level-2 sub-list group was left empty.
    """

    docx_path = tmp_path / "list_starting_at_level_1.docx"
    _build_docx_with_levels([1, 2, 1], abstract_id="900", num_id="901").save(
        str(docx_path)
    )

    converted = DocumentConverter(allowed_formats=[InputFormat.DOCX]).convert(
        docx_path
    ).document

    # All three items must survive, in order.
    assert [t.text for t in converted.texts] == ["Item A", "Item B", "Item C"]

    list_items = [
        item for item, _ in converted.iterate_items() if isinstance(item, ListItem)
    ]
    assert [item.text for item in list_items] == ["Item A", "Item B", "Item C"]

    # The item that returns to the starting level must rejoin the starting
    # level's ListGroup (the same one Item A lives in) -- not be dropped.
    group_of_a = list_items[0].parent.resolve(converted)
    group_of_c = list_items[2].parent.resolve(converted)
    assert isinstance(group_of_a, ListGroup)
    assert group_of_c.get_ref() == group_of_a.get_ref()

    # The deeper item must nest inside the same list (as a sub-group of the
    # outer ListGroup), not become a sibling top-level list -- which is what
    # the bug produced for Item A/Item B.
    group_of_b = list_items[1].parent.resolve(converted)
    assert isinstance(group_of_b, ListGroup)
    assert group_of_b.parent == group_of_a.get_ref()

    # The markdown must reflect a single nested list, not two separate lists.
    markdown = converted.export_to_markdown()
    assert "Item C" in markdown
    # Item B is rendered indented under Item A; Item C returns to the top level.
    lines = [line for line in markdown.splitlines() if line.strip()]
    assert lines[0].startswith("- ") and "Item A" in lines[0]
    assert lines[1].startswith("    - ") and "Item B" in lines[1]
    assert lines[2].startswith("- ") and "Item C" in lines[2]


def test_list_returning_to_starting_level_zero_still_works(tmp_path):
    """Control case: levels 0, 1, 2, 1 must keep working unchanged."""

    docx_path = tmp_path / "list_starting_at_level_0.docx"
    _build_docx_with_levels([0, 1, 2, 1], abstract_id="902", num_id="903").save(
        str(docx_path)
    )

    converted = DocumentConverter(allowed_formats=[InputFormat.DOCX]).convert(
        docx_path
    ).document

    assert [t.text for t in converted.texts] == [
        "Item A",
        "Item B",
        "Item C",
        "Item D",
    ]

    list_items = [
        item for item, _ in converted.iterate_items() if isinstance(item, ListItem)
    ]
    # Item D (level 1) rejoins Item B's (level 1) group.
    group_of_b = list_items[1].parent.resolve(converted)
    group_of_d = list_items[3].parent.resolve(converted)
    assert group_of_d.get_ref() == group_of_b.get_ref()


def test_list_starting_at_level_one_branch_structure():
    """Unit-level guard: _manage_list_structure slot bookkeeping for 1, 2, 1.

    The bug was that opening a list at Word level 1 placed the first ListGroup
    at slot 0, while the deeper level computed slot 0 + 2 = 2, leaving slot 1
    as a hole that a return to level 1 then tried to use. With the fix, level 1
    maps back to slot 0 and level 2 maps to slot 1.
    """

    buf = BytesIO()
    _build_docx_with_levels([1, 2, 1], abstract_id="904", num_id="905").save(buf)
    buf.seek(0)

    in_doc = InputDocument(
        path_or_stream=BytesIO(buf.getvalue()),
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename="list_starting_at_level_1.docx",
    )
    backend = MsWordDocumentBackend(
        in_doc=in_doc, path_or_stream=BytesIO(buf.getvalue())
    )

    out_doc = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            filename="test.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            binary_hash="0",
        ),
    )

    # First item at Word level 1 opens the list.
    elem_ref, use_level = backend._manage_list_structure(
        doc=out_doc, numid=905, ilevel=1
    )
    backend.history = {
        "names": [None, "list"],
        "levels": [None, 0],
        "numids": [None, 905],
        "indents": [None, 1],
    }
    assert isinstance(backend.parents[use_level], ListGroup)
    base_level = use_level

    # Deeper item at Word level 2 must not leave a hole below its slot.
    _, use_level = backend._manage_list_structure(doc=out_doc, numid=905, ilevel=2)
    backend.history["indents"].append(2)
    assert use_level == base_level + 1
    assert isinstance(backend.parents[use_level], ListGroup)

    # Returning to Word level 1 must land on the starting list group.
    _, use_level = backend._manage_list_structure(doc=out_doc, numid=905, ilevel=1)
    assert use_level == base_level
    assert isinstance(backend.parents[use_level], ListGroup)