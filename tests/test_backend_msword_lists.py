# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for Word list numbering behaviour.

Kept separate from ``test_backend_msword.py`` so that file stays under the
repository's per-file line limit.
"""

from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DoclingDocument, DocumentOrigin, ListGroup, ListItem
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter

_DOCX_ROOT = Path("./tests/data/docx/sources")


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


def test_manage_list_structure_no_keyerror_when_use_level_exceeds_parents(tmp_path):
    """_manage_list_structure must not raise KeyError when use_level exceeds parents.

    The pathological state is:
      - parents has keys 0..11, with key 0 set to None (cleared) and others
        holding NodeItems from headings / earlier lists.
      - level_at_new_list is 11 (set by the previous list item of the same numId).
      - A second item for the same numId arrives with ilevel=2, which triggers
        the "New list sequence" branch and computes use_level = 11 + 2 = 13.
      - parents.get(12) is not a key at all → previously raised KeyError: 12.
    """

    docx_io = _make_empty_docx()
    docx_path = tmp_path / "empty.docx"
    docx_path.write_bytes(docx_io.getvalue())

    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename=docx_path.name,
    )
    backend = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=docx_path)

    out_doc = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            filename="test.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            binary_hash="0",
        ),
    )

    body_node = out_doc.body
    first_list_gr = out_doc.add_list_group(name="list", parent=body_node)
    backend.parents = dict.fromkeys(range(12), body_node)
    backend.parents[0] = None  # gap that makes _get_level() return 0
    backend.parents[11] = first_list_gr

    # Simulate history: the immediately previous item was numid=27, ilevel=2.
    backend.history = {
        "names": [None, "list"],
        "levels": [None, 11],
        "numids": [None, 27],
        "indents": [None, 2],
    }
    backend.level_at_new_list = 11
    backend.last_numid = 27

    # This must not raise KeyError.
    elem_ref, use_level = backend._manage_list_structure(
        doc=out_doc, numid=27, ilevel=2
    )

    assert isinstance(backend.parents.get(use_level), ListGroup)


def test_manage_list_structure_no_keyerror_open_indented_list_exceeds_parents(tmp_path):
    """_manage_list_structure must not raise KeyError in the "Open indented list" branch.

    The pathological state is:
      - parents has keys 0..11, with keys 0..10 holding body nodes and key 11
        holding a ListGroup.
      - level_at_new_list is 11, prev_indent is 2, ilevel is 4.
      - The "Open indented list" loop runs for i in range(14, 16), accessing
        parents[i - 1] where i - 1 = 13 is not a key in parents.
    """
    docx_path = tmp_path / "empty.docx"
    docx_path.write_bytes(_make_empty_docx().getvalue())

    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename=docx_path.name,
    )
    backend = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=docx_path)

    out_doc = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            filename="test.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            binary_hash="0",
        ),
    )

    body_node = out_doc.body
    first_list_gr = out_doc.add_list_group(name="list", parent=body_node)
    backend.parents = dict.fromkeys(range(12), body_node)
    backend.parents[11] = first_list_gr

    # Simulate history: same numId, previous item was at ilevel=2.
    backend.history = {
        "names": [None, "list"],
        "levels": [None, 11],
        "numids": [None, 27],
        "indents": [None, 2],
    }
    backend.level_at_new_list = 11
    backend.last_numid = 27

    elem_ref, use_level = backend._manage_list_structure(
        doc=out_doc, numid=27, ilevel=4
    )

    assert isinstance(backend.parents.get(use_level), ListGroup)


def test_list_markers_follow_num_fmt_end_to_end(tmp_path):
    """Lettered/roman markers must survive convert(), including plain-suffix lvlText.

    Word keeps the counter as an integer and the display form in ``w:numFmt``.
    A level whose ``lvlText`` is only ``%2)`` (no extra text after stripping
    placeholders/punctuation) must still take the lvlText path when numFmt is
    non-decimal, so ``lowerLetter`` renders ``a)`` rather than hierarchical
    ``1.a.``. Hierarchical decimal/letter/roman mixes stay on the fallback form.
    """

    doc = Document()
    numbering = doc.part.numbering_part.element

    def add_numbering(abstract_id: str, num_id: str, levels: list[tuple[str, str]]):
        abstract_num = OxmlElement("w:abstractNum")
        abstract_num.set(qn("w:abstractNumId"), abstract_id)
        for ilvl, (num_fmt, lvl_text) in enumerate(levels):
            lvl = OxmlElement("w:lvl")
            lvl.set(qn("w:ilvl"), str(ilvl))
            start = OxmlElement("w:start")
            start.set(qn("w:val"), "1")
            lvl.append(start)
            fmt = OxmlElement("w:numFmt")
            fmt.set(qn("w:val"), num_fmt)
            lvl.append(fmt)
            text_el = OxmlElement("w:lvlText")
            text_el.set(qn("w:val"), lvl_text)
            lvl.append(text_el)
            abstract_num.append(lvl)
        numbering.append(abstract_num)
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), num_id)
        ref = OxmlElement("w:abstractNumId")
        ref.set(qn("w:val"), abstract_id)
        num.append(ref)
        numbering.append(num)

    # Hierarchical mix: decimal / lowerLetter / upperRoman (fallback form).
    add_numbering(
        "500",
        "501",
        [("decimal", "%1."), ("lowerLetter", "%1.%2."), ("upperRoman", "%1.%2.%3.")],
    )
    # Plain-suffix lowerLetter: the issue's a) / b) case (lvlText "%1)").
    add_numbering("600", "601", [("lowerLetter", "%1)")])
    # Other non-decimal formats with punctuation-only templates.
    add_numbering("700", "701", [("upperLetter", "%1.")])
    add_numbering("800", "801", [("lowerRoman", "%1.")])
    add_numbering("900", "901", [("upperRoman", "%1.")])
    add_numbering("1000", "1001", [("decimalZero", "%1.")])

    def add_item(text: str, num_id: str, ilvl_val: int = 0):
        paragraph = doc.add_paragraph(text, style="List Paragraph")
        num_pr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), str(ilvl_val))
        num_pr.append(ilvl)
        num_id_elem = OxmlElement("w:numId")
        num_id_elem.set(qn("w:val"), num_id)
        num_pr.append(num_id_elem)
        paragraph._element.get_or_add_pPr().append(num_pr)

    add_item("top one", "501", 0)
    add_item("lettered first", "501", 1)
    add_item("lettered second", "501", 1)
    add_item("roman first", "501", 2)
    add_item("roman second", "501", 2)
    add_item("top two", "501", 0)

    add_item("plain suffix first", "601")
    add_item("plain suffix second", "601")

    add_item("upper letter", "701")
    add_item("lower roman", "801")
    add_item("upper roman", "901")
    add_item("zero pad", "1001")

    docx_path = tmp_path / "num_fmt_markers.docx"
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
        ("top one", "1."),
        ("lettered first", "1.a."),
        ("lettered second", "1.b."),
        ("roman first", "1.b.I."),
        ("roman second", "1.b.II."),
        ("top two", "2."),
        ("plain suffix first", "a)"),
        ("plain suffix second", "b)"),
        ("upper letter", "A."),
        ("lower roman", "i."),
        ("upper roman", "I."),
        ("zero pad", "01."),
    ]


def _docx_list_items(name: str):
    """Return the converted document and its list items."""
    docx_path = _DOCX_ROOT / f"{name}.docx"
    assert docx_path.exists()
    converted = (
        DocumentConverter(allowed_formats=[InputFormat.DOCX])
        .convert(docx_path)
        .document
    )
    return converted, [
        item for item, _ in converted.iterate_items() if isinstance(item, ListItem)
    ]


def _scenario_items(converted, list_items):
    """Group the fixture's "Item *" list items by their top-level list.

    The two #4185 scenarios append to the shared ``docx_lists.docx`` fixture,
    so the "Item *" items span two separate top-level lists. Group them by the
    root ListGroup each item belongs to and return the groups in document order.
    """
    groups: list = []
    for item in list_items:
        if not item.text.startswith("Item "):
            continue
        root = item
        while isinstance(root.parent.resolve(converted), ListGroup):
            root = root.parent.resolve(converted)
        root_ref = root.get_ref()
        for idx, (ref, _items) in enumerate(groups):
            if ref == root_ref:
                groups[idx][1].append(item)
                break
        else:
            groups.append((root_ref, [item]))
    return [items for _, items in groups]


def test_list_returning_to_starting_level_above_zero_keeps_items():
    """A list at levels 1, 2, 1 must not drop the item returning to level 1.

    Regression for #4185: a numbered list that starts at ``w:ilvl`` 1 (never
    touching level 0) used to lose the third item, because the level-1 slot
    between the list base and the level-2 sub-list group was left empty. The
    scenario lives in the ``docx_lists.docx`` fixture under the heading
    "List starting above indent level 0".
    """

    converted, list_items = _docx_list_items("docx_lists")
    groups = _scenario_items(converted, list_items)

    # First top-level list: the above-zero scenario, levels 1, 2, 1.
    above_zero = groups[0]
    assert [item.text for item in above_zero] == ["Item A", "Item B", "Item C"]

    # The item that returns to the starting level must rejoin the starting
    # level's ListGroup (the same one Item A lives in) -- not be dropped.
    group_of_a = above_zero[0].parent.resolve(converted)
    group_of_c = above_zero[2].parent.resolve(converted)
    assert isinstance(group_of_a, ListGroup)
    assert group_of_c.get_ref() == group_of_a.get_ref()

    # The deeper item must nest inside the same list (as a sub-group of the
    # outer ListGroup), not become a sibling top-level list.
    group_of_b = above_zero[1].parent.resolve(converted)
    assert isinstance(group_of_b, ListGroup)
    assert group_of_b.parent == group_of_a.get_ref()


def test_list_returning_to_starting_level_zero_still_works():
    """Control case: levels 0, 1, 2, 1 must keep working unchanged.

    Lives in the ``docx_lists.docx`` fixture under the heading
    "List starting at indent level 0".
    """

    converted, list_items = _docx_list_items("docx_lists")
    groups = _scenario_items(converted, list_items)

    # Second top-level list: the level-0 control scenario, levels 0, 1, 2, 1.
    level_zero = groups[1]
    assert [item.text for item in level_zero] == [
        "Item A",
        "Item B",
        "Item C",
        "Item D",
    ]

    # Item D (level 1) rejoins Item B's (level 1) group.
    group_of_b = level_zero[1].parent.resolve(converted)
    group_of_d = level_zero[3].parent.resolve(converted)
    assert group_of_d.get_ref() == group_of_b.get_ref()


def _make_empty_docx():
    """Return an in-memory .docx with no content."""

    buf = BytesIO()
    Document().save(buf)
    buf.seek(0)
    return buf
