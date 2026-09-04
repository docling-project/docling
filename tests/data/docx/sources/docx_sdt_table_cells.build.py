# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regenerate ``docx_sdt_table_cells.docx``.

Word emits this shape when a content control *contains* a table cell rather
than sitting inside the cell's paragraph: the ``w:sdt`` becomes a child of
``w:tr`` and the ``w:tc`` moves under ``w:sdtContent``. ``CT_Row.tc_lst`` then
reports fewer cells than the row really has, so the cell is dropped and the
remaining cells shift left.

Run with ``python-docx`` installed; writes the file next to this script.

Do not open and re-save the result in LibreOffice. Its docx export rewrites the
cell-level control as a run-level one inside the cell, which is a shape the
backend already handles, so the regression quietly disappears from the file.
"""

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

_ID = [1024]


def wrap_cell_in_content_control(cell, alias, tag):
    """Move `cell`'s w:tc under a new w:sdt/w:sdtContent in the same position."""
    tc = cell._tc
    tr = tc.getparent()
    idx = list(tr).index(tc)

    sdt = tr.makeelement(qn("w:sdt"), {})
    pr = sdt.makeelement(qn("w:sdtPr"), {})
    for name, attr, val in (
        ("w:alias", "w:val", alias),
        ("w:tag", "w:val", tag),
        ("w:id", "w:val", str(_ID[0])),
    ):
        el = sdt.makeelement(qn(name), {qn(attr): val})
        pr.append(el)
    _ID[0] += 1
    pr.append(sdt.makeelement(qn("w:text"), {}))
    sdt.append(pr)

    content = sdt.makeelement(qn("w:sdtContent"), {})
    tr.remove(tc)
    content.append(tc)
    sdt.append(content)
    tr.insert(idx, sdt)
    return sdt


doc = Document()

doc.add_paragraph("before")

# Case 1: the 1x1 furniture table Word uses for cover-page blocks.
t1 = doc.add_table(rows=1, cols=1)
t1.style = "Table Grid"
t1.rows[0].cells[0].text = "Cover value"
wrap_cell_in_content_control(t1.rows[0].cells[0], "Title", "coverTitle")

doc.add_paragraph("between")

# Case 2: a properties table where only the value column is a content control,
# and only one middle column is wrapped, so a grid shift shows up independently.
t2 = doc.add_table(rows=2, cols=3)
t2.style = "Table Grid"
rows = [("Owner", "A. Reviewer", "2026-09-04"), ("Status", "Draft", "2026-09-11")]
for r, vals in zip(t2.rows, rows):
    for c, v in zip(r.cells, vals):
        c.text = v
for r, tagname in zip(t2.rows, ("ownerValue", "statusValue")):
    wrap_cell_in_content_control(r.cells[1], "Value", tagname)

doc.add_paragraph("after")

doc.save(Path(__file__).with_name("docx_sdt_table_cells.docx"))
