# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for header and footer text that python-docx does not read as paragraph text."""

from io import BytesIO

import pytest
from docling_core.types.doc import ContentLayer, DoclingDocument, TextItem
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_HEADER_PARAGRAPHS = {
    "content control": (
        "<w:p {ns}><w:sdt><w:sdtPr><w:alias w:val='Title'/></w:sdtPr><w:sdtContent>"
        "<w:r><w:t>Quarterly Report</w:t></w:r></w:sdtContent></w:sdt></w:p>"
    ),
    "tracked insertion": (
        "<w:p {ns}><w:ins w:id='1' w:author='a'>"
        "<w:r><w:t>Quarterly Report</w:t></w:r></w:ins></w:p>"
    ),
    "smart tag": (
        "<w:p {ns}><w:smartTag w:uri='urn:x' w:element='y'>"
        "<w:r><w:t>Quarterly Report</w:t></w:r></w:smartTag></w:p>"
    ),
    "block content control": (
        "<w:sdt {ns}><w:sdtPr><w:docPartObj><w:docPartGallery w:val='Header'/>"
        "</w:docPartObj></w:sdtPr><w:sdtContent><w:p><w:r><w:t>Quarterly Report"
        "</w:t></w:r></w:p></w:sdtContent></w:sdt>"
    ),
}


def _convert(header_xml: str, *, footer: bool) -> DoclingDocument:
    docx = Document()
    docx.add_paragraph("Body text")
    part = docx.sections[0].footer if footer else docx.sections[0].header
    part.is_linked_to_previous = False
    element = part._element
    for child in list(element):
        element.remove(child)
    element.append(parse_xml(header_xml.format(ns=nsdecls("w"))))

    stream = BytesIO()
    docx.save(stream)
    stream.seek(0)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename="header.docx",
    )
    return in_doc._backend.convert()


@pytest.mark.parametrize("footer", [False, True], ids=["header", "footer"])
@pytest.mark.parametrize("kind", list(_HEADER_PARAGRAPHS))
def test_header_text_outside_direct_runs_is_kept(kind: str, footer: bool):
    """Text that sits in a content control, a tracked insertion or a smart tag counts.

    The body keeps this text. A header or footer whose only text is wrapped
    like this was dropped as empty, for example a title content control.
    """
    doc = _convert(_HEADER_PARAGRAPHS[kind], footer=footer)

    furniture = [
        item.text
        for item, _ in doc.iterate_items(
            included_content_layers={ContentLayer.FURNITURE}
        )
        if isinstance(item, TextItem) and item.text
    ]
    assert furniture == ["Quarterly Report"]
    group_names = {group.name for group in doc.groups}
    assert ("page footer" if footer else "page header") in group_names
