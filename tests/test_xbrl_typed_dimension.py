"""Regression tests for typed-dimension facts in the XBRL backend.

A typed dimension (``xbrldi:typedMember``) has no member QName by design —
arelle leaves ``ModelDimensionValue.memberQname`` ``None`` and carries the
value on the ``typedMember`` element instead.  The backend dereferenced
``memberQname.localName`` unconditionally, so every conversion of a valid
typed-dimension fact crashed with
``AttributeError: 'NoneType' object has no attribute 'localName'`` (issue #4437).

The fixtures mirror the minimal synthetic report from the issue; arelle
resolves the standard XBRL 2.1 schema imports from its bundled copies.
"""

from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.xml.xbrl_backend import XBRLDocumentBackend
from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

XSD = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldt="http://xbrl.org/2005/xbrldt" xmlns:t="https://example.invalid/o20/typed" targetNamespace="https://example.invalid/o20/typed" elementFormDefault="qualified">
  <xs:import namespace="http://www.xbrl.org/2003/instance" schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>
  <xs:element name="Revenue" id="t_Revenue" type="xbrli:monetaryItemType" substitutionGroup="xbrli:item" xbrli:periodType="duration"/>
  <xs:element name="RegionAxis" id="t_RegionAxis" type="xbrli:stringItemType" substitutionGroup="xbrli:item" xbrli:periodType="duration" abstract="true" xbrldt:typedDomainRef="typed.xsd#t_RegionDomain"/>
  <xs:element name="RegionDomain" id="t_RegionDomain" type="xs:string"/>
</xs:schema>
"""

XML = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:link="http://www.xbrl.org/2003/linkbase" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:iso4217="http://www.xbrl.org/2003/iso4217" xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:t="https://example.invalid/o20/typed">
<link:schemaRef xlink:type="simple" xlink:href="typed.xsd"/>
<xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="https://example.invalid/entity">Entity</xbrli:identifier><xbrli:segment><xbrldi:typedMember dimension="t:RegionAxis"><t:RegionDomain>North</t:RegionDomain></xbrldi:typedMember></xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:unit id="u1"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
<t:Revenue contextRef="c1" unitRef="u1" decimals="0">123</t:Revenue>
</xbrli:xbrl>
"""


@pytest.fixture
def taxonomy_dir(tmp_path: Path) -> Path:
    (tmp_path / "typed.xsd").write_text(XSD, encoding="utf-8")
    (tmp_path / "typed.xml").write_text(XML, encoding="utf-8")
    return tmp_path


def _convert(taxonomy_dir: Path):
    xml_path = taxonomy_dir / "typed.xml"
    options = XBRLBackendOptions(
        taxonomy=taxonomy_dir,
        enable_local_fetch=True,
        enable_remote_fetch=False,
    )
    in_doc = InputDocument(
        path_or_stream=xml_path,
        format=InputFormat.XML_XBRL,
        backend=XBRLDocumentBackend,
        backend_options=options,
        filename="typed.xml",
    )
    backend = XBRLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=xml_path,
        options=options,
    )
    doc = backend.convert()
    return doc, backend


def test_typed_dimension_fact_converts(taxonomy_dir: Path) -> None:
    """A valid typed-dimension instance must convert without crashing.

    Before the fix, ``convert()`` raised
    ``AttributeError: 'NoneType' object has no attribute 'localName'``
    while rendering the fact's dimension cells.
    """
    doc, backend = _convert(taxonomy_dir)

    # The fact itself is still rendered into the key-value cells.
    cell_texts = [cell.text for cell in backend._cells]
    assert "Revenue" in cell_texts
    assert "value: 123" in cell_texts
    assert doc.name == "typed"
