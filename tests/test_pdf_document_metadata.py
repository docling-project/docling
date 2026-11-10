# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for PDF document-level metadata: catalog /Lang, /Info and XMP properties."""

import logging
from pathlib import Path

import pypdfium2.raw as pdfium_c
import pytest
from docling_core.types.doc.common.meta import BaseMeta, LanguageMetaField
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import HumanLanguageLabel

from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.pdf_metadata import (
    LANGUAGE_SOURCE_PDF_CATALOG,
    _PdfDocumentMetadata,
    attach_document_metadata,
    extract_metadata_from_docling_parse,
    parse_xmp_properties,
    pdf_date_to_iso8601,
)

# Catalog /Lang (en-US), an /Info dictionary and an XMP packet; built with pypdf.
METADATA_PDF = Path("./tests/data/pdf/document_metadata.pdf")
# Same shape, with the PDF/UA identification (pdfuaid:part 2, pdfuaid:rev 2024) and two PDF
# Declarations (WTPDF accessibility 1.0, ISO/TS 32005) in the XMP packet; built with pypdf.
DECLARATIONS_PDF = Path("./tests/data/pdf/document_metadata_declarations.pdf")
# No /Lang, no /Info, no XMP.
PLAIN_PDF = Path("./tests/data/pdf/text_with_vector_glyphs.pdf")
# An image input: its backend has no PDF catalog to read.
PLAIN_IMAGE = Path("./tests/data/ocr/sources/rotated_180_dense_text.png")

XMP_PACKET = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:pdf="http://ns.adobe.com/pdf/1.3/"
    xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/"
    xmlns:custom="http://example.com/ns/unknown/"
    pdfuaid:part="2">
   <dc:title><rdf:Alt>
     <rdf:li xml:lang="de">Titel</rdf:li>
     <rdf:li xml:lang="x-default">Fixture title</rdf:li>
   </rdf:Alt></dc:title>
   <dc:creator><rdf:Seq><rdf:li>Ada Lovelace</rdf:li><rdf:li>Charles Babbage</rdf:li></rdf:Seq></dc:creator>
   <dc:subject><rdf:Bag></rdf:Bag></dc:subject>
   <pdf:Producer> docling-machine-test-fixture </pdf:Producer>
   <pdfuaid:rev>2024</pdfuaid:rev>
   <custom:ignored>outside the known namespaces</custom:ignored>
   <unqualified>no namespace at all</unqualified>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

# PDF Declarations (WTPDF 1.0 clause 6.1; ISO/TS 32005 clause 4): a bag of structures, each
# with a pdfd:conformsTo URI and optional pdfd:claimData. The third one uses the XMP attribute
# shorthand for its conformsTo.
DECLARATIONS_PACKET = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/"
    xmlns:pdfd="http://pdfa.org/declarations/">
   <pdfuaid:part>2</pdfuaid:part>
   <pdfd:declarations><rdf:Bag>
    <rdf:li rdf:parseType="Resource">
     <pdfd:conformsTo>http://pdfa.org/declarations/wtpdf/#accessibility1.0</pdfd:conformsTo>
     <pdfd:claimData><rdf:Bag><rdf:li rdf:parseType="Resource">
      <pdfd:claimBy>Example Corp</pdfd:claimBy>
      <pdfd:claimDate>2019-01-17</pdfd:claimDate>
     </rdf:li></rdf:Bag></pdfd:claimData>
    </rdf:li>
    <rdf:li rdf:parseType="Resource">
     <pdfd:conformsTo>https://pdfa.org/declarations#iso32005</pdfd:conformsTo>
    </rdf:li>
    <rdf:li pdfd:conformsTo="http://pdfa.org/declarations/wtpdf/#reuse1.0"/>
   </rdf:Bag></pdfd:declarations>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def _backend_metadata(
    path: Path, backend, input_format: InputFormat = InputFormat.PDF
) -> _PdfDocumentMetadata:
    in_doc = InputDocument(path_or_stream=path, format=input_format, backend=backend)
    try:
        return in_doc._backend.get_document_metadata()
    finally:
        in_doc._backend.unload()


def test_parse_xmp_properties_flattens_known_namespaces():
    props = parse_xmp_properties(XMP_PACKET)

    assert props == {
        "pdfuaid:part": "2",  # property written as an rdf:Description attribute
        "pdfuaid:rev": "2024",  # PDF/UA identification, simple element property
        "dc:title": "Fixture title",  # x-default alternative wins over document order
        "dc:creator": "Ada Lovelace, Charles Babbage",  # rdf:Seq joined
        "pdf:Producer": "docling-machine-test-fixture",  # whitespace stripped
    }
    assert "dc:subject" not in props  # empty array carries no value
    # properties in an unknown namespace, or in none, are skipped
    assert not any(key.startswith("custom:") for key in props)
    assert not any("unqualified" in key for key in props)


def test_parse_xmp_properties_collects_pdf_declarations():
    props = parse_xmp_properties(DECLARATIONS_PACKET)

    assert props == {
        "pdfuaid:part": "2",
        # every conformsTo URI under the declarations bag, verbatim, in document order
        "pdfd:conformsTo": (
            "http://pdfa.org/declarations/wtpdf/#accessibility1.0, "
            "https://pdfa.org/declarations#iso32005, "
            "http://pdfa.org/declarations/wtpdf/#reuse1.0"
        ),
    }
    # the claim structures (who, when) are not flattened into properties
    assert not any(key.startswith("pdfd:claim") for key in props)


# The same declarations in the general RDF form: a nested rdf:Description instead of
# rdf:parseType="Resource", a claim structure with its own description, one URI declared twice,
# and a declaration bag nested inside a claim structure (not a declaration of the document).
NESTED_DECLARATIONS_PACKET = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:pdfd="http://pdfa.org/declarations/">
   <pdfd:declarations><rdf:Bag>
    <rdf:li><rdf:Description>
     <pdfd:conformsTo>http://pdfa.org/declarations/wtpdf/#accessibility1.0</pdfd:conformsTo>
     <pdfd:claimData><rdf:Bag><rdf:li><rdf:Description>
      <pdfd:claimBy>Example Corp</pdfd:claimBy>
      <pdfd:declarations><rdf:Bag>
       <rdf:li pdfd:conformsTo="http://example.com/not-a-document-declaration"/>
      </rdf:Bag></pdfd:declarations>
     </rdf:Description></rdf:li></rdf:Bag></pdfd:claimData>
    </rdf:Description></rdf:li>
    <rdf:li><rdf:Description pdfd:conformsTo="https://pdfa.org/declarations#iso32005"/></rdf:li>
    <rdf:li rdf:parseType="Resource">
     <pdfd:conformsTo>http://pdfa.org/declarations/wtpdf/#accessibility1.0</pdfd:conformsTo>
    </rdf:li>
   </rdf:Bag></pdfd:declarations>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def test_parse_xmp_properties_reads_only_the_declaration_structures():
    props = parse_xmp_properties(NESTED_DECLARATIONS_PACKET)

    # each document declaration once, in document order; the claim structures and the bag
    # nested inside one of them contribute nothing
    assert props == {
        "pdfd:conformsTo": (
            "http://pdfa.org/declarations/wtpdf/#accessibility1.0, "
            "https://pdfa.org/declarations#iso32005"
        ),
    }

    # a hostile packet cannot grow the value without bound: the list is capped
    members = "".join(
        f'<rdf:li pdfd:conformsTo="http://example.com/claim/{i}"/>' for i in range(40)
    )
    flood = NESTED_DECLARATIONS_PACKET.replace(
        "<pdfd:declarations><rdf:Bag>\n    <rdf:li><rdf:Description>",
        f"<pdfd:declarations><rdf:Bag>{members}\n    <rdf:li><rdf:Description>",
        1,
    )
    assert flood != NESTED_DECLARATIONS_PACKET
    assert len(parse_xmp_properties(flood)["pdfd:conformsTo"].split(", ")) == 16


# A declarations bag that does not follow the schema: a conformsTo directly under
# pdfd:declarations (no array), an array member that is not rdf:li, and a declaration that
# carries claim data but no conformsTo. Only the last member is a usable declaration.
MALFORMED_DECLARATIONS_PACKET = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:pdfd="http://pdfa.org/declarations/">
   <pdfd:declarations>
    <pdfd:conformsTo>http://example.com/outside-the-bag</pdfd:conformsTo>
    <rdf:Bag>
     <rdf:Description pdfd:conformsTo="http://example.com/not-a-list-member"/>
     <rdf:li rdf:parseType="Resource">
      <pdfd:claimData><rdf:Bag><rdf:li rdf:parseType="Resource">
       <pdfd:claimBy>Example Corp</pdfd:claimBy>
      </rdf:li></rdf:Bag></pdfd:claimData>
     </rdf:li>
     <rdf:li pdfd:conformsTo="https://pdfa.org/declarations#iso32005"/>
    </rdf:Bag>
   </pdfd:declarations>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def test_parse_xmp_properties_skips_malformed_declarations():
    assert parse_xmp_properties(MALFORMED_DECLARATIONS_PACKET) == {
        "pdfd:conformsTo": "https://pdfa.org/declarations#iso32005"
    }


def test_parse_xmp_properties_tolerates_malformed_packet():
    assert parse_xmp_properties("<x:xmpmeta><unclosed") == {}


def test_parse_xmp_properties_rejects_entity_declarations(caplog):
    # defusedxml raises EntitiesForbidden (a ValueError, not a ParseError) for a packet
    # that declares entities; it must be swallowed like any other unusable packet.
    hostile = (
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY lol "lol">]>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>&lol;</dc:title></rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    with caplog.at_level(logging.WARNING, logger="docling.utils.pdf_metadata"):
        assert parse_xmp_properties(hostile) == {}
    assert "rejected by defusedxml" in caplog.text


@pytest.mark.parametrize(
    ("pdf_date", "expected"),
    [
        ("D:20261012100000Z", "2026-10-12T10:00:00Z"),
        ("D:20260625130811+08'00'", "2026-06-25T13:08:11+08:00"),
        ("D:20260625130811-05'30", "2026-06-25T13:08:11-05:30"),
        ("D:202610", "2026-10"),
        ("2026-10-12T10:00:00Z", "2026-10-12T10:00:00Z"),  # already ISO 8601
        ("yesterday", "yesterday"),  # not a date: returned as is
    ],
)
def test_pdf_date_to_iso8601(pdf_date: str, expected: str):
    assert pdf_date_to_iso8601(pdf_date) == expected


def test_docling_parse_backend_reads_lang_and_xmp():
    metadata = _backend_metadata(METADATA_PDF, ThreadedDoclingParseDocumentBackend)

    assert metadata.language == "en-US"
    assert metadata.properties["dc:title"] == "Fixture title"
    assert metadata.properties["dc:creator"] == "Ada Lovelace, Charles Babbage"
    assert metadata.properties["pdf:Producer"] == "docling-machine-test-fixture"
    assert metadata.properties["xmp:CreateDate"] == "2026-10-12T10:00:00Z"
    assert metadata.properties["pdfuaid:part"] == "2"


def test_docling_parse_backend_reads_conformance_declarations():
    metadata = _backend_metadata(DECLARATIONS_PDF, ThreadedDoclingParseDocumentBackend)

    assert metadata.properties["pdfuaid:part"] == "2"
    assert metadata.properties["pdfuaid:rev"] == "2024"
    assert metadata.properties["pdfd:conformsTo"] == (
        "http://pdfa.org/declarations/wtpdf/#accessibility1.0, "
        "https://pdfa.org/declarations#iso32005"
    )
    # pypdfium2 reads /Info only, so the XMP-only conformance keys are absent there.
    pdfium_metadata = _backend_metadata(DECLARATIONS_PDF, PyPdfiumDocumentBackend)
    assert pdfium_metadata.properties["dc:title"] == "Declarations fixture"
    assert not any(
        key.startswith(("pdfuaid:", "pdfd:")) for key in pdfium_metadata.properties
    )


def test_pypdfium_backend_reads_lang_and_info():
    metadata = _backend_metadata(METADATA_PDF, PyPdfiumDocumentBackend)

    assert metadata.language == "en-US"
    # /Info keys are reported under the XMP names they pair with, dates as ISO 8601.
    assert metadata.properties == {
        "dc:title": "Fixture title",
        "dc:creator": "Ada Lovelace",
        "pdf:Producer": "docling-machine-test-fixture",
        "xmp:CreateDate": "2026-10-12T10:00:00Z",
    }


def test_pypdfium_backend_reads_info_without_catalog_language_api(monkeypatch):
    # pypdfium2 4.30.0, the oldest release docling accepts, ships a pdfium build without
    # FPDFCatalog_GetLanguage. The language is then unknown; the /Info properties still read.
    monkeypatch.delattr(pdfium_c, "FPDFCatalog_GetLanguage")

    metadata = _backend_metadata(METADATA_PDF, PyPdfiumDocumentBackend)

    assert metadata.language is None
    assert metadata.properties["dc:title"] == "Fixture title"


@pytest.mark.parametrize(
    ("path", "input_format", "backend"),
    [
        (PLAIN_PDF, InputFormat.PDF, ThreadedDoclingParseDocumentBackend),
        (PLAIN_PDF, InputFormat.PDF, PyPdfiumDocumentBackend),
        # no catalog at all: the pipeline still asks the backend when the option is on
        (PLAIN_IMAGE, InputFormat.IMAGE, ImageDocumentBackend),
    ],
)
def test_backends_report_nothing_without_metadata(path, input_format, backend):
    assert _backend_metadata(path, backend, input_format).is_empty()


def test_extractors_report_nothing_without_a_native_document():
    # docling-parse returns no annotations object for a document that has none.
    assert extract_metadata_from_docling_parse(None).is_empty()

    # A closed pypdfium2 backend has released its native document.
    in_doc = InputDocument(
        path_or_stream=METADATA_PDF,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    in_doc._backend.unload()
    assert in_doc._backend.get_document_metadata().is_empty()


def test_attach_document_metadata_fills_body_meta():
    doc = DoclingDocument(name="fixture")
    attach_document_metadata(
        doc,
        _PdfDocumentMetadata(
            language="en-US",
            properties={
                "dc:title": "Fixture title",
                "xmp:CreatorTool": "pypdf fixture builder",
                "pdfd:conformsTo": "http://pdfa.org/declarations/wtpdf/#accessibility1.0",
            },
        ),
    )

    meta = doc.body.meta
    assert meta is not None
    assert meta.language == LanguageMetaField(
        code=HumanLanguageLabel.EN, created_by=LANGUAGE_SOURCE_PDF_CATALOG
    )
    assert meta.get_custom_part() == {
        "pdf__lang": "en-US",
        "dc__title": "Fixture title",
        "xmp__creator_tool": "pypdf fixture builder",
        "pdfd__conforms_to": "http://pdfa.org/declarations/wtpdf/#accessibility1.0",
    }
    # The declared language survives JSON serialization of the document.
    dumped = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert dumped.body.meta is not None
    assert dumped.body.meta.get_custom_part()["pdf__lang"] == "en-US"


def test_attach_document_metadata_without_declared_language():
    # Properties without a catalog /Lang: no language field and no pdf__lang.
    doc = DoclingDocument(name="fixture")
    attach_document_metadata(
        doc, _PdfDocumentMetadata(properties={"dc:title": "Fixture title"})
    )

    assert doc.body.meta is not None
    assert doc.body.meta.language is None
    assert doc.body.meta.get_custom_part() == {"dc__title": "Fixture title"}


def test_attach_document_metadata_keeps_full_tag_when_subtag_unknown():
    doc = DoclingDocument(name="fixture")
    attach_document_metadata(doc, _PdfDocumentMetadata(language="x-private"))

    assert doc.body.meta is not None
    assert doc.body.meta.language is None
    assert doc.body.meta.get_custom_part() == {"pdf__lang": "x-private"}


def test_attach_document_metadata_does_not_overwrite_existing_meta():
    doc = DoclingDocument(name="fixture")
    detected = LanguageMetaField(code=HumanLanguageLabel.DE, created_by="langdetect")
    doc.body.meta = BaseMeta(language=detected, dc__title="Kept")
    attach_document_metadata(
        doc,
        _PdfDocumentMetadata(language="en-US", properties={"dc:title": "Replaced?"}),
    )

    assert doc.body.meta.language == detected
    assert doc.body.meta.get_custom_part() == {
        "dc__title": "Kept",
        "pdf__lang": "en-US",
    }


def test_attach_document_metadata_leaves_body_meta_unset_when_empty():
    doc = DoclingDocument(name="fixture")
    attach_document_metadata(doc, _PdfDocumentMetadata())

    assert doc.body.meta is None
