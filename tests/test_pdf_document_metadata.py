# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for PDF document-level metadata: catalog /Lang, /Info and XMP properties."""

from pathlib import Path

import pytest
from docling_core.types.doc.common.meta import BaseMeta, LanguageMetaField
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import HumanLanguageLabel

from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.pdf_metadata import (
    LANGUAGE_SOURCE_PDF_CATALOG,
    _PdfDocumentMetadata,
    attach_document_metadata,
    parse_xmp_properties,
    pdf_date_to_iso8601,
)

# Catalog /Lang (en-US), an /Info dictionary and an XMP packet; built with pypdf.
METADATA_PDF = Path("./tests/data/pdf/document_metadata.pdf")
# No /Lang, no /Info, no XMP.
PLAIN_PDF = Path("./tests/data/pdf/text_with_vector_glyphs.pdf")

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
   <custom:ignored>outside the known namespaces</custom:ignored>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _backend_metadata(path: Path, backend) -> _PdfDocumentMetadata:
    in_doc = InputDocument(path_or_stream=path, format=InputFormat.PDF, backend=backend)
    try:
        return in_doc._backend.get_document_metadata()
    finally:
        in_doc._backend.unload()


def test_parse_xmp_properties_flattens_known_namespaces():
    props = parse_xmp_properties(XMP_PACKET)

    assert props == {
        "pdfuaid:part": "2",  # property written as an rdf:Description attribute
        "dc:title": "Fixture title",  # x-default alternative wins over document order
        "dc:creator": "Ada Lovelace, Charles Babbage",  # rdf:Seq joined
        "pdf:Producer": "docling-machine-test-fixture",  # whitespace stripped
    }
    assert "dc:subject" not in props  # empty array carries no value
    assert not any(key.startswith("custom:") for key in props)


def test_parse_xmp_properties_tolerates_malformed_packet():
    assert parse_xmp_properties("<x:xmpmeta><unclosed") == {}


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


@pytest.mark.parametrize(
    "backend", [ThreadedDoclingParseDocumentBackend, PyPdfiumDocumentBackend]
)
def test_backends_report_nothing_for_plain_pdf(backend):
    assert _backend_metadata(PLAIN_PDF, backend).is_empty()


def test_attach_document_metadata_fills_body_meta():
    doc = DoclingDocument(name="fixture")
    attach_document_metadata(
        doc,
        _PdfDocumentMetadata(
            language="en-US",
            properties={
                "dc:title": "Fixture title",
                "xmp:CreatorTool": "pypdf fixture builder",
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
    }
    # The declared language survives JSON serialization of the document.
    dumped = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert dumped.body.meta is not None
    assert dumped.body.meta.get_custom_part()["pdf__lang"] == "en-US"


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
