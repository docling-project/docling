# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Extract a PDF's document-level metadata: the declared language and the document properties.

A PDF declares its language in the catalog ``/Lang`` entry (ISO 32000-2, 14.9.2) and its
properties in two places: the ``/Info`` dictionary (14.3.3) and the XMP packet in the catalog
``/Metadata`` stream (14.3.2). Two extractors are provided so each backend uses its own native
capability:

* :func:`extract_metadata_from_pdfium` -- for the pypdfium2 backend. Reads ``/Lang`` and the
  ``/Info`` dictionary; pypdfium2 does not expose the XMP packet.
* :func:`extract_metadata_from_docling_parse` -- for the docling-parse backend. Reads ``/Lang``
  and the XMP packet from its native document annotations; docling-parse does not expose the
  ``/Info`` dictionary.

Properties from either source are keyed by their XMP qualified name (``dc:title``,
``pdf:Producer``, ``xmp:CreateDate``); ``/Info`` keys are mapped to the XMP names ISO 32000-2
pairs them with (Table 349), so both backends report the same vocabulary.

:func:`attach_document_metadata` records the result on ``DoclingDocument.body.meta``, the
document-level meta slot.

``pypdfium2`` is imported lazily, inside the function that uses it, never at module level:
``datamodel.document`` imports this module for the ``_PdfDocumentMetadata`` model, which places
it on the ``docling.service_client`` import chain. That chain must stay importable on any
docling-slim install that does not enable the PDF pipeline (and therefore ships no ``pypdfium2``).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from defusedxml.ElementTree import fromstring as _defused_fromstring
from docling_core.types.doc.common.meta import BaseMeta, LanguageMetaField
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import HumanLanguageLabel
from pydantic import BaseModel, Field

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    import pypdfium2 as pdfium
    from docling_parse.pdf_parser import PdfAnnotations

_log = logging.getLogger(__name__)

# ``created_by`` value recorded on the language field: the value is declared by the PDF
# producer, not detected from the text.
LANGUAGE_SOURCE_PDF_CATALOG = "pdf:/Lang"


class _PdfDocumentMetadata(BaseModel):
    """Document-level metadata read from a PDF (internal).

    Internal data-passing structure between a PDF backend's ``get_document_metadata()`` and
    the pipeline's document assembly; not part of the public datamodel or the serialized output.
    """

    # The catalog ``/Lang`` entry as written: a BCP 47 tag such as ``en-US``.
    language: str | None = None
    # Document properties keyed by XMP qualified name (``dc:title``, ``pdf:Producer``).
    properties: dict[str, str] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return self.language is None and not self.properties


# XMP namespaces whose properties are kept, keyed by URI (XMP Specification Part 1, 8.2 to 8.4,
# plus the PDF/A and PDF/UA identification schemas). Prefixes are the ones the specification
# registers; the packet's own prefixes are not trusted.
_XMP_NAMESPACES: dict[str, str] = {
    "http://purl.org/dc/elements/1.1/": "dc",
    "http://ns.adobe.com/pdf/1.3/": "pdf",
    "http://ns.adobe.com/xap/1.0/": "xmp",
    "http://ns.adobe.com/xap/1.0/mm/": "xmpMM",
    "http://ns.adobe.com/xap/1.0/rights/": "xmpRights",
    "http://ns.adobe.com/pdfx/1.3/": "pdfx",
    "http://ns.adobe.com/photoshop/1.0/": "photoshop",
    "http://www.aiim.org/pdfa/ns/id/": "pdfaid",
    "http://www.aiim.org/pdfua/ns/id/": "pdfuaid",
}
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDF_DESCRIPTION = f"{{{_RDF_NS}}}Description"
_RDF_LI = f"{{{_RDF_NS}}}li"
_RDF_ALT = f"{{{_RDF_NS}}}Alt"
_RDF_ARRAYS = {_RDF_ALT, f"{{{_RDF_NS}}}Bag", f"{{{_RDF_NS}}}Seq"}
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_XMP_DEFAULT_LANG = "x-default"

# ``/Info`` keys and the XMP properties ISO 32000-2 Table 349 pairs them with.
_INFO_TO_XMP: dict[str, str] = {
    "Title": "dc:title",
    "Author": "dc:creator",
    "Subject": "dc:description",
    "Keywords": "pdf:Keywords",
    "Creator": "xmp:CreatorTool",
    "Producer": "pdf:Producer",
    "CreationDate": "xmp:CreateDate",
    "ModDate": "xmp:ModifyDate",
}
_XMP_DATE_PROPERTIES = {"xmp:CreateDate", "xmp:ModifyDate"}

# PDF date string (ISO 32000-2, 7.9.4): ``D:YYYYMMDDHHmmSSOHH'mm'``, every part after the year
# optional. The trailing apostrophes are tolerated in both positions.
_PDF_DATE_RE = re.compile(
    r"^D:(?P<y>\d{4})(?P<mo>\d{2})?(?P<d>\d{2})?(?P<h>\d{2})?(?P<mi>\d{2})?(?P<s>\d{2})?"
    r"(?:(?P<tz>[Zz])|(?P<sign>[+-])(?P<tzh>\d{2})'?(?P<tzm>\d{2})?'?)?$"
)


def parse_xmp_properties(xml: str) -> dict[str, str]:
    """Flatten an XMP packet into ``prefix:LocalName`` -> text for the known namespaces.

    Simple properties keep their text. Language alternatives (``rdf:Alt``) yield the
    ``x-default`` entry, else the first. Ordered and unordered arrays (``rdf:Seq``,
    ``rdf:Bag``) are joined with ``", "``. Properties written as attributes of
    ``rdf:Description`` are read too. Properties in other namespaces, and nested structures
    without a text value, are skipped. An unparsable packet yields no properties.
    """
    try:
        root = _defused_fromstring(xml)
    except ElementTree.ParseError as e:
        _log.debug("XMP packet is not well-formed XML, skipping: %s", e)
        return {}

    properties: dict[str, str] = {}
    for description in root.iter(_RDF_DESCRIPTION):
        for attr_name, attr_value in description.attrib.items():
            name = _xmp_qualified_name(attr_name)
            value = attr_value.strip()
            if name is not None and value:
                properties.setdefault(name, value)
        for child in description:
            name = _xmp_qualified_name(child.tag)
            if name is None:
                continue
            value = _xmp_property_value(child)
            if value is not None:
                properties.setdefault(name, value)
    return properties


def _xmp_qualified_name(tag: str) -> str | None:
    if not tag.startswith("{"):
        return None
    uri, _, local_name = tag[1:].partition("}")
    prefix = _XMP_NAMESPACES.get(uri)
    if prefix is None or not local_name:
        return None
    return f"{prefix}:{local_name}"


def _xmp_property_value(element: ElementTree.Element) -> str | None:
    array = next((child for child in element if child.tag in _RDF_ARRAYS), None)
    if array is None:
        text = (element.text or "").strip()
        return text or None

    items = [
        (item.get(_XML_LANG), (item.text or "").strip())
        for item in array
        if item.tag == _RDF_LI
    ]
    items = [(lang, text) for lang, text in items if text]
    if not items:
        return None
    if array.tag == _RDF_ALT:
        return next(
            (text for lang, text in items if lang == _XMP_DEFAULT_LANG), items[0][1]
        )
    return ", ".join(text for _, text in items)


def pdf_date_to_iso8601(value: str) -> str:
    """Convert a PDF date string to ISO 8601, the XMP date form; other input is returned as is."""
    match = _PDF_DATE_RE.match(value.strip())
    if match is None:
        return value
    parts = match.groupdict()
    result = parts["y"]
    for key, separator in (
        ("mo", "-"),
        ("d", "-"),
        ("h", "T"),
        ("mi", ":"),
        ("s", ":"),
    ):
        if parts[key] is None:
            break
        result += separator + parts[key]
    if parts["tz"] is not None:
        result += "Z"
    elif parts["sign"] is not None:
        result += f"{parts['sign']}{parts['tzh']}:{parts['tzm'] or '00'}"
    return result


def extract_metadata_from_pdfium(pdoc: pdfium.PdfDocument) -> _PdfDocumentMetadata:
    """Read ``/Lang`` and the ``/Info`` dictionary through pypdfium2."""
    with pypdfium2_lock:
        info = pdoc.get_metadata_dict(skip_empty=True)
        language = _pdfium_catalog_language(pdoc)

    properties: dict[str, str] = {}
    for key, raw_value in info.items():
        name = _INFO_TO_XMP.get(key)
        value = raw_value.strip()
        if name is None or not value:
            continue
        properties[name] = (
            pdf_date_to_iso8601(value) if name in _XMP_DATE_PROPERTIES else value
        )
    return _PdfDocumentMetadata(language=language, properties=properties)


def _pdfium_catalog_language(pdoc: pdfium.PdfDocument) -> str | None:
    """Return the catalog ``/Lang`` via ``FPDFCatalog_GetLanguage``; caller holds the lock."""
    import ctypes

    import pypdfium2.raw as pdfium_c

    try:
        get_language = pdfium_c.FPDFCatalog_GetLanguage
    except AttributeError:
        # pypdfium2 binds whatever pdfium build it ships; builds older than the
        # FPDFCatalog_GetLanguage API (added to pdfium in 2024) have no catalog language.
        return None

    n_bytes = get_language(pdoc.raw, None, 0)
    if n_bytes <= 2:  # 0 on failure; 2 is the UTF-16LE terminator alone
        return None
    buffer = ctypes.create_string_buffer(n_bytes)
    get_language(
        pdoc.raw, ctypes.cast(buffer, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), n_bytes
    )
    language = buffer.raw[: n_bytes - 2].decode("utf-16-le", errors="replace").strip()
    return language or None


def extract_metadata_from_docling_parse(
    annotations: PdfAnnotations | None,
) -> _PdfDocumentMetadata:
    """Read ``/Lang`` and the XMP packet from docling-parse's native document annotations."""
    if annotations is None:
        return _PdfDocumentMetadata()
    language = (annotations.language or "").strip() or None
    properties = (
        parse_xmp_properties(annotations.meta_xml) if annotations.meta_xml else {}
    )
    return _PdfDocumentMetadata(language=language, properties=properties)


def primary_language_subtag(tag: str) -> HumanLanguageLabel | None:
    """Map a BCP 47 tag to the two-letter primary subtag docling-core accepts, if it is one."""
    primary = tag.split("-", 1)[0].lower()
    try:
        return HumanLanguageLabel(primary)
    except ValueError:
        return None


def attach_document_metadata(
    doc: DoclingDocument, metadata: _PdfDocumentMetadata
) -> None:
    """Record the PDF's language and properties on ``doc.body.meta``.

    The language goes to ``BaseMeta.language`` (primary subtag, ``created_by`` set to
    :data:`LANGUAGE_SOURCE_PDF_CATALOG`) when docling-core's label set has it, and always to
    the custom field ``pdf__lang`` with the full tag, so region and script subtags are kept.
    Each property becomes a custom field named after its XMP qualified name,
    ``dc:title`` -> ``dc__title`` and ``xmp:CreatorTool`` -> ``xmp__creator_tool``. An existing
    language or field on the body meta is left in place.
    """
    if metadata.is_empty():
        return
    meta = doc.body.meta if doc.body.meta is not None else BaseMeta()

    if metadata.language is not None:
        _set_custom_field_if_absent(meta, "pdf", "lang", metadata.language)
        code = primary_language_subtag(metadata.language)
        if code is not None and meta.language is None:
            meta.language = LanguageMetaField(
                code=code, created_by=LANGUAGE_SOURCE_PDF_CATALOG
            )

    for name, value in metadata.properties.items():
        prefix, _, local_name = name.partition(":")
        _set_custom_field_if_absent(meta, prefix, _snake_case(local_name), value)

    doc.body.meta = meta


def _set_custom_field_if_absent(
    meta: BaseMeta, namespace: str, name: str, value: str
) -> None:
    key = f"{namespace}__{name}"
    if key in meta.get_custom_part():
        return
    meta.set_custom_field(namespace=namespace, name=name, value=value)


_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _snake_case(name: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub("_", name).lower()
