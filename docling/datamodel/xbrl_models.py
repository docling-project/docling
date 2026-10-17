# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Shared XBRL constants and ZIP-inspection utilities.

This module is intentionally kept free of Arelle and of any ``docling.*``
imports so it can be imported both by the XBRL backend
(:mod:`docling.backend.xml.xbrl_backend`) and by the format-detection
layer (:mod:`docling.datamodel.document`) without introducing circular
dependencies.
"""

import importlib.util
import zipfile
from typing import Final

XBRL_AVAILABLE: bool = importlib.util.find_spec("arelle") is not None
"""True when the ``arelle`` package is installed and XBRL/iXBRL parsing is available."""

IXBRL_MARKERS: Final[tuple[bytes, ...]] = (
    b"xmlns:ix=",
    b"xmlns:ix =",
    b"<ix:",
    b"http://www.xbrl.org/2013/inlineXBRL",
)
"""Byte patterns that identify an inline XBRL (iXBRL) document.

All markers are ASCII and appear in the opening ``<html>`` tag, so a few
hundred bytes of the document head are sufficient for a reliable match.
Checked against raw bytes — no decoding required.
"""

XBRL_TAXONOMY_LINKBASE_SUFFIXES: Final[tuple[str, ...]] = (
    "_cal.xml",
    "_def.xml",
    "_lab.xml",
    "_pre.xml",
)
"""Filename suffixes that identify XBRL taxonomy linkbase files.

Entries with these suffixes are never a primary instance document and are
excluded from instance-document candidate searches.
"""

XBRL_INSTANCE_NS: Final[bytes] = b"http://www.xbrl.org/2003/instance"
"""Namespace URI that identifies a traditional XBRL instance document."""

IXBRL_SNIFF_BYTES: Final[int] = 1024
"""Number of bytes read from the document head when sniffing for iXBRL markers.

iXBRL markers appear in the opening ``<html>`` tag, but XML declarations and
HTML comments that precede the tag can easily consume several hundred bytes.
1024 bytes provides a comfortable margin while remaining negligible compared
to the size of a real iXBRL document.
"""

_XBRL_IXBRL_SUFFIXES: Final[tuple[str, ...]] = (".htm", ".html", ".xhtml")
_XBRL_INSTANCE_XML_SUFFIXES: Final[tuple[str, ...]] = (".xml", ".xbrl")


def scan_xbrl_zip_entries(
    zf: zipfile.ZipFile,
) -> tuple[list[tuple[int, str]], list[str]]:
    """Scan an open ZIP archive for XBRL instance document candidates.

    Reads the first :data:`IXBRL_SNIFF_BYTES` bytes of each candidate member
    to distinguish inline XBRL (iXBRL) from traditional XBRL instance files.
    Taxonomy linkbase files (``_cal``, ``_def``, ``_lab``, ``_pre``) and
    ``.xsd`` schema files are excluded from consideration.

    Args:
        zf: An already-open :class:`zipfile.ZipFile` to inspect.

    Returns:
        A 2-tuple ``(ixbrl_candidates, xml_candidates)`` where:

        * ``ixbrl_candidates`` is a list of ``(file_size, entry_name)`` tuples
          for members that carry ``ix:`` namespace markers.
        * ``xml_candidates`` is a list of entry names whose head contains
          the XBRL 2003 instance namespace URI.
    """
    ixbrl_candidates: list[tuple[int, str]] = []
    xml_candidates: list[str] = []

    for name in zf.namelist():
        name_lower = name.lower()
        if any(name_lower.endswith(suf) for suf in XBRL_TAXONOMY_LINKBASE_SUFFIXES):
            continue
        if name_lower.endswith(".xsd"):
            continue
        if any(name_lower.endswith(suf) for suf in _XBRL_IXBRL_SUFFIXES):
            with zf.open(name) as fh:
                head = fh.read(IXBRL_SNIFF_BYTES)
            if any(marker in head for marker in IXBRL_MARKERS):
                ixbrl_candidates.append((zf.getinfo(name).file_size, name))
        elif any(name_lower.endswith(suf) for suf in _XBRL_INSTANCE_XML_SUFFIXES):
            with zf.open(name) as fh:
                head = fh.read(IXBRL_SNIFF_BYTES)
            if XBRL_INSTANCE_NS in head:
                xml_candidates.append(name)

    return ixbrl_candidates, xml_candidates


def is_ixbrl(content: bytes) -> bool:
    """Return whether ``content`` contains inline XBRL (iXBRL) markers.

    Checks for namespace declarations or element prefixes that are exclusive
    to iXBRL documents.  All markers are ASCII and appear in the opening
    ``<html>`` tag, so a few hundred bytes of the document head are sufficient
    for a reliable result.

    Args:
        content: Raw bytes to inspect.

    Returns:
        True if any iXBRL marker is found, False otherwise.
    """
    return any(marker in content for marker in IXBRL_MARKERS)
