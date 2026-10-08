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


IXBRL_SNIFF_BYTES: Final[int] = 1024
"""Number of bytes read from the document head when sniffing for iXBRL markers.

iXBRL markers appear in the opening ``<html>`` tag, but XML declarations and
HTML comments that precede the tag can easily consume several hundred bytes.
1024 bytes provides a comfortable margin while remaining negligible compared
to the size of a real iXBRL document.
"""


def is_xbrl(content_str: str) -> bool:
    """Return whether ``content`` contains traditional XBRL markers.

    Checks for the <xbrl> as root element with the 2003 namespace.

    Args:
        content: Text to inspect.

    Returns:
        True if any XBRL marker is found, False otherwise.
    """
    return (
        "http://www.xbrl.org/2003/instance" in content_str
        and "<xbrl" in content_str.lower()
    )


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
