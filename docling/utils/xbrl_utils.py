# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers to detect XBRL and inline XBRL (iXBRL) documents.

The helpers do not import Arelle, so format detection works without the XBRL extra.
"""

import re
from typing import Final

_PROLOG_RE: Final = re.compile(
    rb"<!--.*?-->|<\?.*?\?>|<!DOCTYPE[^>]*>", re.DOTALL | re.IGNORECASE
)
_ROOT_TAG_RE: Final = re.compile(
    rb"<(?:[\w.-]+:)?(?:html|xhtml)\b[^>]*>", re.IGNORECASE
)
_IXBRL_NAMESPACE_RE: Final = re.compile(
    rb"""\bxmlns:[\w.-]+\s*=\s*["']http://www\.xbrl\.org/(?:2008|2013)/inlineXBRL["']"""
)


def is_xbrl(content_str: str) -> bool:
    """Return whether a text contains the markers of an XBRL instance document.

    The markers are the 2003 instance namespace and an element name that starts
    with `<xbrl` (for example `<xbrl` or a prefixed `<xbrli:xbrl`).

    Args:
        content_str: Text to inspect.

    Returns:
        True if both markers are found, False otherwise.
    """
    return (
        "http://www.xbrl.org/2003/instance" in content_str
        and "<xbrl" in content_str.lower()
    )


def is_ixbrl(head: bytes) -> bool:
    """Return whether a document is inline XBRL (iXBRL) from its first bytes.

    A document is iXBRL when its root `html` or `xhtml` element declares the
    inline XBRL namespace (2008 or 2013 version). Comments, processing
    instructions and the doctype before the root element are ignored, so text
    that only mentions the namespace does not match.

    Args:
        head: The first bytes of the document, up to and including its root tag.

    Returns:
        True if the root element declares the inline XBRL namespace.
    """
    root = _ROOT_TAG_RE.search(_PROLOG_RE.sub(b"", head))
    return root is not None and _IXBRL_NAMESPACE_RE.search(root.group(0)) is not None
