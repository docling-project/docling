# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse the target of a link in a text-based document."""

import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import AnyUrl, ValidationError

_log = logging.getLogger(__name__)


def parse_hyperlink(address: str) -> AnyUrl | Path | None:
    """Return the link target as a URL, a path, or nothing.

    An address without a URL scheme, such as ``sub/page.html``, is a ``Path``.
    An address with a scheme that ``AnyUrl`` rejects, such as
    ``https://api.example.com:{port}/v1``, gives ``None`` with a warning, so the
    link text is kept without a target. ``Path`` would collapse its ``//`` into a
    different URL. An empty address gives ``None``.
    """
    if not address:
        return None
    try:
        # A one-letter scheme is a Windows drive, as in "C:\docs\page.html".
        has_scheme = len(urlparse(address).scheme) > 1
    except ValueError:  # e.g. "https://[broken"
        has_scheme = True
    if not has_scheme:
        return Path(address)
    try:
        return AnyUrl(address)
    except ValidationError:
        _log.warning("Skipping malformed hyperlink address: %r", address)
        return None
