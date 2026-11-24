# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Whitespace normalization for text taken from LaTeX source."""

from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"[ \t\r\n]+")


def collapse_source_layout_whitespace(text: str) -> str:
    """Replace each whitespace run that has a tab or line break with one space.

    Runs of plain spaces stay unchanged. Each run is matched once, so the work is
    linear in the text length.
    """
    return _WHITESPACE_RUN.sub(
        lambda match: match.group() if not match.group().strip(" ") else " ",
        text,
    )
