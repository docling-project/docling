# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse the compact HTML output of Surya 2.5 (datalab-to/surya-ocr-2.5).

Surya 2.5 writes a layout block as one label token and four coordinate tokens
(0-1000), followed by the block HTML and no closing ``</div>``::

    <|Text|><|120|><|87|><|881|><|255|><p>...</p>

A ``data-bbox`` on a nested tag is written as ``<|box|>`` and four coordinate
tokens just before that tag. Tables leave out ``</td>``, ``</th>`` and ``</tr>``,
and a merged cell is a ``data-kind="parent"`` cell with ``data-cs`` / ``data-rs``,
followed by empty ``data-kind="ghost"`` cells.

This module turns that output into the ``<div data-bbox data-label>`` dialect of
Chandra, which is also from Datalab, and maps it with the shared Chandra mapper.
It does not need the ``surya-ocr`` package.
"""

from __future__ import annotations

import re

from docling_core.types.doc import DoclingDocument, Size
from PIL import Image as PILImage

from docling.utils.chandra_utils import parse_chandra_html

_COORDS = r"<\|(\d{1,4})\|>" * 4
# Any capitalized label token followed by four coordinates is a block header.
_HEADER_RE = re.compile(r"<\|([A-Z][A-Za-z-]*)\|>" + _COORDS)
_BOX_RE = re.compile(r"<\|box\|>" + _COORDS + r"\s*<([a-zA-Z][a-zA-Z0-9-]*)")
# Markers that are left after decoding: end tokens, or a header or box cut by
# the token limit.
_LEFTOVER_MARKER_RE = re.compile(r"<\|[^<>|\s]*\|>|<\|[^<>|]*$")
_GHOST_CELL_RE = re.compile(
    r"<t[dh]\b[^<>]*\bdata-kind\s*=\s*[\"']?ghost[\"']?[^<>]*>"
    r".*?(?=<t[dh][\s>/]|<tr[\s>/]|</tr\s*>|<t(?:head|body)[\s>]|"
    r"</t(?:able|head|body)\s*>|<div[\s>]|</div\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_CELL_TAG_RE = re.compile(r"<t[dh]\b[^<>]*>", re.IGNORECASE)
_SPAN_ATTR_RE = re.compile(r"\bdata-(cs|rs)(\s*=)", re.IGNORECASE)
_KIND_ATTR_RE = re.compile(
    r"\s+data-kind\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s<>]+)", re.IGNORECASE
)
# Surya labels that the Chandra mapper knows under another name.
_LABEL_ALIASES = {"Picture": "Image"}


def _format_box(match: re.Match[str], first: int) -> str:
    return " ".join(str(int(match.group(first + k))) for k in range(4))


def _convert_merged_cells(html: str) -> str:
    """Turn parent/ghost merged cells into ``colspan`` / ``rowspan`` cells.

    The ``data-cs`` and ``data-rs`` values of the parent cell give the span.
    Ghost cells only mark the covered grid positions, so they are removed.
    """
    html = _GHOST_CELL_RE.sub("", html)

    def rewrite(match: re.Match[str]) -> str:
        tag = _KIND_ATTR_RE.sub("", match.group(0))
        return _SPAN_ATTR_RE.sub(
            lambda m: (
                ("colspan" if m.group(1).lower() == "cs" else "rowspan") + m.group(2)
            ),
            tag,
        )

    return _CELL_TAG_RE.sub(rewrite, html)


def decode_surya_html(content: str) -> str:
    """Convert compact Surya 2.5 output to Chandra-style layout block HTML.

    Output that already uses ``<div data-bbox data-label>`` blocks passes
    through unchanged, apart from the merged-cell conversion.
    """
    text = _BOX_RE.sub(
        lambda m: f'<{m.group(5)} data-bbox="{_format_box(m, 1)}"', content
    )
    parts: list[str] = []
    position = 0
    in_block = False
    for match in _HEADER_RE.finditer(text):
        parts.append(text[position : match.start()])
        if in_block:
            parts.append("</div>")
        label = _LABEL_ALIASES.get(match.group(1), match.group(1))
        parts.append(f'<div data-bbox="{_format_box(match, 2)}" data-label="{label}">')
        in_block = True
        position = match.end()
    parts.append(text[position:])
    if in_block:
        parts.append("</div>")
    html = _LEFTOVER_MARKER_RE.sub("", "".join(parts))
    if "data-kind" in html:
        html = _convert_merged_cells(html)
    return html


def parse_surya_html(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    *,
    normalize_heading_levels: bool = True,
) -> DoclingDocument:
    """Map one page of Surya 2.5 output to a ``DoclingDocument``.

    Raises ``ValueError`` when the response has no document content. Set
    ``normalize_heading_levels`` to False to keep raw heading depths for a later
    document-wide :func:`normalize_section_header_levels`.
    """
    return parse_chandra_html(
        decode_surya_html(content),
        original_page_size=original_page_size,
        page_no=page_no,
        filename=filename,
        page_image=page_image,
        normalize_heading_levels=normalize_heading_levels,
    )
