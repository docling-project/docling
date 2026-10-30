# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse LightOnOCR-3 Markdown with normalized block bounding boxes."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    Size,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _add_html_fragment
from docling.utils.markdown_math import create_markdown_parser
from docling.utils.vlm_utils import (
    normalize_section_header_levels,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

_MARKER = re.compile(
    r"!\[([A-Za-z_][\w-]*?)\+?\]"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*"
)
# Upstream reports replies that start with the closing tag of the empty think block.
_THINK = re.compile(r"\A\s*(?:<think>.*?)?</think>\s*", re.DOTALL)
_LABELS = {
    "text": "Text",
    "aside_text": "Text",
    "title": "Title",
    "section": "Section-Header",
    "header": "Page-Header",
    "footer": "Page-Footer",
    "page_number": "Page-Footer",
    "footnote": "Footnote",
    "caption": "Caption",
    "formula": "Equation-Block",
    "code": "Code-Block",
    "table": "Table",
    "list": "List-Group",
    "image": "Image",
    "chart": "Figure",
    "header_image": "Image",
    "footer_image": "Image",
}
_PICTURE_LABELS = {"image", "chart", "header_image", "footer_image"}


def _blocks(content: str) -> Iterator[tuple[str, str, str]]:
    label, bbox = "text", ""
    lines: list[str] = []
    fence = ""
    for line in content.splitlines(keepends=True):
        while not fence and (marker := _MARKER.search(line)) is not None:
            yield label, bbox, "".join(lines) + line[: marker.start()]
            label, *box = marker.groups()
            bbox, lines = " ".join(box), []
            line = line[marker.end() :]

        # Check the block body too: a fence can start on the marker's line.
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_match:
            delimiter = fence_match[1]
            if not fence:
                fence = delimiter
            elif (
                delimiter[0] == fence[0]
                and len(delimiter) >= len(fence)
                and not line[fence_match.end() :].strip()
            ):
                fence = ""
        lines.append(line)
    yield label, bbox, "".join(lines)


def parse_lightonocr_markdown(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    *,
    normalize_heading_levels: bool = True,
) -> DoclingDocument:
    """Preserve Markdown/HTML content and assign each block bbox at most once.

    Continuation labels retain their base class. Separate blocks are not merged
    or associated by proximity. Plain transcription is recovered without boxes.
    A ``title`` block with ``##`` or deeper is a section header, because the
    model has one label for all headings.
    Heading depths normalize per page unless the pipeline defers normalization
    until all pages are joined with ``normalize_heading_levels=False``.
    """
    markdown = create_markdown_parser()
    # The parser above checks this optional dependency before its types are used.
    from marko.block import Heading, SetextHeading

    doc = DoclingDocument(
        name=Path(filename).stem,
        origin=DocumentOrigin(
            filename=filename, mimetype="text/markdown", binary_hash=0
        ),
    )
    dpi = (
        max(1, round(72 * page_image.width / original_page_size.width))
        if page_image is not None
        else 72
    )
    doc.add_page(
        page_no=page_no,
        size=original_page_size,
        image=ImageRef.from_pil(image=page_image, dpi=dpi) if page_image else None,
    )
    for label, bbox, body in _blocks(_THINK.sub("", content)):
        if not body.strip() and label not in _PICTURE_LABELS:
            continue
        if label not in _LABELS:
            _log.warning("Unknown LightOnOCR label %r; keeping it as text", label)
        layout_label = _LABELS.get(label, "Text")
        if label == "formula":
            rendered = f'<math display="block">{html.escape(strip_math_delimiters(body))}</math>'
        elif label in _PICTURE_LABELS - {"chart"}:
            rendered = f'<img alt="{html.escape(body.strip(), quote=True)}">'
        else:
            parsed = markdown.parse(body)
            rendered = markdown.render(parsed)
            if label == "title" and any(
                isinstance(node, (Heading, SetextHeading)) and node.level > 1
                for node in parsed.children
            ):
                layout_label = "Section-Header"
        attrs = f'data-bbox="{bbox}" data-label="{layout_label}"' if bbox else ""
        fragment = f"<div {attrs}>{rendered}</div>"
        picture_start = len(doc.pictures)
        _add_html_fragment(doc, fragment, label=DocItemLabel.TEXT)
        for picture in doc.pictures[picture_start:]:
            if picture.meta is not None and picture.meta.description is not None:
                picture.meta.description.created_by = "lightonocr"
            if label in {"header_image", "footer_image"}:
                picture.content_layer = ContentLayer.FURNITURE
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
