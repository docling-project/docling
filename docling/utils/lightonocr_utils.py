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
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    Size,
    TableData,
    TabularChartMetaField,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import (
    _add_html_fragment,
    _HTMLTreeParser,
    _table_cells,
    _table_data,
)
from docling.utils.markdown_math import create_markdown_parser
from docling.utils.vlm_utils import (
    link_adjacent_captions,
    normalize_section_header_levels,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

_MARKER = re.compile(
    r"!\[([A-Za-z_][\w-]*?)\+?\]"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*"
)
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)")
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


def _next_marker(line: str) -> re.Match[str] | None:
    """Find the first marker outside inline code spans.

    shortcut: indented code, <code> tags and table cells can still split a block.
    If real output has no markers in mid-line, anchor markers to the line start again.
    """
    spans = [span.span() for span in _CODE_SPAN.finditer(line)]
    for marker in _MARKER.finditer(line):
        if not any(start <= marker.start() < end for start, end in spans):
            return marker
    return None


def _blocks(content: str) -> Iterator[tuple[str, str, str]]:
    label, bbox = "text", ""
    lines: list[str] = []
    fence = ""
    for line in content.splitlines(keepends=True):
        while not fence and (marker := _next_marker(line)) is not None:
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


def _chart_data(body: str) -> TableData | None:
    """Read the HTML table of a chart block. Other bodies describe the chart."""
    root = _HTMLTreeParser(body).root
    table = next((node for node in root.elements() if node.tag == "table"), None)
    if table is None:
        return None
    try:
        cells = _table_cells(table)
    except ValueError as exc:
        # Like the Chandra mapper: drop the grid, keep the page.
        _log.warning("LightOnOCR chart table is too large to map (%s)", exc)
        return None
    return _table_data([cell for cell, _ in cells]) if cells else None


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

    Continuation labels retain their base class, and separate blocks are not
    merged. A caption links to its one adjacent table or picture. Plain
    transcription is recovered without boxes. A ``title`` block with ``##`` or
    deeper is a section header, because the model has one label for all headings.
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
    block_starts: list[int] = []
    caption_owners: dict[int, set[DocItemLabel]] = {}
    for index, (label, bbox, body) in enumerate(_blocks(_THINK.sub("", content))):
        block_starts.append(len(doc.body.children))
        if not body.strip() and label not in _PICTURE_LABELS:
            continue
        if label not in _LABELS:
            _log.warning("Unknown LightOnOCR label %r; keeping it as text", label)
        layout_label = _LABELS.get(label, "Text")
        chart_data = None
        if label == "formula":
            rendered = f'<math display="block">{html.escape(strip_math_delimiters(body))}</math>'
        elif label in _PICTURE_LABELS:
            chart_data = _chart_data(body) if label == "chart" else None
            alt = "" if chart_data is not None else body.strip()
            rendered = f'<img alt="{html.escape(alt, quote=True)}">'
        else:
            parsed = markdown.parse(body)
            rendered = markdown.render(parsed)
            if label == "title" and any(
                isinstance(node, (Heading, SetextHeading)) and node.level > 1
                for node in parsed.children
            ):
                layout_label = "Section-Header"
        if label == "caption":
            caption_owners[index] = {DocItemLabel.TABLE, DocItemLabel.PICTURE}
        attrs = f'data-bbox="{bbox}" data-label="{layout_label}"' if bbox else ""
        fragment = f"<div {attrs}>{rendered}</div>"
        picture_start = len(doc.pictures)
        _add_html_fragment(doc, fragment, label=DocItemLabel.TEXT)
        for picture in doc.pictures[picture_start:]:
            if picture.meta is not None and picture.meta.description is not None:
                picture.meta.description.created_by = "lightonocr"
            if label in {"header_image", "footer_image"}:
                picture.content_layer = ContentLayer.FURNITURE
            if label == "chart" and picture.meta is not None:
                # shortcut: the model gives no chart type, so the class is OTHER_CHART.
                # DCLX and DocLang lose its chart data, and export_to_doctags() raises
                # ValueError. Upgrade when docling-core supports this class in both.
                picture.meta.classification = PictureClassificationMetaField(
                    predictions=[
                        PictureClassificationPrediction(
                            class_name=PictureClassificationLabel.OTHER_CHART.value
                        )
                    ]
                )
                if chart_data is not None:
                    picture.meta.tabular_chart = TabularChartMetaField(
                        chart_data=chart_data
                    )
    link_adjacent_captions(doc, block_starts, caption_owners)
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
