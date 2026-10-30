# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse Unlimited-OCR 2 layout blocks into a :class:`DoclingDocument`.

The model does not emit Markdown. A page is one string of blocks::

    <PAGE><BLOCK>title<LEVEL1>[x1, y1, x2, y2]<FORK>Introduction
    <BLOCK>text[x1, y1, x2, y2]<FORK>End of the left column
    <MERGE>
    <BLOCK>text[x1, y1, x2, y2]<FORK>start of the right column

Boxes are integers on a 0-1000 grid, in x1, y1, x2, y2 order, relative to the page.
Prefixes ``cross_`` and ``append_`` mark content that continues on another page.
Docling sends one page per request, so these prefixes are ignored and nothing is
merged across pages.

The parser accepts the malformed shapes that the vendor's post-processing script
accepts. These include a doubled or missing ``<BLOCK>``, ``</LEVEL`` or ``LEVEL}``,
a level after the box, fullwidth commas, extra box numbers, and a missing box or
``<FORK>``.
"""

import logging
import re
from dataclasses import dataclass

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
    TextItem,
)
from PIL import Image as PILImage

from docling.utils.deepseekocr_utils import _parse_table_html
from docling.utils.mineru_utils import _CJK_PATTERN
from docling.utils.vlm_utils import (
    link_adjacent_captions,
    normalize_section_header_levels,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

_TEXT_LABELS = {
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "footer": DocItemLabel.PAGE_FOOTER,
    "header": DocItemLabel.PAGE_HEADER,
    "image_caption": DocItemLabel.CAPTION,
    "image_footnote": DocItemLabel.FOOTNOTE,
    "page_footnote": DocItemLabel.FOOTNOTE,
    "page_number": DocItemLabel.PAGE_FOOTER,
    "ref_text": DocItemLabel.REFERENCE,
    "table_caption": DocItemLabel.CAPTION,
    "text": DocItemLabel.TEXT,
}
_FURNITURE_LABELS = {DocItemLabel.PAGE_FOOTER, DocItemLabel.PAGE_HEADER}
_PICTURE_LABELS = {"chart", "figure", "image", "img", "picture", "seal"}
_KNOWN_LABELS = {*_TEXT_LABELS, *_PICTURE_LABELS, "equation", "table", "title"}
_CAPTION_OWNER_LABELS = {
    "image_caption": {DocItemLabel.PICTURE},
    "table_caption": {DocItemLabel.TABLE},
}
_NOISE_TEXT = {"[figure]", "[image]", "[non-text]", "[unreadable]"}

_NUMBER = r"-?\d{1,9}"
_LEVEL = r"</?LEVEL\d{0,9}[>}]?"
# `\Z` keeps the box of a block that max_tokens cut before the closing bracket.
_BOX = rf"\[\s*{_NUMBER}(?:\s*[,\uff0c]\s*{_NUMBER}){{3,}}[\s,\uff0c]*(?:\]|\Z)"
_PAGE_RE = re.compile(r"</?PAGE\d*>")
_BLOCK_RE = re.compile(
    rf"(?:<BLOCK>\s*)+(?P<label>[a-zA-Z_]+)(?P<attrs>(?:\s*(?:{_LEVEL}|{_BOX}))*)"
    r"\s*(?:<FORK>)?"
)
_BARE_BLOCK_RE = re.compile(
    rf"^[ \t]*(?=(?:(?:cross|append)_)?(?:{'|'.join(sorted(_KNOWN_LABELS))})"
    rf"\s*(?:{_LEVEL}\s*)?{_BOX})",
    re.MULTILINE,
)
_BOX_RE = re.compile(_BOX)
_LEVEL_DIGITS_RE = re.compile(r"LEVEL(\d+)")
_STRAY_TOKEN_RE = re.compile(
    rf"<[|\uff5c][^<>\n]*[|\uff5c]>|</?(?:BLOCK|CHILD|FORK|MERGE)>|{_LEVEL}"
)
_SPAN_RE = re.compile(r'\b(colspan|rowspan)\s*=\s*"([^"]*)"', re.IGNORECASE)
_SPAN_VALUE_RE = re.compile(r"[1-9]\d{0,3}")
_CUT_TAG_RE = re.compile(r"<[^>]*\Z")


@dataclass(frozen=True)
class _Block:
    """One layout block, with its box as fractions of the page size."""

    label: str
    level: int | None
    box: tuple[float, float, float, float] | None
    text: str
    merge_next: bool = False


def _page_segment(content: str) -> str:
    """Return the text of the first page and remove text before its ``<PAGE>``."""
    segments = _PAGE_RE.split(content)
    if len(segments) == 1:
        return content
    if any(segment.strip() for segment in segments[2:]):
        _log.warning(
            "Unlimited-OCR 2 output holds more than one page for one input page; "
            "keeping the first page"
        )
    return segments[1]


def _parse_box(attrs: str) -> tuple[float, float, float, float] | None:
    match = _BOX_RE.search(attrs)
    if match is None:
        return None
    x1, y1, x2, y2 = (
        min(max(int(value), 0), 1000) / 1000
        for value in re.findall(_NUMBER, match.group())[:4]
    )
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2, y2


def _clean_text(text: str) -> str:
    text = _STRAY_TOKEN_RE.sub("", text).strip()
    return "" if text.lower() in _NOISE_TEXT else text


def _parse_blocks(segment: str) -> list[_Block]:
    segment = _BARE_BLOCK_RE.sub("<BLOCK>", segment)
    matches = list(_BLOCK_RE.finditer(segment))
    head = _clean_text(segment[: matches[0].start()] if matches else segment)
    blocks = [_Block("text", None, None, head)] if head else []
    ends = [*(match.start() for match in matches[1:]), len(segment)]
    for match, end in zip(matches, ends):
        raw_text = segment[match.end() : end]
        level = _LEVEL_DIGITS_RE.search(match["attrs"])
        blocks.append(
            _Block(
                label=match["label"]
                .lower()
                .removeprefix("cross_")
                .removeprefix("append_"),
                level=int(level.group(1)) if level else None,
                box=_parse_box(match["attrs"]),
                text=_clean_text(raw_text),
                merge_next="<MERGE>" in raw_text,
            )
        )
    unknown_labels = {block.label for block in blocks} - _KNOWN_LABELS
    if unknown_labels:
        _log.warning(
            "Reading unknown Unlimited-OCR 2 labels as text: %s",
            sorted(unknown_labels),
        )
    return blocks


def _clean_span(match: re.Match[str]) -> str:
    value = _SPAN_VALUE_RE.search(match.group(2))
    return f'{match.group(1)}="{value.group() if value else 1}"'


def _repair_table_html(html: str) -> str:
    """Fix what makes ``_parse_table_html`` give an empty table."""
    html = _SPAN_RE.sub(_clean_span, html)
    if "</table>" not in html.lower():
        html = _CUT_TAG_RE.sub("", html) + "</table>"
    return html


def _provenance(block: _Block, page_size: Size, page_no: int) -> ProvenanceItem | None:
    if block.box is None:
        _log.warning("Unlimited-OCR 2 %s block has no valid box", block.label)
        return None
    x1, y1, x2, y2 = block.box
    return ProvenanceItem(
        page_no=page_no,
        charspan=(0, len(block.text)),
        bbox=BoundingBox(
            l=x1 * page_size.width,
            t=y1 * page_size.height,
            r=x2 * page_size.width,
            b=y2 * page_size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )


def _add_block(
    doc: DoclingDocument, block: _Block, prov: ProvenanceItem | None
) -> TextItem | None:
    """Add one block and return the item that a later ``<MERGE>`` can extend."""
    text = block.text
    if block.label in _PICTURE_LABELS:
        doc.add_picture(prov=prov)
        if text:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                prov=prov.model_copy() if prov else None,
            )
        return None
    if not text:
        return None
    if block.label == "title":
        if block.level is None:
            doc.add_title(text=text, prov=prov)
        else:
            doc.add_heading(text=text, level=min(max(block.level, 1), 100), prov=prov)
        return None
    if block.label == "equation":
        doc.add_formula(text=strip_math_delimiters(text), orig=text, prov=prov)
        return None
    if block.label == "table":
        data = _parse_table_html(_repair_table_html(text))
        if data.table_cells:
            doc.add_table(data=data, prov=prov)
            return None
        _log.warning("Keeping an unreadable Unlimited-OCR 2 table as text")
    label = _TEXT_LABELS.get(block.label, DocItemLabel.TEXT)
    item = doc.add_text(
        label=label,
        text=text,
        prov=prov,
        content_layer=ContentLayer.FURNITURE if label in _FURNITURE_LABELS else None,
    )
    return item if block.label == "text" else None


def _append_text(item: TextItem, text: str, prov: ProvenanceItem | None) -> None:
    separator = "" if _CJK_PATTERN.search(text) else " "
    start = len(item.text) + len(separator)
    item.text += separator + text
    item.orig += separator + text
    if prov is not None:
        prov.charspan = (start, start + len(text))
        item.prov.append(prov)


def parse_unlimited_ocr2(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    *,
    normalize_heading_levels: bool = True,
) -> DoclingDocument:
    """Parse the output of one Unlimited-OCR 2 page request.

    Args:
        content: Raw model output for one page.
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.
        normalize_heading_levels: If False, keep raw heading depths for a later
            document-wide :func:`normalize_section_header_levels`.

    Returns:
        A document for this page. An empty or ``<PAGE>``-only response gives an
        empty page.
    """
    doc = DoclingDocument(
        name=filename.rsplit(".", 1)[0],
        origin=DocumentOrigin(filename=filename, mimetype="text/plain", binary_hash=0),
    )
    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / original_page_size.width)
    doc.add_page(
        page_no=page_no,
        size=original_page_size,
        image=(
            ImageRef.from_pil(image=page_image, dpi=image_dpi)
            if page_image is not None
            else None
        ),
    )

    merge_target: TextItem | None = None
    merge_requested = False
    block_starts: list[int] = []
    caption_owners: dict[int, set[DocItemLabel]] = {}
    for index, block in enumerate(_parse_blocks(_page_segment(content))):
        block_starts.append(len(doc.body.children))
        if block.label in _CAPTION_OWNER_LABELS:
            caption_owners[index] = _CAPTION_OWNER_LABELS[block.label]
        prov = _provenance(block, original_page_size, page_no)
        # shortcut: <MERGE> joins only `text` blocks, as MinerU does; add titles
        # if real output splits them.
        if (
            merge_requested
            and merge_target is not None
            and block.label == "text"
            and block.text
        ):
            _append_text(merge_target, block.text, prov)
        else:
            merge_target = _add_block(doc, block, prov)
        merge_requested = block.merge_next

    link_adjacent_captions(doc, block_starts, caption_owners)
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
