# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse NVIDIA Nemotron 3.5 Parse native output into a DoclingDocument.

The model emits blocks in reading order::

    <x_X1><y_Y1>BODY<x_X2><y_Y2><class_CLASS>

Coordinates are normalized to the page image content, so they scale directly
to the page. Bodies are Markdown text, LaTeX ``tabular`` tables, or a
``<chart type="...">`` header with Markdown data tables.
"""

from __future__ import annotations

import html
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
    GroupItem,
    GroupLabel,
    ImageRef,
    ListGroup,
    NodeItem,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    TabularChartMetaField,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _MAX_TABLE_GRID_CELLS, _add_html_table
from docling.utils.code_language import detect_code_language
from docling.utils.dots_utils import (
    _add_inline_text,
    _markdown_inline_html,
    _unescape_markdown_literals,
)
from docling.utils.vlm_utils import (
    link_adjacent_captions,
    normalize_section_header_levels,
    parse_markdown_heading,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

_CLASS_TOKEN = re.compile(r"<class_([^>]+)>")
_COORD_TOKEN = re.compile(r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>")
_ARTIFACT_TOKENS = ("<tbc>", r"\<|unk|\>", r"\unknown")
_TEXT_LABELS = {
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Bibliography": DocItemLabel.REFERENCE,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
}
_FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
_ORDERED_MARKER = r"\(?(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)[.)]"
_LIST_MARKER = re.compile(
    # Bullets: hyphen, en and em dash, symbols, or LaTeX like \(\bullet\).
    rf"^\s*({_ORDERED_MARKER}|[-\u2013\u2014•►▪■□◦●○·*+]|\\\(\\[A-Za-z]+\\\))\s+"
)
_LATEX_TEXT = re.compile(
    r"\\(textless|textgreater|textasciitilde|linebreak|mbox)(?![A-Za-z])(?:\{\})?"
    r"|\\([%&#$])"
)
_LATEX_TEXT_REPLACEMENTS = {
    "textless": "&lt;",
    "textgreater": "&gt;",
    "textasciitilde": "~",
    "linebreak": "<br>",
    "mbox": "",
    "&": "&amp;",
}
# Inline tags that a table cell keeps as markup, and LaTeX text commands that it maps;
# all other text is escaped.
_CELL_MARKUP = re.compile(
    rf"{_LATEX_TEXT.pattern}|(?i:</?(?:b|br|code|em|i|strong|sub|sup|u)\s*/?>)"
)
_CHART_OPEN = re.compile(r'<chart\s+type="([^"]*)"\s*>')
_CHART_TABLE_KEY = re.compile(r"table\[(\w+)\]")


@dataclass(frozen=True)
class NemotronParseBlock:
    """One decoded Nemotron 3.5 Parse block with a content-normalized bbox."""

    label: str
    bbox: tuple[float, float, float, float]
    text: str


@dataclass
class _LatexCell:
    text: str
    row_span: int
    col_span: int
    header: bool


def extract_nemotron_parse_v3_5_blocks(content: str) -> list[NemotronParseBlock]:
    """Split the native output into blocks; drop malformed and repeated blocks."""
    blocks: list[NemotronParseBlock] = []
    seen: set[NemotronParseBlock] = set()
    start = 0
    for match in _CLASS_TOKEN.finditer(content):
        segment = content[start : match.start()]
        start = match.end()
        coords = list(_COORD_TOKEN.finditer(segment))
        if len(coords) < 2:
            continue
        first, last = coords[0], coords[-1]
        # Stray coordinate tokens inside a body come from degenerate generations.
        body = _COORD_TOKEN.sub("", segment[first.end() : last.start()])
        for token in _ARTIFACT_TOKENS:
            body = body.replace(token, "")
        block = NemotronParseBlock(
            label=match.group(1),
            bbox=(
                float(first.group(1)),
                float(first.group(2)),
                float(last.group(1)),
                float(last.group(2)),
            ),
            text=body.strip(),
        )
        if block not in seen:
            seen.add(block)
            blocks.append(block)
    return blocks


def _page_bbox(bbox: tuple[float, float, float, float], size: Size) -> BoundingBox:
    x1, y1, x2, y2 = (min(1.0, max(0.0, value)) for value in bbox)
    return BoundingBox(
        l=min(x1, x2) * size.width,
        t=min(y1, y2) * size.height,
        r=max(x1, x2) * size.width,
        b=max(y1, y2) * size.height,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _add_rich_text(
    doc: DoclingDocument,
    text: str,
    *,
    label: DocItemLabel,
    prov: ProvenanceItem | None,
    parent: NodeItem | None = None,
) -> None:
    first_text, first_group = len(doc.texts), len(doc.groups)
    if not _add_inline_text(doc, text, label=label, prov=prov, parent=parent):
        doc.add_text(
            label=label,
            text=_unescape_markdown_literals(text),
            orig=text,
            prov=prov,
            parent=parent,
        )
    if label in _FURNITURE_LABELS:
        for item in (*doc.texts[first_text:], *doc.groups[first_group:]):
            item.content_layer = ContentLayer.FURNITURE


def _read_group(text: str, pos: int) -> tuple[str, int] | None:
    """Read one balanced ``{...}`` group at ``pos``, skipping leading spaces."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None
    depth = 0
    index = pos
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1 : index], index + 1
        index += 1
    return None


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        if depth == 0 and text.startswith(separator, index):
            parts.append(text[start:index])
            index += len(separator)
            start = index
            continue
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            # The model sometimes closes a group twice; do not go negative.
            depth = max(0, depth - 1)
        index += 1
    parts.append(text[start:])
    return parts


def _unwrap_command(text: str, command: str, arg_count: int) -> list[str] | None:
    """Return the arguments of ``\\command{..}..`` when it wraps the whole cell."""
    stripped = text.strip()
    prefix = "\\" + command
    if (
        not stripped.startswith(prefix)
        or stripped[len(prefix) : len(prefix) + 1].isalpha()
    ):
        return None
    args: list[str] = []
    pos = len(prefix)
    for _ in range(arg_count):
        group = _read_group(stripped, pos)
        if group is None:
            return None
        content, pos = group
        args.append(content)
    rest = stripped[pos:].strip().lstrip("}").strip()
    if rest:
        args[-1] = f"{args[-1]} {rest}"
    return args


def _span(value: str) -> int:
    digits = value.strip().lstrip("0")
    if not digits.isdecimal():
        return 1
    # Above the limit _add_html_table rejects the table; do not build a huge grid first.
    if len(digits) > len(str(_MAX_TABLE_GRID_CELLS)):
        return _MAX_TABLE_GRID_CELLS + 1
    return min(int(digits), _MAX_TABLE_GRID_CELLS + 1)


def _escape_cell_markup(text: str) -> str:
    """Escape cell text so that it cannot change the table; keep inline tags."""
    parts: list[str] = []
    pos = 0
    for match in _CELL_MARKUP.finditer(text):
        parts.append(html.escape(text[pos : match.start()], quote=False))
        command = match.group(1) or match.group(2)
        parts.append(
            match.group()
            if command is None
            else _LATEX_TEXT_REPLACEMENTS.get(command, command)
        )
        pos = match.end()
    parts.append(html.escape(text[pos:], quote=False))
    return "".join(parts)


def _parse_latex_cell(text: str) -> _LatexCell:
    row_span = col_span = 1
    header = False
    while True:
        if (args := _unwrap_command(text, "multicolumn", 3)) is not None:
            col_span, text = _span(args[0]), args[2]
        elif (args := _unwrap_command(text, "multirow", 3)) is not None:
            row_span, text = _span(args[0]), args[2]
        elif (args := _unwrap_command(text, "thead", 1)) is not None:
            header, text = True, args[0]
        else:
            break
    text = _escape_cell_markup(text)
    return _LatexCell(
        text=text.strip(), row_span=row_span, col_span=col_span, header=header
    )


def latex_tabular_to_html(latex: str) -> str:
    """Convert Nemotron's LaTeX ``tabular`` dialect to an HTML table.

    An empty cell at a column that a ``\\multirow`` above still covers is the
    placeholder of that spanned cell and is dropped, as HTML omits it. A
    non-empty cell there ends the span instead, so no text is lost.
    Without ``\\begin{tabular}`` (a truncated prefix) the whole body is rows.
    """
    body = latex
    begin = latex.find(r"\begin{tabular}")
    if begin >= 0:
        body = latex[begin + len(r"\begin{tabular}") :]
        column_spec = _read_group(body, 0)
        if column_spec is not None:
            body = body[column_spec[1] :]
    end = body.find(r"\end{tabular}")
    if end >= 0:
        body = body[:end]
    rows = [row for row in _split_top_level(body, "\\\\") if row.strip()]
    # Column -> (cell, start row) of the last cell placed in that column.
    owners: dict[int, tuple[_LatexCell, int]] = {}
    grid: list[list[_LatexCell]] = []
    for row_index, row in enumerate(rows):
        col = 0
        cells: list[_LatexCell] = []
        for raw_cell in _split_top_level(row, "&"):
            cell = _parse_latex_cell(raw_cell)
            owner = owners.get(col)
            if owner is not None and owner[1] + owner[0].row_span > row_index:
                if not cell.text:
                    col += cell.col_span
                    continue
                # Text under a multirow: the predicted row span was too long.
                owner[0].row_span = row_index - owner[1]
            if col + cell.col_span > _MAX_TABLE_GRID_CELLS:
                raise ValueError(f"LaTeX table exceeds {_MAX_TABLE_GRID_CELLS} columns")
            for owned_col in range(col, col + cell.col_span):
                owners[owned_col] = (cell, row_index)
            cells.append(cell)
            col += cell.col_span
        grid.append(cells)

    html_rows: list[str] = []
    for cells in grid:
        html_cells: list[str] = []
        for cell in cells:
            tag = "th" if cell.header else "td"
            scope = ' scope="col"' if cell.header else ""
            html_cells.append(
                f'<{tag}{scope} rowspan="{cell.row_span}" colspan="{cell.col_span}">'
                f"{_markdown_inline_html(cell.text)}</{tag}>"
            )
        html_rows.append(f"<tr>{''.join(html_cells)}</tr>")
    return f"<table>{''.join(html_rows)}</table>"


def _add_latex_table(
    doc: DoclingDocument,
    latex: str,
    prov: ProvenanceItem,
    parent: NodeItem | None = None,
) -> None:
    try:
        if _add_html_table(doc, latex_tabular_to_html(latex), prov, parent) is not None:
            return
    except ValueError as exc:
        _log.warning("Nemotron Parse table is too large to map: %s", exc)
    _log.warning("Invalid Nemotron Parse table; preserving an empty table")
    doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=prov, parent=parent)


def _markdown_table_data(rows: list[str]) -> TableData:
    grid = [
        [cell.strip() for cell in row.strip().strip("|").split("|")]
        for row in rows
        if not set(row) <= set("|-: ")
    ]
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=row_index,
            end_row_offset_idx=row_index + 1,
            start_col_offset_idx=col_index,
            end_col_offset_idx=col_index + 1,
            column_header=row_index == 0,
        )
        for row_index, row in enumerate(grid)
        for col_index, text in enumerate(row)
    ]
    return TableData(
        num_rows=len(grid),
        num_cols=max((len(row) for row in grid), default=0),
        table_cells=cells,
    )


def _chart_label(chart_type: str) -> PictureClassificationLabel:
    # DocLang keeps chart data for the bar, stacked bar, line, pie, scatter, flow and
    # heatmap classes. BOX_PLOT and OTHER_CHART lose it, and export_to_doctags()
    # raises ValueError on OTHER_CHART.
    if "heatmap" in chart_type:
        return PictureClassificationLabel.HEATMAP
    if "stacked" in chart_type and "bar" in chart_type:
        return PictureClassificationLabel.STACKED_BAR_CHART
    if "bar" in chart_type:
        return PictureClassificationLabel.BAR_CHART
    if "line" in chart_type:
        return PictureClassificationLabel.LINE_CHART
    if "scatter" in chart_type or "bubble" in chart_type:
        return PictureClassificationLabel.SCATTER_CHART
    if "pie" in chart_type or "donut" in chart_type:
        return PictureClassificationLabel.PIE_CHART
    if "box" in chart_type:
        return PictureClassificationLabel.BOX_PLOT
    return PictureClassificationLabel.OTHER_CHART


def _add_chart_picture(
    doc: DoclingDocument,
    prov: ProvenanceItem,
    *,
    chart_type: str,
    title: str | None,
    rows: list[str],
) -> None:
    picture = doc.add_picture(prov=prov)
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=_chart_label(chart_type).value
                )
            ]
        ),
        tabular_chart=TabularChartMetaField(
            title=title, chart_data=_markdown_table_data(rows)
        ),
    )


def _add_chart(doc: DoclingDocument, text: str, prov: ProvenanceItem) -> None:
    """Add one picture per chart, or one per panel when panels have own tables."""
    open_match = _CHART_OPEN.search(text)
    chart_type = open_match.group(1) if open_match is not None else "other"
    body = text[open_match.end() :] if open_match is not None else text
    body = body.split("</chart>", 1)[0]

    title: str | None = None
    panels: list[tuple[str, str, str]] = []
    tables: dict[str | None, list[str]] = {None: []}
    table_key: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            tables.setdefault(table_key, []).append(stripped)
            continue
        name, separator, value = stripped.partition(":")
        if not separator:
            continue
        if name == "title":
            title = value.strip()
        elif name == "panel":
            # "id ; type" with an optional "; label".
            parts = [part.strip() for part in value.split(";", 2)]
            if len(parts) >= 2:
                panels.append((parts[0], parts[1], parts[2] if len(parts) == 3 else ""))
        elif (key_match := _CHART_TABLE_KEY.fullmatch(name)) is not None:
            table_key = key_match.group(1)

    panel_charts = [panel for panel in panels if tables.get(panel[0])]
    if panel_charts:
        for panel_id, panel_type, panel_label in panel_charts:
            _add_chart_picture(
                doc,
                prov,
                chart_type=panel_type,
                title=panel_label or None,
                rows=tables[panel_id],
            )
        return
    rows = tables[None] or next((rows for rows in tables.values() if rows), [])
    _add_chart_picture(doc, prov, chart_type=chart_type, title=title, rows=rows)


def parse_nemotron_parse_v3_5(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
    *,
    normalize_heading_levels: bool = True,
) -> DoclingDocument:
    """Parse one page of Nemotron 3.5 Parse native output.

    Set ``normalize_heading_levels`` to False to keep raw heading depths for a
    later document-wide :func:`normalize_section_header_levels`.
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

    list_group: ListGroup | None = None
    toc_group: GroupItem | None = None
    block_starts: list[int] = []
    caption_owners: dict[int, set[DocItemLabel]] = {}
    for index, block in enumerate(extract_nemotron_parse_v3_5_blocks(content)):
        block_starts.append(len(doc.body.children))
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=_page_bbox(block.bbox, original_page_size),
            charspan=(0, 0),
        )
        text = block.text
        if block.label != "List-item":
            list_group = None
        if block.label != "TOC":
            toc_group = None

        if block.label == "List-item":
            if list_group is None:
                list_group = doc.add_list_group()
            marker_match = _LIST_MARKER.match(text)
            marker = marker_match.group(1) if marker_match is not None else None
            body = text[marker_match.end() :] if marker_match is not None else text
            item = doc.add_list_item(
                text="",
                enumerated=marker is not None
                and re.fullmatch(_ORDERED_MARKER, marker) is not None,
                marker=marker,
                orig=text,
                parent=list_group,
                prov=prov,
            )
            if not _add_inline_text(doc, body, label=DocItemLabel.TEXT, parent=item):
                item.text = _unescape_markdown_literals(body)
        elif block.label == "TOC":
            if toc_group is None:
                toc_group = doc.add_group(label=GroupLabel.UNSPECIFIED, name="toc")
            if r"\begin{tabular}" in text:
                _add_latex_table(doc, text, prov, parent=toc_group)
            else:
                _add_rich_text(
                    doc, text, label=DocItemLabel.TEXT, prov=prov, parent=toc_group
                )
        elif block.label == "Table":
            _add_latex_table(doc, text, prov)
        elif block.label == "Chart":
            _add_chart(doc, text, prov)
        elif block.label == "Picture":
            picture = doc.add_picture(prov=prov)
            if text:
                _add_rich_text(
                    doc, text, label=DocItemLabel.TEXT, prov=None, parent=picture
                )
        elif block.label in {"Title", "Section-header"}:
            clean_text, level = parse_markdown_heading(text)
            heading = (
                doc.add_title(text="", orig=text, prov=prov)
                if block.label == "Title"
                else doc.add_heading(
                    text="",
                    orig=text,
                    # Raw markup depth, normalized per document later.
                    level=level if level is not None else 2,
                    prov=prov,
                )
            )
            if not _add_inline_text(
                doc, clean_text, label=DocItemLabel.TEXT, parent=heading
            ):
                heading.text = _unescape_markdown_literals(clean_text)
        elif block.label == "Formula":
            doc.add_formula(text=strip_math_delimiters(text), orig=text, prov=prov)
        elif block.label == "Code":
            doc.add_code(
                text=text,
                orig=text,
                prov=prov,
                code_language=detect_code_language(text),
            )
        else:
            if block.label == "Caption":
                caption_owners[index] = {DocItemLabel.TABLE, DocItemLabel.PICTURE}
            _add_rich_text(
                doc,
                text,
                label=_TEXT_LABELS.get(block.label, DocItemLabel.TEXT),
                prov=prov,
            )

    link_adjacent_captions(doc, block_starts, caption_owners)
    if normalize_heading_levels:
        normalize_section_header_levels(doc)
    return doc
