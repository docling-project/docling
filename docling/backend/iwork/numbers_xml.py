# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the ``index.xml`` of an iWork '09 (or earlier) Numbers document.

Everything the spreadsheet shows sits inside an ``ls:workspace``, one per sheet.
The rest of the file also carries the tables and charts of the template the
document was made from, which the spreadsheet never draws, so the workspaces are
what is walked rather than the document as a whole.

Two things are stored more indirectly than they look. A table's datasource holds
only the cells a row actually uses, with no coordinates on them, so the grid's
occupancy counts are what put them back; and a chart bound to a table keeps the
values it last plotted in a property list of its own rather than inline.
"""

import logging
import plistlib
import zipfile
import zlib
from decimal import Decimal
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from docling_core.types.doc import BoundingBox, CoordOrigin

from docling.backend.iwork.numbers_content import (
    Cell,
    Chart,
    Comment,
    Sheet,
    Table,
    format_bool,
    format_date,
    format_number,
    reading_order,
)
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

MAX_LEGACY_XML_BYTES = 100 * 1024 * 1024
"""Ceiling on a decompressed ``index.xml.gz``.

The stored size of a gzipped member says nothing about what it expands to, so
the output is capped rather than the input.
"""

SF_NAMESPACE = "http://developer.apple.com/namespaces/sf"
SFA_NAMESPACE = "http://developer.apple.com/namespaces/sfa"
LS_NAMESPACE = "http://developer.apple.com/namespaces/ls"

LS_WORKSPACE = f"{{{LS_NAMESPACE}}}workspace"
LS_ATTR_WORKSPACE_NAME = f"{{{LS_NAMESPACE}}}workspace-name"

SF_TABULAR_INFO = f"{{{SF_NAMESPACE}}}tabular-info"
SF_TABULAR_MODEL = f"{{{SF_NAMESPACE}}}tabular-model"
SF_GEOMETRY = f"{{{SF_NAMESPACE}}}geometry"
SF_POSITION = f"{{{SF_NAMESPACE}}}position"
SF_SIZE = f"{{{SF_NAMESPACE}}}size"
SF_GRID = f"{{{SF_NAMESPACE}}}grid"
SF_ROWS = f"{{{SF_NAMESPACE}}}rows"
SF_GRID_ROW = f"{{{SF_NAMESPACE}}}grid-row"
SF_COLUMNS = f"{{{SF_NAMESPACE}}}columns"
SF_GRID_COLUMN = f"{{{SF_NAMESPACE}}}grid-column"
SF_DATASOURCE = f"{{{SF_NAMESPACE}}}datasource"
SF_CELL_TEXT = f"{{{SF_NAMESPACE}}}ct"
SF_RESULT = f"{{{SF_NAMESPACE}}}r"
SF_PROXIED_CELL = f"{{{SF_NAMESPACE}}}proxied-cell-ref"
SF_CHART_INFO = f"{{{SF_NAMESPACE}}}chart-info"
SF_CHART_NAME = f"{{{SF_NAMESPACE}}}chart-name"
SF_CHART_COLUMN_NAMES = f"{{{SF_NAMESPACE}}}chart-column_names"
SF_CHART_ROW_NAMES = f"{{{SF_NAMESPACE}}}chart-row_names"
SF_ENTITY_ID = f"{{{SF_NAMESPACE}}}entity-id"
SF_STICKY_NOTE = f"{{{SF_NAMESPACE}}}sticky-note"
SF_PARAGRAPH = f"{{{SF_NAMESPACE}}}p"

SF_ATTR_NAME = f"{{{SF_NAMESPACE}}}name"
SF_ATTR_NUMCOLS = f"{{{SF_NAMESPACE}}}numcols"
SF_ATTR_NUMROWS = f"{{{SF_NAMESPACE}}}numrows"
SF_ATTR_HEADER_ROWS = f"{{{SF_NAMESPACE}}}num-header-rows"
SF_ATTR_HEADER_COLS = f"{{{SF_NAMESPACE}}}num-header-columns"
SF_ATTR_CELL_COUNT = f"{{{SF_NAMESPACE}}}nc"
SF_ATTR_VALUE = f"{{{SF_NAMESPACE}}}v"
SF_ATTR_CELL_DATE = f"{{{SF_NAMESPACE}}}cell-date"
SF_ATTR_COL_SPAN = f"{{{SF_NAMESPACE}}}col-span"

SFA_ATTR_STRING = f"{{{SFA_NAMESPACE}}}s"
SFA_ATTR_TEXT = f"{{{SFA_NAMESPACE}}}string"
SFA_ATTR_ID = f"{{{SFA_NAMESPACE}}}ID"
SFA_ATTR_IDREF = f"{{{SFA_NAMESPACE}}}IDREF"
SFA_ATTR_X = f"{{{SFA_NAMESPACE}}}x"
SFA_ATTR_Y = f"{{{SFA_NAMESPACE}}}y"
SFA_ATTR_W = f"{{{SFA_NAMESPACE}}}w"
SFA_ATTR_H = f"{{{SFA_NAMESPACE}}}h"

TEXT_CELL = f"{{{SF_NAMESPACE}}}t"
NUMBER_CELL = f"{{{SF_NAMESPACE}}}n"
DATE_CELL = f"{{{SF_NAMESPACE}}}d"
BOOL_CELL = f"{{{SF_NAMESPACE}}}b"
CHECKBOX_CELL = f"{{{SF_NAMESPACE}}}cb"
POPUP_CELL = f"{{{SF_NAMESPACE}}}pm"
FORMULA_CELL = f"{{{SF_NAMESPACE}}}f"
SPANNED_CELL = f"{{{SF_NAMESPACE}}}s"
"""The cell elements a datasource is built from.

Anything else there — ``sf:g`` for an empty cell — holds no value of its own and
only takes up its place in the row.
"""

RESULT_NUMBER = f"{{{SF_NAMESPACE}}}rn"
RESULT_TEXT = f"{{{SF_NAMESPACE}}}rt"
RESULT_DATE = f"{{{SF_NAMESPACE}}}rd"
RESULT_BOOL = f"{{{SF_NAMESPACE}}}rb"
"""The cached results a formula cell keeps beside its expression."""

CHART_SHARE_SUFFIX = ".chrtshr"
"""Extension of the member holding one chart's cached data.

A chart bound to a table keeps its ``sf:chart-data`` empty and its numbers in a
property list of its own, named after the chart's entity id. That is the only
place the plotted values survive when the chart is read without recomputing its
formula.
"""

SHARE_COLUMNS_KEY = "columns"
SHARE_ROWS_KEY = "rows"
SHARE_ROW_NAME_KEY = "rowName"
SHARE_ROW_VALUES_KEY = "rowValues"
"""Keys of a chart share: its series, and one entry per category."""

MAX_TABLE_CELLS = 4_000_000
"""Cells one table may declare before it is rejected as implausible."""


def read_content(
    archive: zipfile.ZipFile,
    member: str,
    max_total_bytes: int,
    max_file_bytes: int,
    document_hash: str,
) -> list[Sheet]:
    """Read the sheets of an iWork '09 document out of its ``index.xml``.

    Args:
        archive: The open ``.numbers`` container, which also holds the chart data.
        member: The name of its index member.
        max_total_bytes: The largest index this is willing to decompress to.
        max_file_bytes: The largest chart share this is willing to read.
        document_hash: The document's hash, for error messages.

    Returns:
        The document's sheets, in document order.

    Raises:
        DocumentLoadError: If the index cannot be decompressed or parsed.
    """
    root = parse_index(archive, member, max_total_bytes, document_hash)
    shares = read_chart_shares(archive, max_file_bytes)

    sheets: list[Sheet] = []
    for workspace in root.iter(LS_WORKSPACE):
        tables = [
            table
            for info in workspace.iter(SF_TABULAR_INFO)
            if (table := read_table(info)) is not None
        ]
        charts = [
            chart
            for info in workspace.iter(SF_CHART_INFO)
            if (chart := read_chart(info, shares)) is not None
        ]
        comments = [
            comment
            for note in workspace.iter(SF_STICKY_NOTE)
            if (comment := read_comment(note)) is not None
        ]
        tables.sort(key=reading_order)
        charts.sort(key=reading_order)
        comments.sort(key=reading_order)
        sheets.append(
            Sheet(
                name=workspace.get(LS_ATTR_WORKSPACE_NAME) or "",
                tables=tables,
                charts=charts,
                comments=comments,
            )
        )
    return sheets


def parse_index(
    archive: zipfile.ZipFile, member: str, max_total_bytes: int, document_hash: str
) -> Element:
    """Decompress and parse the ``index.xml`` of an iWork '09 document.

    Args:
        archive: The open ``.numbers`` container.
        member: The name of its index member.
        max_total_bytes: The largest index this is willing to decompress to.
        document_hash: The document's hash, for error messages.

    Returns:
        The parsed root element.

    Raises:
        DocumentLoadError: If the member cannot be decompressed or parsed.
    """
    raw = archive.read(member)
    if member.endswith(".gz"):
        # max_total_bytes only counts the stored size of a gzipped member, so a
        # small index.xml.gz could otherwise expand without bound. Cap the
        # output instead of using gzip.decompress, which has no limit.
        limit = min(MAX_LEGACY_XML_BYTES, max_total_bytes)
        try:
            decompressor = zlib.decompressobj(wbits=31)
            raw = decompressor.decompress(raw, limit)
            if decompressor.unconsumed_tail:
                raise DocumentLoadError(
                    f"'{member}' in Numbers document with hash {document_hash} "
                    f"expands beyond the {limit} byte limit."
                )
        except zlib.error as exc:
            raise DocumentLoadError(
                f"Could not decompress '{member}' in Numbers document with hash "
                f"{document_hash}."
            ) from exc

    try:
        return ET.fromstring(raw)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not parse '{member}' in Numbers document with hash {document_hash}."
        ) from exc


def read_chart_shares(archive: zipfile.ZipFile, max_file_bytes: int) -> dict[str, dict]:
    """Read the cached data of every chart in the container.

    Args:
        archive: The open ``.numbers`` container.
        max_file_bytes: The largest share this is willing to read.

    Returns:
        One property list per chart, keyed by the chart's entity id. A share that
        is missing or unreadable is left out rather than failing the document,
        since the chart still has its names without it.
    """
    shares: dict[str, dict] = {}
    for info in archive.infolist():
        if not info.filename.endswith(CHART_SHARE_SUFFIX):
            continue
        if info.file_size > max_file_bytes:
            _log.warning(
                "Skipping chart data '%s': %d bytes exceeds max_file_bytes.",
                info.filename,
                info.file_size,
            )
            continue
        try:
            share = plistlib.loads(archive.read(info))
        except Exception:
            _log.debug("Could not read chart data '%s'.", info.filename)
            continue
        if isinstance(share, dict):
            shares[info.filename[: -len(CHART_SHARE_SUFFIX)]] = share
    return shares


def read_table(info: Element) -> Table | None:
    """Build one table from an ``sf:tabular-info``."""
    model = info.find(SF_TABULAR_MODEL)
    if model is None:
        return None
    grid = model.find(SF_GRID)
    if grid is None:
        return None

    num_rows = int_attr(grid, SF_ATTR_NUMROWS) or 0
    num_cols = int_attr(grid, SF_ATTR_NUMCOLS) or 0
    if num_rows <= 0 or num_cols <= 0:
        return None
    if num_rows * num_cols > MAX_TABLE_CELLS:
        _log.warning(
            "Skipping a Numbers table declaring %d rows by %d columns.",
            num_rows,
            num_cols,
        )
        return None

    return Table(
        name=model.get(SF_ATTR_NAME) or "",
        num_rows=num_rows,
        num_cols=num_cols,
        header_rows=int_attr(model, SF_ATTR_HEADER_ROWS) or 0,
        header_cols=int_attr(model, SF_ATTR_HEADER_COLS) or 0,
        cells=read_cells(grid, num_rows, num_cols),
        bbox=frame(info),
    )


def read_cells(grid: Element, num_rows: int, num_cols: int) -> list[Cell]:
    """Place a datasource's flat cell list back onto its grid.

    The datasource holds one element per stored cell, in row-major order and with
    no coordinates of its own; what puts them back is the occupancy counts on the
    grid lines. Each ``sf:grid-row`` says how many of the entries belong to that
    row, and each ``sf:grid-column`` how many rows store anything in that column
    — which is what places the entries of a row that does not fill it: a column
    whose quota is already spent is skipped over rather than written to, so a
    sparse row lands where the spreadsheet shows it instead of packing left.

    Two elements are counted differently from the rest. A cell with
    ``sf:col-span`` takes up every column it covers, and an ``sf:s`` placeholder,
    which stands where a merge has swallowed a cell, takes up its column without
    ever having been counted in that column's quota. A merge reaching down into
    later rows is left as the one cell that starts it, since a row on its own does
    not say how far down it goes.

    Args:
        grid: The table's ``sf:grid`` element.
        num_rows: How many rows the grid declares.
        num_cols: How many columns the grid declares.

    Returns:
        The cells that hold a value, in document order.
    """
    datasource = grid.find(SF_DATASOURCE)
    rows = grid.find(SF_ROWS)
    if datasource is None or rows is None:
        return []

    entries = list(datasource)
    quotas = column_quotas(grid, num_cols, len(entries))
    cells: list[Cell] = []
    consumed = 0

    for row_index, row in enumerate(rows.iter(SF_GRID_ROW)):
        if row_index >= num_rows:
            break
        count = max(0, int_attr(row, SF_ATTR_CELL_COUNT) or 0)
        cells.extend(
            place_row(entries[consumed : consumed + count], row_index, num_cols, quotas)
        )
        consumed += count

    return cells


def place_row(
    entries: list[Element],
    row_index: int,
    num_cols: int,
    quotas: list[int] | None,
) -> list[Cell]:
    """Lay one row's entries across the grid, spending each column's quota."""
    cells: list[Cell] = []
    column = 0

    for entry in entries:
        if entry.tag == SPANNED_CELL:
            # Stands where a merge has swallowed a cell: it holds the column but
            # was never counted against that column's quota.
            column += 1
            continue

        if quotas is not None:
            while column < num_cols and quotas[column] == 0:
                column += 1
        if column >= num_cols:
            break

        col_span = min(
            max(1, int_attr(entry, SF_ATTR_COL_SPAN) or 1), num_cols - column
        )
        if quotas is not None:
            for spanned in range(column, column + col_span):
                quotas[spanned] -= 1

        text = cell_text(entry)
        if text:
            cells.append(Cell(row=row_index, col=column, text=text, col_span=col_span))
        column += col_span

    return cells


def column_quotas(grid: Element, num_cols: int, entry_count: int) -> list[int] | None:
    """Read how many entries each column stores, or None when they do not add up.

    The counts only place cells if they describe the datasource that is actually
    there, so a document whose totals disagree with it falls back to laying each
    row out from its first column.

    Args:
        grid: The table's ``sf:grid`` element.
        num_cols: How many columns the grid declares.
        entry_count: How many entries the datasource holds.

    Returns:
        One remaining count per column, or None when they cannot be trusted.
    """
    columns = grid.find(SF_COLUMNS)
    if columns is None:
        return None

    quotas = [
        max(0, int_attr(column, SF_ATTR_CELL_COUNT) or 0)
        for column in columns.iter(SF_GRID_COLUMN)
    ]
    if len(quotas) != num_cols or sum(quotas) > entry_count:
        return None
    return quotas


def cell_text(cell: Element) -> str | None:
    """Render one cell, or None when it holds nothing readable."""
    tag = cell.tag
    if tag == TEXT_CELL:
        return text_value(cell)
    if tag in (NUMBER_CELL, CHECKBOX_CELL, BOOL_CELL):
        value = float_attr(cell, SF_ATTR_VALUE)
        if value is None:
            return None
        return format_number(value) if tag == NUMBER_CELL else format_bool(value)
    if tag == DATE_CELL:
        seconds = float_attr(cell, SF_ATTR_CELL_DATE)
        return format_date(seconds) if seconds is not None else None
    if tag == POPUP_CELL:
        return popup_text(cell)
    if tag == FORMULA_CELL:
        return formula_text(cell)
    # sf:g is an empty cell; it only holds the place its column takes up.
    return None


def text_value(cell: Element) -> str | None:
    """Read a text cell, whose string is an attribute or an inline run."""
    content = cell.find(SF_CELL_TEXT)
    if content is None:
        return None
    text = content.get(SFA_ATTR_STRING)
    if text is None:
        text = "".join(content.itertext())
    return text.strip() or None


def popup_text(cell: Element) -> str | None:
    """Read a pop-up menu cell, which points at whichever choice is selected."""
    reference = cell.find(SF_PROXIED_CELL)
    if reference is None:
        return None
    chosen = reference.get(SFA_ATTR_IDREF)
    if chosen is None:
        return None
    for choice in cell.iter():
        if choice.get(SFA_ATTR_ID) == chosen:
            return cell_text(choice)
    return None


def formula_text(cell: Element) -> str | None:
    """Read a formula cell's cached result rather than its expression.

    Docling records what the spreadsheet shows, and what a formula cell shows is
    the value Numbers last computed for it, which it caches beside the formula.
    """
    result = cell.find(SF_RESULT)
    if result is None:
        return None
    for value in result:
        if value.tag == RESULT_NUMBER:
            number = float_attr(value, SF_ATTR_VALUE)
            return format_number(number) if number is not None else None
        if value.tag == RESULT_TEXT:
            return text_value(value)
        if value.tag == RESULT_DATE:
            seconds = float_attr(value, SF_ATTR_CELL_DATE)
            return format_date(seconds) if seconds is not None else None
        if value.tag == RESULT_BOOL:
            flag = float_attr(value, SF_ATTR_VALUE)
            return format_bool(flag) if flag is not None else None
    return None


def read_chart(info: Element, shares: dict[str, dict]) -> Chart | None:
    """Build one chart from an ``sf:chart-info``.

    A chart bound to a table leaves ``sf:chart-data`` empty and keeps the values
    it last plotted in a share of its own, so the names come from the chart and
    the numbers from the share.

    Args:
        info: The chart element.
        shares: The cached chart data, keyed by chart entity id.

    Returns:
        The chart, or None when it names neither a category nor a series.
    """
    categories = strings(info.find(f".//{SF_CHART_ROW_NAMES}"))
    series = strings(info.find(f".//{SF_CHART_COLUMN_NAMES}"))
    values: list[list[Decimal | float | None]] = []

    entity = info.find(f".//{SF_ENTITY_ID}")
    key = entity.get(SFA_ATTR_TEXT) if entity is not None else None
    share = shares.get(key or "")
    if share is not None:
        # The share is what the chart last drew, so where the two disagree — a
        # stale name left behind in the element, say — the share wins.
        shared_series = [str(name) for name in share.get(SHARE_COLUMNS_KEY, [])]
        if shared_series:
            series = shared_series
        rows = share.get(SHARE_ROWS_KEY) or []
        if rows:
            categories = [str(row.get(SHARE_ROW_NAME_KEY, "")) for row in rows]
            values = [points(row) for row in rows]

    if not categories and not series:
        return None

    name = info.find(f".//{SF_CHART_NAME}")
    return Chart(
        name=(name.get(SFA_ATTR_TEXT) or "" if name is not None else "").strip(),
        categories=categories,
        series=series,
        values=values,
        bbox=frame(info),
    )


def points(row: dict) -> list[Decimal | float | None]:
    """Read one category's plotted values, leaving the gaps empty."""
    return [
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
        for value in row.get(SHARE_ROW_VALUES_KEY, [])
    ]


def strings(container: Element | None) -> list[str]:
    """Read a list of ``sf:string`` children as plain text."""
    if container is None:
        return []
    return [child.get(SFA_ATTR_TEXT) or "" for child in container]


def read_comment(note: Element) -> Comment | None:
    """Build one comment from an ``sf:sticky-note``.

    The note keeps its text and nothing else: iWork '09 recorded neither who
    wrote a sticky note nor when.
    """
    text = "\n".join(
        line
        for paragraph in note.iter(SF_PARAGRAPH)
        if (line := "".join(paragraph.itertext()).strip())
    )
    if not text:
        return None
    return Comment(text=text, author="", timestamp=None, bbox=frame(note))


def frame(element: Element) -> BoundingBox | None:
    """Read where a drawable sits on its sheet, in points."""
    geometry = element.find(SF_GEOMETRY)
    if geometry is None:
        return None
    position = geometry.find(SF_POSITION)
    size = geometry.find(SF_SIZE)
    if position is None or size is None:
        return None

    left = float_attr(position, SFA_ATTR_X) or 0.0
    top = float_attr(position, SFA_ATTR_Y) or 0.0
    width = float_attr(size, SFA_ATTR_W) or 0.0
    height = float_attr(size, SFA_ATTR_H) or 0.0
    return BoundingBox(
        l=left,
        t=top,
        r=left + width,
        b=top + height,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def int_attr(element: Element, name: str) -> int | None:
    """Read an integer attribute, tolerating absent or malformed values."""
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def float_attr(element: Element, name: str) -> float | None:
    """Read a floating point attribute, tolerating absent or malformed values."""
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
