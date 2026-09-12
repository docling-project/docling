# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the object graph of a Numbers 3+ (2013 onwards) document.

The route through the graph is short: ``TN.DocumentArchive`` lists the sheets,
each ``TN.SheetArchive`` lists what is drawn on it, and every drawable is a
table, a chart or a sticky note. Tables themselves are the same ``TST`` archives
Pages embeds, so :mod:`docling.backend.iwork.tables` reads them; what is added
here is everything around them — the sheets, the frames that position them, and
the values a spreadsheet cell holds that a Pages cell never does.

Only the message and field numbers are format knowledge; the container layer
lives in :mod:`docling.backend.iwork.iwa`.
"""

import logging
import struct
import zipfile
from decimal import Decimal

from docling_core.types.doc import BoundingBox, CoordOrigin

from docling.backend.iwork import tables
from docling.backend.iwork.iwa import IWAObject, iter_objects, read_reference
from docling.backend.iwork.numbers_content import (
    Cell,
    Chart,
    Comment,
    Sheet,
    Table,
    format_bool,
    format_date,
    format_duration,
    format_number,
    moment,
    reading_order,
)
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

TN_DOCUMENT_ARCHIVE = 1
"""Message type of ``TN.DocumentArchive``, the root of a Numbers document."""

TN_SHEET_ARCHIVE = 2
"""Message type of ``TN.SheetArchive``, one tab of the document."""

TN_CHART_INFO = 5021
"""Message type of the archive placing a chart on a sheet.

Its drawable half carries the frame; the chart itself rides along in the
extension field the writer appends after the numbered ones.
"""

TN_COMMENT_INFO = 2014
"""Message type of the archive placing a comment on a sheet.

Numbers calls these sticky notes. A comment anchored to a cell is stored in a
list beside the table instead, and is not read.
"""

TSK_ANNOTATION = 3056
"""Message type of the annotation a comment archive points at."""

TSK_AUTHOR = 212
"""Message type of the author archive an annotation is attributed to."""

DOCUMENT_SHEETS_FIELD = 1
"""Field of ``TN.DocumentArchive`` listing its sheets."""

SHEET_NAME_FIELD = 1

SHEET_DRAWABLES_FIELD = 2
"""Fields of ``TN.SheetArchive``: its name, and what is drawn on it."""

INFO_SUPER_FIELD = 1

INFO_MODEL_FIELD = 2
"""Fields of ``TST.TableInfoArchive``: its drawable super, and its table."""

TABLE_NAME_FIELD = 8

TABLE_HEADER_COLS_FIELD = 10
"""Fields of ``TST.TableModelArchive`` a Pages table has no use for.

Numbers names every table and can freeze columns as well as rows, so both are
read here rather than in the shared table layer.
"""

DRAWABLE_GEOMETRY_FIELD = 1

GEOMETRY_POSITION_FIELD = 1

GEOMETRY_SIZE_FIELD = 2

POINT_X_FIELD = 1

POINT_Y_FIELD = 2
"""Fields leading from a drawable to where it sits on the sheet, in points."""

INFO_CHART_FIELD = 10000
"""Extension field of a chart's archive holding the chart itself."""

CHART_DATA_FIELD = 7

CHART_CATEGORY_FIELD = 1

CHART_SERIES_FIELD = 2

CHART_ROW_FIELD = 3

CHART_POINT_FIELD = 1

CHART_VALUE_FIELD = 1
"""Fields of a chart's cached data.

The data is stored the way the chart reads it rather than the way the table it
came from is laid out: the category names, then the series names, then one entry
per category holding one point per series. A point wraps its value rather than
being one, so a gap in a series is a point with nothing in it.
"""

COMMENT_ANNOTATION_FIELD = 2

ANNOTATION_TEXT_FIELD = 1

ANNOTATION_TIME_FIELD = 2

ANNOTATION_AUTHOR_FIELD = 3

AUTHOR_NAME_FIELD = 1
"""Fields leading from a comment archive to its text, date and author."""

MAX_TABLE_CELLS = 4_000_000
"""Cells a single table may claim before it is rejected as implausible.

The row and column counts come from the document, so a corrupt or hostile one
can declare a grid far larger than it stores. Numbers' own ceiling is a million
cells per table.
"""


def read_content(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    max_file_bytes: int,
    document_hash: str,
) -> list[Sheet]:
    """Read the sheets of a Numbers 3+ document out of its IWA object graph.

    Args:
        archive: The open ``.numbers`` container.
        infos: Its members.
        max_file_bytes: The largest member this is willing to decompress.
        document_hash: The document's hash, for error messages.

    Returns:
        The document's sheets, in document order.

    Raises:
        DocumentLoadError: If a member is too large, or the object graph has no
            document archive.
    """
    objects: dict[int, IWAObject] = {}
    for info in infos:
        if not info.filename.endswith(".iwa"):
            continue
        if info.file_size > max_file_bytes:
            raise DocumentLoadError(
                f"Numbers archive member {info.filename} is {info.file_size} "
                f"bytes, exceeding the max_file_bytes limit of {max_file_bytes}."
            )
        for obj in iter_objects(archive.read(info)):
            objects[obj.identifier] = obj

    document = next(
        (o for o in objects.values() if o.message_type == TN_DOCUMENT_ARCHIVE), None
    )
    if document is None:
        raise DocumentLoadError(
            f"Numbers document with hash {document_hash} has no "
            "TN.DocumentArchive; the container may be corrupt or "
            "password-protected."
        )

    sheets: list[Sheet] = []
    for reference in tables.safe_fields(document.payload).get(
        DOCUMENT_SHEETS_FIELD, []
    ):
        sheet = resolve(reference, objects, TN_SHEET_ARCHIVE)
        if sheet is not None:
            sheets.append(read_sheet(sheet, objects))
    return sheets


def read_sheet(sheet: IWAObject, objects: dict[int, IWAObject]) -> Sheet:
    """Read one sheet's name and the tables, charts and notes drawn on it."""
    fields = tables.safe_fields(sheet.payload)

    sheet_tables: list[Table] = []
    charts: list[Chart] = []
    comments: list[Comment] = []
    for reference in fields.get(SHEET_DRAWABLES_FIELD, []):
        drawable = dereference(reference, objects)
        if drawable is None:
            continue
        if drawable.message_type == tables.TST_TABLE_INFO:
            table = read_table(drawable, objects)
            if table is not None:
                sheet_tables.append(table)
        elif drawable.message_type == TN_CHART_INFO:
            chart = read_chart(drawable)
            if chart is not None:
                charts.append(chart)
        elif drawable.message_type == TN_COMMENT_INFO:
            comment = read_comment(drawable, objects)
            if comment is not None:
                comments.append(comment)

    sheet_tables.sort(key=reading_order)
    charts.sort(key=reading_order)
    comments.sort(key=reading_order)
    return Sheet(
        name=text_of(fields.get(SHEET_NAME_FIELD, [None])[0]) or "",
        tables=sheet_tables,
        charts=charts,
        comments=comments,
    )


def read_table(info: IWAObject, objects: dict[int, IWAObject]) -> Table | None:
    """Build one table from the archive placing it on its sheet.

    Args:
        info: The ``TST.TableInfoArchive`` for this table.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The table, or None when it does not resolve to a readable model.
    """
    info_fields = tables.safe_fields(info.payload)
    model = resolve(
        info_fields.get(INFO_MODEL_FIELD, [None])[0], objects, tables.TST_TABLE_MODEL
    )
    if model is None:
        return None

    fields = tables.safe_fields(model.payload)
    num_rows = fields.get(tables.TABLE_ROWS_FIELD, [None])[0]
    num_cols = fields.get(tables.TABLE_COLS_FIELD, [None])[0]
    if not isinstance(num_rows, int) or not isinstance(num_cols, int):
        return None
    if num_rows <= 0 or num_cols <= 0:
        return None
    if num_rows * num_cols > MAX_TABLE_CELLS:
        _log.warning(
            "Skipping a Numbers table declaring %d rows by %d columns.",
            num_rows,
            num_cols,
        )
        return None

    store_raw = fields.get(tables.TABLE_DATA_STORE_FIELD, [None])[0]
    store = tables.safe_fields(store_raw) if isinstance(store_raw, bytes) else {}
    values = tables.cell_values(store, objects)

    return Table(
        name=text_of(fields.get(TABLE_NAME_FIELD, [None])[0]) or "",
        num_rows=num_rows,
        num_cols=num_cols,
        header_rows=count(fields.get(tables.TABLE_HEADER_ROWS_FIELD, [None])[0]),
        header_cols=count(fields.get(TABLE_HEADER_COLS_FIELD, [None])[0]),
        cells=read_cells(store, objects, values, num_rows, num_cols),
        bbox=frame(info_fields.get(INFO_SUPER_FIELD, [None])[0]),
    )


def read_cells(
    store: dict[int, list[int | bytes]],
    objects: dict[int, IWAObject],
    values: tables.CellValues,
    num_rows: int,
    num_cols: int,
) -> list[Cell]:
    """Read every cell of a table that holds something, in tile order.

    Args:
        store: The table's decoded data store.
        objects: Every object in the document, keyed by identifier.
        values: The table's shared value lists.
        num_rows: How many rows the table declares.
        num_cols: How many columns the table declares.

    Returns:
        The cells that hold something, rendered to text.
    """
    cells: list[Cell] = []
    for placed in tables.placements(store, objects):
        if placed.row >= num_rows or placed.col >= num_cols:
            continue
        rendered = render(tables.cell(placed.storage, placed.start, values))
        if rendered:
            cells.append(Cell(row=placed.row, col=placed.col, text=rendered))
    return cells


def render(decoded: tables.Cell | None) -> str | None:
    """Turn a decoded cell into the text the spreadsheet shows in it.

    Args:
        decoded: The cell as the table layer read it.

    Returns:
        The text, or None for an empty cell or a value type this reader has not
        been shown how to render.
    """
    if decoded is None or decoded.type == tables.CELL_TYPE_EMPTY:
        return None
    if decoded.type in (tables.CELL_TYPE_TEXT, tables.CELL_TYPE_RICH_TEXT):
        return decoded.text
    if decoded.number is None:
        return None
    if decoded.type in (tables.CELL_TYPE_NUMBER, tables.CELL_TYPE_CURRENCY):
        return format_number(decoded.number)
    if decoded.type == tables.CELL_TYPE_DATE:
        return format_date(float(decoded.number))
    if decoded.type == tables.CELL_TYPE_DURATION:
        return format_duration(float(decoded.number))
    if decoded.type == tables.CELL_TYPE_BOOL:
        return format_bool(float(decoded.number))
    return None


def read_chart(info: IWAObject) -> Chart | None:
    """Read one chart and the data Numbers cached for it.

    A chart keeps its own copy of what it plots, so its numbers can be read
    without following the formula back to the table they came from — and are
    still there when that table has since been deleted.

    Args:
        info: The archive placing the chart on its sheet.

    Returns:
        The chart, or None when it carries no data to plot.
    """
    fields = tables.safe_fields(info.payload)
    chart = nested(fields, INFO_CHART_FIELD)
    data = nested(chart, CHART_DATA_FIELD)

    categories = [text_of(name) or "" for name in data.get(CHART_CATEGORY_FIELD, [])]
    series = [text_of(name) or "" for name in data.get(CHART_SERIES_FIELD, [])]
    if not categories and not series:
        return None

    values: list[list[Decimal | float | None]] = []
    for row in data.get(CHART_ROW_FIELD, []):
        if not isinstance(row, bytes):
            continue
        points: list[Decimal | float | None] = []
        for point in tables.safe_fields(row).get(CHART_POINT_FIELD, []):
            raw = (
                tables.safe_fields(point).get(CHART_VALUE_FIELD, [None])[0]
                if isinstance(point, bytes)
                else None
            )
            points.append(double(raw) if isinstance(raw, bytes) else None)
        values.append(points)

    return Chart(
        name="",
        categories=categories,
        series=series,
        values=values,
        bbox=frame(fields.get(INFO_SUPER_FIELD, [None])[0]),
    )


def read_comment(info: IWAObject, objects: dict[int, IWAObject]) -> Comment | None:
    """Read one sticky note: its text, and who left it when.

    Args:
        info: The archive placing the comment on its sheet.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The comment, or None when it has no text.
    """
    fields = tables.safe_fields(info.payload)
    annotation = resolve(
        fields.get(COMMENT_ANNOTATION_FIELD, [None])[0], objects, TSK_ANNOTATION
    )
    if annotation is None:
        return None

    parsed = tables.safe_fields(annotation.payload)
    text = (text_of(parsed.get(ANNOTATION_TEXT_FIELD, [None])[0]) or "").strip()
    if not text:
        return None

    author = resolve(
        parsed.get(ANNOTATION_AUTHOR_FIELD, [None])[0], objects, TSK_AUTHOR
    )
    name = (
        text_of(tables.safe_fields(author.payload).get(AUTHOR_NAME_FIELD, [None])[0])
        if author is not None
        else None
    )

    return Comment(
        text=text,
        author=name or "",
        timestamp=timestamp(parsed.get(ANNOTATION_TIME_FIELD, [None])[0]),
        bbox=frame(fields.get(INFO_SUPER_FIELD, [None])[0]),
    )


def timestamp(raw: int | bytes | None):
    """Read a ``TSP.Date``, which counts seconds from the Apple epoch."""
    if not isinstance(raw, bytes):
        return None
    seconds = tables.safe_fields(raw).get(1, [None])[0]
    value = double(seconds) if isinstance(seconds, bytes) else None
    return None if value is None else moment(value)


def frame(super_raw: int | bytes | None) -> BoundingBox | None:
    """Read where a drawable sits on its sheet, in points from the top left."""
    if not isinstance(super_raw, bytes):
        return None
    geometry = tables.safe_fields(super_raw).get(DRAWABLE_GEOMETRY_FIELD, [None])[0]
    if not isinstance(geometry, bytes):
        return None
    fields = tables.safe_fields(geometry)
    left, top = pair(fields.get(GEOMETRY_POSITION_FIELD, [None])[0])
    width, height = pair(fields.get(GEOMETRY_SIZE_FIELD, [None])[0])
    return BoundingBox(
        l=left,
        t=top,
        r=left + width,
        b=top + height,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def pair(raw: int | bytes | None) -> tuple[float, float]:
    """Read a ``TSP.Point`` or ``TSP.Size``, both a pair of 32-bit floats."""
    if not isinstance(raw, bytes):
        return (0.0, 0.0)
    fields = tables.safe_fields(raw)
    return (
        float32(fields.get(POINT_X_FIELD, [None])[0]),
        float32(fields.get(POINT_Y_FIELD, [None])[0]),
    )


def float32(raw: int | bytes | None) -> float:
    """Decode one 32-bit protobuf float, treating a malformed one as zero."""
    if not isinstance(raw, bytes) or len(raw) != 4:
        return 0.0
    return float(struct.unpack("<f", raw)[0])


def double(raw: bytes) -> float | None:
    """Decode one 64-bit protobuf double."""
    if len(raw) != 8:
        return None
    return float(struct.unpack("<d", raw)[0])


def dereference(
    reference: int | bytes | None, objects: dict[int, IWAObject]
) -> IWAObject | None:
    """Follow a ``TSP.Reference`` to whatever archive it lands on."""
    if not isinstance(reference, bytes):
        return None
    target = tables.reference_field(b"\x0a" + bytes([len(reference)]) + reference, 1)
    return objects.get(target) if target is not None else None


def resolve(
    reference: int | bytes | None,
    objects: dict[int, IWAObject],
    message_type: int,
) -> IWAObject | None:
    """Follow a ``TSP.Reference``, checking it lands on the expected archive."""
    obj = dereference(reference, objects)
    return obj if obj is not None and obj.message_type == message_type else None


def text_of(raw: int | bytes | None) -> str | None:
    """Decode a UTF-8 string field, tolerating one that is not a string."""
    if not isinstance(raw, bytes):
        return None
    return raw.decode("utf-8", errors="replace")


def count(raw: int | bytes | None) -> int:
    """Read a non-negative count field, treating anything else as zero."""
    return raw if isinstance(raw, int) and raw >= 0 else 0


def nested(
    fields: dict[int, list[int | bytes]], field_no: int
) -> dict[int, list[int | bytes]]:
    """Decode a sub-message of an already-decoded one, or nothing when absent."""
    raw = fields.get(field_no, [None])[0]
    return tables.safe_fields(raw) if isinstance(raw, bytes) else {}
