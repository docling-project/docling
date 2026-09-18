# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for the ``TST`` table archives of an iWork 2013+ document.

TST is Apple's table engine, shared the way ``TSWP`` is its text engine: a table
embedded in a Pages document and a table on a Numbers sheet are the same
archives, laid out the same way, so they are read once here.

A table keeps its geometry on the model, its cell contents in shared value lists,
and the placement of those values in tiles. Cells reference their value by key,
so equal values share one entry — which is why the tiles have to be read rather
than assuming a value list is already in cell order.
"""

import struct
import zipfile
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import NamedTuple

from docling_core.types.doc import TableCell, TableData

from docling.backend.iwork.iwa import IWAObject, read_fields, read_reference
from docling.exceptions import DocumentLoadError


class CellValues(NamedTuple):
    """A table's shared value lists, keyed as its cells reference them.

    Cells reference their contents by key rather than holding them, so two cells
    with the same text share one entry.
    """

    strings: dict[int, str] = {}
    rich_text: dict[int, str] = {}


class Placement(NamedTuple):
    """Where one stored cell sits in a table, and where to find its bytes."""

    row: int
    col: int
    storage: bytes
    start: int


class Cell(NamedTuple):
    """One packed cell, decoded into whichever pieces it turned out to carry.

    A cell holds at most one of these: text from one of the shared value lists,
    or a number. What the number *means* is the type's business — a date and a
    duration are both spans of seconds, and a boolean is a number that is either
    positive or not — so it is left as read and named by ``type``.
    """

    type: int
    text: str | None = None
    number: Decimal | float | None = None


TST_TABLE_INFO = 6000
"""Message type of ``TST.TableInfoArchive``, one table placed in a document."""

TST_TABLE_MODEL = 6001
"""Message type of ``TST.TableModelArchive``, the table itself."""

TST_TILE = 6002
"""Message type of ``TST.Tile``, which lays a table's cells out into rows."""

TST_DATA_LIST = 6005
"""Message type of ``TST.TableDataList``, a table's shared value table.

Cells reference their contents by key rather than holding them, so two cells
with the same text share a single entry.
"""

TABLE_ROWS_FIELD = 6

TABLE_COLS_FIELD = 7

TABLE_HEADER_ROWS_FIELD = 9

TABLE_DATA_STORE_FIELD = 4
"""Fields of ``TST.TableModelArchive``: geometry, header rows, and data store."""

STORE_TILES_FIELD = 3

TILE_LIST_FIELD = 1

TILE_ENTRY_ROW_FIELD = 1

TILE_ENTRY_TILE_FIELD = 2
"""Fields of the tile list: each entry's first row, and the tile itself."""

STORE_STRINGS_FIELD = 4

STORE_RICH_TEXT_FIELD = 17
"""Fields of a table's data store: its tiles, and its two value lists.

A cell holding plain text references the string list; one holding styled text
references the rich text list instead, whose entries point at a whole
``TSWP.StorageArchive``.
"""

LIST_ENTRIES_FIELD = 3

LIST_SEGMENTS_FIELD = 4
"""Fields of ``TST.TableDataList``: its entries, and the segments they spill into.

A list long enough to be split keeps its entries in referenced segments instead,
which have the same entry shape.
"""

ENTRY_KEY_FIELD = 1

ENTRY_STRING_FIELD = 3

ENTRY_RICH_TEXT_FIELD = 9
"""Fields of one value list entry: the key cells reference it by, and its value."""

TST_TEXT_REF = 6218
"""Message type of the indirection a rich text entry points at.

It holds nothing but a reference to the ``TSWP.StorageArchive`` with the text.
"""

RICH_TEXT_STORAGE = 2001
"""Message type of ``TSWP.StorageArchive``, where a styled cell parks its text.

TSWP is Apple's text engine, so a cell whose text carries formatting keeps it in
the same archive the body of a Pages document uses. Only the text is wanted here
— what a table cell is styled with is not recovered — so the archive is read for
that one field rather than through the text engine proper.
"""

RICH_TEXT_FIELD = 3
"""Field of ``TSWP.StorageArchive`` holding the text itself."""

TILE_ROWS_FIELD = 5

ROW_INDEX_FIELD = 1

ROW_STORAGE_FIELD = 3

ROW_OFFSETS_FIELD = 4

ROW_WIDE_STORAGE_FIELD = 6

ROW_WIDE_OFFSETS_FIELD = 7

ROW_WIDE_OFFSETS_FLAG = 8
"""Fields of ``TST.Tile`` and of one of its rows.

A row holds a packed cell buffer plus one ``int16`` offset per column, where a
negative offset marks a column with no cell. Pages 5.2 moved both to their own
fields and started scaling the offsets by four, keeping the older pair in place
for the benefit of releases that could not read the new one, so the newer pair
is preferred when it is there.
"""

CELL_VERSION_LEGACY = 4

CELL_VERSION_CURRENT = 5
"""Storage versions of a packed cell, in byte 0."""

CELL_TYPE_EMPTY = 0

CELL_TYPE_NUMBER = 2

CELL_TYPE_TEXT = 3

CELL_TYPE_DATE = 5

CELL_TYPE_BOOL = 6

CELL_TYPE_DURATION = 7

CELL_TYPE_RICH_TEXT = 9

CELL_TYPE_CURRENCY = 10
"""Value types of a packed cell, in byte 1.

Anything not named here is left undecoded rather than guessed at from bytes
whose meaning has not been established against a real document.
"""

CELL_NUMERIC_TYPES = frozenset(
    {
        CELL_TYPE_NUMBER,
        CELL_TYPE_DATE,
        CELL_TYPE_BOOL,
        CELL_TYPE_DURATION,
        CELL_TYPE_CURRENCY,
    }
)
"""Types whose cell carries a number rather than only identifiers."""

CELL_FLAGS_OFFSET_LEGACY = 4
"""Where a version 4 cell keeps the bitmask of the fields it carries.

Apple never published this layout and it does not describe itself, so it is read
the way a genuine document writes it: one four-byte identifier per set bit, a
single IEEE double in front of the last identifier when the cell's type carries a
value, and the key of a text cell's string *as* that last identifier. The length
that implies is checked against the bytes actually present, so a cell that does
not fit the layout yields nothing rather than misread bytes.
"""

CELL_VALUE_WIDTH = 8

CELL_IDENTIFIER_WIDTH = 4
"""Widths of the two things a version 4 cell appends after its header."""

CELL_FLAGS_OFFSET = 8

CELL_VALUES_OFFSET = 12
"""Where a version 5 cell keeps its flags, and where its values begin.

The flags say which values are present; each one that is takes a fixed width,
so the position of any of them depends on all the ones before it.
"""

CELL_FLAG_STRING = 0x8

CELL_FLAG_RICH_TEXT = 0x10

CELL_FLAG_DECIMAL = 0x1

CELL_FLAG_DOUBLE = 0x2

CELL_FLAG_SECONDS = 0x4

CELL_VALUE_WIDTHS = (
    (CELL_FLAG_DECIMAL, 16),
    (CELL_FLAG_DOUBLE, 8),
    (CELL_FLAG_SECONDS, 8),
    (CELL_FLAG_STRING, 4),
    (CELL_FLAG_RICH_TEXT, 4),
)
"""The values a version 5 cell may hold, in the order they are laid out.

A decimal, a double and a span of seconds come first, then the keys of the string
and the rich text a cell may reference. Nothing after the rich text key is wanted,
so the walk stops there.
"""

DECIMAL128_BIAS = 6176
"""Amount subtracted from a decimal128's stored exponent to get the real one."""


def reference_field(payload: bytes, field: int) -> int | None:
    """Read the object identifier a message's reference field points at."""
    reference = safe_fields(payload).get(field, [None])[0]
    if not isinstance(reference, bytes):
        return None
    return read_reference(reference)


def reference_list(payload: bytes, field: int) -> list[int]:
    """Read the object identifiers a message's repeated reference field holds."""
    identifiers = []
    for reference in safe_fields(payload).get(field, []):
        if not isinstance(reference, bytes):
            continue
        target = read_reference(reference)
        if target is not None:
            identifiers.append(target)
    return identifiers


def table(model: IWAObject, objects: dict[int, IWAObject]) -> TableData | None:
    """Build table data from one ``TST.TableModelArchive``.

    A table keeps its geometry on the model, its cell contents in a shared value
    list, and the placement of those values in tiles. Cells reference their value
    by key, so equal values share one entry — which is why the tiles have to be
    read rather than assuming the value list is already in cell order.

    Args:
        model: The table's ``TST.TableModelArchive``.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The table, or None when nothing readable could be placed in it.
    """
    fields = safe_fields(model.payload)
    num_rows = fields.get(TABLE_ROWS_FIELD, [None])[0]
    num_cols = fields.get(TABLE_COLS_FIELD, [None])[0]
    store_raw = fields.get(TABLE_DATA_STORE_FIELD, [None])[0]
    if not isinstance(num_rows, int) or not isinstance(num_cols, int):
        return None
    if not num_rows or not num_cols or not isinstance(store_raw, bytes):
        return None

    header_rows = fields.get(TABLE_HEADER_ROWS_FIELD, [0])[0]
    if not isinstance(header_rows, int):
        header_rows = 0
    store = safe_fields(store_raw)
    values = cell_values(store, objects)

    cells: list[TableCell] = []
    for placed in placements(store, objects):
        if placed.row >= num_rows or placed.col >= num_cols:
            continue
        text = cell_text(placed.storage, placed.start, values)
        if text is None:
            continue
        cells.append(
            TableCell(
                text=text,
                start_row_offset_idx=placed.row,
                end_row_offset_idx=placed.row + 1,
                start_col_offset_idx=placed.col,
                end_col_offset_idx=placed.col + 1,
                column_header=placed.row < header_rows,
            )
        )

    if not cells:
        return None
    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def cell_values(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> CellValues:
    """Read a table's shared value lists, keyed as its cells reference them.

    Args:
        store: Decoded fields of the table's data store.
        objects: Every object in the document, keyed by identifier.

    Returns:
        The plain strings and the rich text, each keyed by cell reference.
    """
    return CellValues(
        strings=value_list(store, STORE_STRINGS_FIELD, objects, entry_string),
        rich_text=value_list(
            store,
            STORE_RICH_TEXT_FIELD,
            objects,
            entry_rich_text,
        ),
    )


def value_list(
    store: dict[int, list[int | bytes]],
    field: int,
    objects: dict[int, IWAObject],
    decode: Callable[[dict[int, list[int | bytes]], dict[int, IWAObject]], str | None],
) -> dict[int, str]:
    """Read one ``TST.TableDataList``, following any segments it spills into."""
    reference = store.get(field, [None])[0]
    target = read_reference(reference) if isinstance(reference, bytes) else None
    data_list = objects.get(target) if target is not None else None
    if data_list is None or data_list.message_type != TST_DATA_LIST:
        return {}

    payloads = [data_list.payload]
    for segment in reference_list(data_list.payload, LIST_SEGMENTS_FIELD):
        spilled = objects.get(segment)
        if spilled is not None:
            payloads.append(spilled.payload)

    values: dict[int, str] = {}
    for payload in payloads:
        for entry in safe_fields(payload).get(LIST_ENTRIES_FIELD, []):
            if not isinstance(entry, bytes):
                continue
            parsed = safe_fields(entry)
            key = parsed.get(ENTRY_KEY_FIELD, [None])[0]
            value = decode(parsed, objects)
            if isinstance(key, int) and value is not None:
                values[key] = value
    return values


def entry_string(
    entry: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> str | None:
    """Read a value list entry that holds its string directly."""
    value = next(
        (v for v in entry.get(ENTRY_STRING_FIELD, []) if isinstance(v, bytes)), None
    )
    return None if value is None else value.decode("utf-8", errors="replace")


def entry_rich_text(
    entry: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> str | None:
    """Read a value list entry that points at a whole text storage."""
    reference = entry.get(ENTRY_RICH_TEXT_FIELD, [None])[0]
    target = read_reference(reference) if isinstance(reference, bytes) else None
    indirection = objects.get(target) if target is not None else None
    if indirection is None or indirection.message_type != TST_TEXT_REF:
        return None

    storage_id = reference_field(indirection.payload, 1)
    storage = objects.get(storage_id) if storage_id is not None else None
    if storage is None or storage.message_type != RICH_TEXT_STORAGE:
        return None

    text = "".join(
        piece.decode("utf-8", errors="replace")
        for piece in safe_fields(storage.payload).get(RICH_TEXT_FIELD, [])
        if isinstance(piece, bytes)
    )
    return text.strip() or None


def placements(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> Iterator[Placement]:
    """Walk a table's tiles, yielding where each stored cell sits.

    A table taller than one tile is split across several, and each tile numbers
    its rows from its own start rather than from the table's, so the entry's
    first row is added back here.

    Args:
        store: The table's decoded data store.
        objects: Every object in the document, keyed by identifier.

    Yields:
        One placement per stored cell, in tile order.
    """
    container = store.get(STORE_TILES_FIELD, [None])[0]
    if not isinstance(container, bytes):
        return

    for entry in safe_fields(container).get(TILE_LIST_FIELD, []):
        if not isinstance(entry, bytes):
            continue
        parsed = safe_fields(entry)
        first_row = parsed.get(TILE_ENTRY_ROW_FIELD, [0])[0]
        reference = parsed.get(TILE_ENTRY_TILE_FIELD, [None])[0]
        target = read_reference(reference) if isinstance(reference, bytes) else None
        tile = objects.get(target) if target is not None else None
        if tile is None or tile.message_type != TST_TILE:
            continue
        yield from tile_placements(tile, first_row if isinstance(first_row, int) else 0)


def tile_placements(tile: IWAObject, first_row: int) -> Iterator[Placement]:
    """Walk one tile's rows, placing each cell by its per-column offset."""
    for row_message in safe_fields(tile.payload).get(TILE_ROWS_FIELD, []):
        if not isinstance(row_message, bytes):
            continue
        row = safe_fields(row_message)
        row_index = row.get(ROW_INDEX_FIELD, [None])[0]
        if not isinstance(row_index, int):
            continue

        storage = row.get(ROW_WIDE_STORAGE_FIELD, [None])[0]
        offsets = row.get(ROW_WIDE_OFFSETS_FIELD, [None])[0]
        scale = 4 if row.get(ROW_WIDE_OFFSETS_FLAG, [0])[0] else 1
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            storage = row.get(ROW_STORAGE_FIELD, [None])[0]
            offsets = row.get(ROW_OFFSETS_FIELD, [None])[0]
            scale = 1
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            continue

        for column in range(len(offsets) // 2):
            start = int.from_bytes(
                offsets[column * 2 : column * 2 + 2], "little", signed=True
            )
            if start < 0:
                continue
            yield Placement(first_row + row_index, column, storage, start * scale)


def cell_text(storage: bytes, start: int, values: CellValues) -> str | None:
    """Read the text of one packed cell, or None when it holds none.

    Args:
        storage: The row's packed cell buffer.
        start: Where in the buffer this cell begins.
        values: The table's shared value lists.

    Returns:
        The cell's text, or None for an empty cell or one holding a number.
    """
    decoded = cell(storage, start, values)
    return decoded.text if decoded is not None else None


def cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
    """Decode one packed cell, in either of the two layouts Apple has written.

    Args:
        storage: The row's packed cell buffer.
        start: Where in the buffer this cell begins.
        values: The table's shared value lists.

    Returns:
        The cell, or None when the bytes there do not describe one this reader
        recognises.
    """
    if start < 0 or start + CELL_VALUES_OFFSET > len(storage):
        return None

    version = storage[start]
    if version == CELL_VERSION_LEGACY:
        return legacy_cell(storage, start, values)
    if version == CELL_VERSION_CURRENT:
        return current_cell(storage, start, values)
    return None


def legacy_cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
    """Decode a version 4 cell, whose layout has to be inferred from its bitmask.

    See :data:`CELL_FLAGS_OFFSET_LEGACY` for the layout and why it is checked
    rather than trusted.
    """
    cell_type = storage[start + 1]
    flags = read_uint32(storage, start + CELL_FLAGS_OFFSET_LEGACY)
    identifiers = bin(flags).count("1")
    if not identifiers:
        return None

    numeric = cell_type in CELL_NUMERIC_TYPES
    length = (
        CELL_VALUES_OFFSET
        + CELL_IDENTIFIER_WIDTH * identifiers
        + (CELL_VALUE_WIDTH if numeric else 0)
    )
    if start + length > len(storage):
        return None

    last = start + length - CELL_IDENTIFIER_WIDTH
    if numeric:
        return Cell(cell_type, number=read_double(storage, last - CELL_VALUE_WIDTH))
    if cell_type != CELL_TYPE_TEXT:
        return None

    # A key that names nothing in the table means the layout was read wrong, so
    # the cell yields nothing rather than an entry that happens to exist.
    text = values.strings.get(read_uint32(storage, last))
    return None if text is None else Cell(cell_type, text=text)


def current_cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
    """Decode a version 5 cell, whose bitmask says which fields follow."""
    cell_type = storage[start + 1]
    flags = read_uint32(storage, start + CELL_FLAGS_OFFSET)

    offset = start + CELL_VALUES_OFFSET
    decoded = Cell(cell_type)
    for flag, width in CELL_VALUE_WIDTHS:
        if not flags & flag:
            continue
        if offset + width > len(storage):
            return None
        if flag == CELL_FLAG_DECIMAL:
            decoded = decoded._replace(number=read_decimal128(storage, offset))
        elif flag in (CELL_FLAG_DOUBLE, CELL_FLAG_SECONDS):
            decoded = decoded._replace(number=read_double(storage, offset))
        elif flag == CELL_FLAG_STRING:
            decoded = decoded._replace(
                text=values.strings.get(read_uint32(storage, offset))
            )
        elif flag == CELL_FLAG_RICH_TEXT:
            decoded = decoded._replace(
                text=values.rich_text.get(read_uint32(storage, offset))
            )
        offset += width

    return decoded


def read_double(buffer: bytes, at: int) -> float | None:
    """Read one little-endian IEEE double out of a packed cell buffer."""
    field = buffer[at : at + CELL_VALUE_WIDTH]
    if len(field) != CELL_VALUE_WIDTH:
        return None
    return float(struct.unpack("<d", field)[0])


def read_decimal128(buffer: bytes, at: int) -> Decimal | None:
    """Read one IEEE 754-2008 decimal128 in its binary integer encoding.

    Numbers has stored numeric cells this way since 2017, which is why a
    spreadsheet no longer rounds 0.1 the way binary floating point would. The
    encoding is a sign bit, a biased fourteen-bit exponent and a coefficient; the
    combination values that mean an infinity or a NaN are rejected rather than
    rendered.

    Args:
        buffer: The row's packed cell buffer.
        at: Where in the buffer the value begins.

    Returns:
        The value, or None when the bytes do not encode a finite number.
    """
    field = buffer[at : at + 16]
    if len(field) != 16 or field[15] & 0x78 == 0x78:
        return None

    exponent = (((field[15] & 0x7F) << 7) | (field[14] >> 1)) - DECIMAL128_BIAS
    coefficient = field[14] & 0x1
    for byte in reversed(field[:14]):
        coefficient = coefficient * 256 + byte
    if field[15] & 0x80:
        coefficient = -coefficient
    return Decimal(coefficient).scaleb(exponent)


def read_uint32(buffer: bytes, at: int) -> int:
    """Read a little-endian 32-bit value out of a packed cell buffer."""
    return int.from_bytes(buffer[at : at + 4], "little")


def safe_fields(payload: bytes) -> dict[int, list[int | bytes]]:
    """Decode a message, treating an unreadable one as empty.

    The table archives carry sub-messages this reader has no need to understand,
    some of which use wire types the fields it does want never use. Failing the
    whole document over one of them would be wrong.
    """
    try:
        return read_fields(payload)
    except DocumentLoadError:
        return {}
