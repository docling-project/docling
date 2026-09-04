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

import zipfile
from collections.abc import Callable
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

CELL_TYPE_TEXT = 3

CELL_TYPE_RICH_TEXT = 9
"""Value types of a packed cell, in byte 1, that carry text."""

CELL_KEY_OFFSET = 16
"""Where a version 4 cell keeps the key of its string."""

CELL_FLAGS_OFFSET = 8

CELL_VALUES_OFFSET = 12
"""Where a version 5 cell keeps its flags, and where its values begin.

The flags say which values are present; each one that is takes a fixed width,
so the position of any of them depends on all the ones before it.
"""

CELL_FLAG_STRING = 0x8

CELL_FLAG_RICH_TEXT = 0x10

CELL_VALUE_WIDTHS = (
    (0x1, 16),
    (0x2, 8),
    (0x4, 8),
    (CELL_FLAG_STRING, 4),
    (CELL_FLAG_RICH_TEXT, 4),
)
"""The values a version 5 cell may hold, in the order they are laid out.

A decimal, a double and a duration come first, then the keys of the string and
the rich text a cell may reference. Nothing after the rich text key is needed,
so the walk stops there.
"""


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
    store = safe_fields(store_raw)
    values = cell_values(store, objects)

    cells: list[TableCell] = []
    for tile in tiles(store, objects):
        cells.extend(
            tile_cells(
                tile,
                values,
                num_cols,
                header_rows if isinstance(header_rows, int) else 0,
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


def tiles(
    store: dict[int, list[int | bytes]], objects: dict[int, IWAObject]
) -> list[IWAObject]:
    """Resolve the tiles a table's data store points at."""
    tiles: list[IWAObject] = []
    container = store.get(STORE_TILES_FIELD, [None])[0]
    if not isinstance(container, bytes):
        return tiles

    for entry in safe_fields(container).get(1, []):
        if not isinstance(entry, bytes):
            continue
        reference = safe_fields(entry).get(2, [None])[0]
        target = read_reference(reference) if isinstance(reference, bytes) else None
        tile = objects.get(target) if target is not None else None
        if tile is not None and tile.message_type == TST_TILE:
            tiles.append(tile)
    return tiles


def tile_cells(
    tile: IWAObject, values: CellValues, num_cols: int, header_rows: int
) -> list[TableCell]:
    """Read one tile's cells, placing them by each row's per-column offsets."""
    cells: list[TableCell] = []

    for row_message in safe_fields(tile.payload).get(TILE_ROWS_FIELD, []):
        if not isinstance(row_message, bytes):
            continue
        row = safe_fields(row_message)
        row_index = row.get(ROW_INDEX_FIELD, [None])[0]
        storage = row.get(ROW_WIDE_STORAGE_FIELD, [None])[0]
        offsets = row.get(ROW_WIDE_OFFSETS_FIELD, [None])[0]
        scale = 4 if row.get(ROW_WIDE_OFFSETS_FLAG, [0])[0] else 1
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            storage = row.get(ROW_STORAGE_FIELD, [None])[0]
            offsets = row.get(ROW_OFFSETS_FIELD, [None])[0]
            scale = 1
        if not isinstance(row_index, int):
            continue
        if not isinstance(storage, bytes) or not isinstance(offsets, bytes):
            continue

        for column in range(min(num_cols, len(offsets) // 2)):
            start = int.from_bytes(
                offsets[column * 2 : column * 2 + 2], "little", signed=True
            )
            text = cell_text(storage, start * scale, values)
            if text is None:
                continue
            cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + 1,
                    start_col_offset_idx=column,
                    end_col_offset_idx=column + 1,
                    column_header=row_index < header_rows,
                )
            )

    return cells


def cell_text(storage: bytes, start: int, values: CellValues) -> str | None:
    """Read one packed cell, or None when there is nothing readable there.

    Only the layouts that carry text are decoded. Any other value type — a
    number, a date, a formula result — is skipped rather than guessed at from
    bytes whose meaning has not been established against a real document.

    Args:
        storage: The row's packed cell buffer.
        start: Where in the buffer this cell begins.
        values: The table's shared value lists.

    Returns:
        The cell's text, or None when it holds none.
    """
    if start < 0 or start + CELL_VALUES_OFFSET > len(storage):
        return None

    version = storage[start]
    if version == CELL_VERSION_LEGACY:
        if storage[start + 1] != CELL_TYPE_TEXT:
            return None
        key_at = start + CELL_KEY_OFFSET
        if key_at + 4 > len(storage):
            return None
        return values.strings.get(read_uint32(storage, key_at))

    if version != CELL_VERSION_CURRENT:
        return None
    if storage[start + 1] not in (CELL_TYPE_TEXT, CELL_TYPE_RICH_TEXT):
        return None

    flags = read_uint32(storage, start + CELL_FLAGS_OFFSET)
    offset = start + CELL_VALUES_OFFSET
    for flag, width in CELL_VALUE_WIDTHS:
        if not flags & flag:
            continue
        if offset + width > len(storage):
            return None
        if flag == CELL_FLAG_STRING:
            return values.strings.get(read_uint32(storage, offset))
        if flag == CELL_FLAG_RICH_TEXT:
            return values.rich_text.get(read_uint32(storage, offset))
        offset += width
    return None


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
