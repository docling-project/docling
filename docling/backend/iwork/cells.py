# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Reader for one packed cell of a ``TST`` table.

A table does not store its cells as messages. Each row of a tile carries them
packed into one buffer, and a cell inside it is a short header, a bitmask saying
which fields follow, and then those fields — so a cell has to be unpacked by
hand rather than walked like the rest of the object graph.

Apple has written two layouts of it, and both are still in circulation:

* **Version 5**, written since the 2017 release that changed how numbers are
  stored, describes itself: the bitmask says which of a fixed list of fields the
  cell carries, and they follow in that order. A numeric cell holds an IEEE
  754-2008 decimal128, which is why a spreadsheet no longer rounds 0.1 the way
  binary floating point would.
* **Version 4**, written before it, does not describe itself and is not written
  down anywhere. See :data:`CELL_FLAGS_OFFSET_LEGACY` for how it is read, and for
  the checks that make a misreading fail rather than return nonsense.

A document saved by a recent release carries both, and blanks out the version 4
copy it leaves behind for older ones, so the version 5 buffer wins wherever it is
there.
"""

import struct
from decimal import Decimal
from typing import NamedTuple


class CellValues(NamedTuple):
    """A table's shared value lists, keyed as its cells reference them.

    Cells reference their contents by key rather than holding them, so two cells
    with the same text share one entry.
    """

    strings: dict[int, str] = {}
    rich_text: dict[int, str] = {}


class Cell(NamedTuple):
    """One packed cell, decoded into whichever pieces it turned out to carry.

    A cell holds at most one of these: text from one of the shared value lists,
    or a number. What the number *means* is the type's business — a date and a
    duration are both spans of seconds, and a boolean is a number that is either
    positive or not — so it is left as read and named by ``type``.

    ``style`` is the key of the cell's own style in the table's style list, which
    is where an image that fills the cell is named. It is read only from a
    version 5 cell, because where a version 4 cell keeps it is not known.
    """

    type: int
    text: str | None = None
    number: Decimal | float | None = None
    style: int | None = None


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

CELL_FLAG_CELL_STYLE = 0x20

CELL_VALUE_WIDTHS = (
    (CELL_FLAG_DECIMAL, 16),
    (CELL_FLAG_DOUBLE, 8),
    (CELL_FLAG_SECONDS, 8),
    (CELL_FLAG_STRING, 4),
    (CELL_FLAG_RICH_TEXT, 4),
    (CELL_FLAG_CELL_STYLE, 4),
)
"""The values a version 5 cell may hold, in the order they are laid out.

A decimal, a double and a span of seconds come first, then the keys of the string
and the rich text a cell may reference, and then the key of the cell's own style.
Nothing after the style key is wanted, so the walk stops there.
"""

DECIMAL128_BIAS = 6176
"""Amount subtracted from a decimal128's stored exponent to get the real one."""


def iwa_cell_text(storage: bytes, start: int, values: CellValues) -> str | None:
    """Read the text of one packed cell, or None when it holds none.

    Args:
        storage: The row's packed cell buffer.
        start: Where in the buffer this cell begins.
        values: The table's shared value lists.

    Returns:
        The cell's text, or None for an empty cell or one holding a number.
    """
    decoded = iwa_cell(storage, start, values)
    return decoded.text if decoded is not None else None


def iwa_cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
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
        return iwa_legacy_cell(storage, start, values)
    if version == CELL_VERSION_CURRENT:
        return iwa_current_cell(storage, start, values)
    return None


def iwa_legacy_cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
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


def iwa_current_cell(storage: bytes, start: int, values: CellValues) -> Cell | None:
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
        elif flag == CELL_FLAG_CELL_STYLE:
            decoded = decoded._replace(style=read_uint32(storage, offset))
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
