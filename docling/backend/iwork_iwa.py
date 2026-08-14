"""Reader for Apple's IWA (iWork Archive) container format.

Pages, Numbers and Keynote from 2013 onwards store their documents as
``Index/*.iwa`` members inside the package. Apple has never published the
schemas, but the container itself is straightforward and stable:

1. An ``.iwa`` file is a sequence of chunks. Each chunk has a four-byte header —
   one compression tag (``0x00`` for Snappy) plus a three-byte little-endian
   payload length — followed by that many bytes of **raw** Snappy: no stream
   identifier, no CRC-32C, so a framed Snappy decoder cannot read it.
2. Concatenating the decompressed chunks yields a stream of archives. Each is a
   varint length, a ``TSP.ArchiveInfo`` message carrying an object identifier and
   one or more ``MessageInfo`` descriptors, then the payload bytes each
   descriptor claims.

That is enough to walk the document's object graph without any ``.proto``
definitions: the fields this backend needs are read positionally by
:func:`read_fields`. Only the message *type numbers* are format knowledge, and
those live in the Pages backend rather than here.
"""

import logging
from typing import Iterator, NamedTuple

from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_SNAPPY_TAG = 0x00
_HEADER_LEN = 4

# Protobuf wire types.
_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH_DELIMITED = 2
_WIRE_32BIT = 5

# Snappy element tags, taken from the low two bits of each tag byte.
_TAG_LITERAL = 0
_TAG_COPY_1B = 1
_TAG_COPY_2B = 2

# A decompressed archive stream must stay under this to bound memory use for a
# hostile container. Real Pages documents decompress to a few MB at most.
_MAX_STREAM_BYTES = 256 * 1024 * 1024

FieldMap = dict[int, list[int | bytes]]


class IWAObject(NamedTuple):
    """One archived object: its identifier, message type and raw payload."""

    identifier: int
    message_type: int
    payload: bytes


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Read a base-128 varint, returning the value and the new position."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise DocumentLoadError("Truncated varint in IWA stream.")
        if shift > 63:
            raise DocumentLoadError("Overlong varint in IWA stream.")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def read_fields(buf: bytes) -> FieldMap:
    """Decode a protobuf message into ``{field_number: [values]}``.

    Values are ``int`` for varint and fixed-width fields and ``bytes`` for
    length-delimited ones, which the caller re-reads as a nested message, a
    UTF-8 string or a packed value as the field requires. Groups (wire types 3
    and 4) are obsolete and unused by iWork, so they are rejected.
    """
    fields: FieldMap = {}
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_no, wire_type = key >> 3, key & 0x07
        value: int | bytes
        if wire_type == _WIRE_VARINT:
            value, pos = read_varint(buf, pos)
        elif wire_type == _WIRE_LENGTH_DELIMITED:
            length, pos = read_varint(buf, pos)
            if pos + length > len(buf):
                raise DocumentLoadError("Truncated length-delimited IWA field.")
            value = buf[pos : pos + length]
            pos += length
        elif wire_type == _WIRE_64BIT:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire_type == _WIRE_32BIT:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise DocumentLoadError(
                f"Unsupported protobuf wire type {wire_type} in IWA stream."
            )
        fields.setdefault(field_no, []).append(value)
    return fields


def read_reference(buf: bytes) -> int | None:
    """Read a ``TSP.Reference``, whose only field is the target object id."""
    target = read_fields(buf).get(1, [None])[0]
    return target if isinstance(target, int) else None


def decompress_snappy_block(block: bytes) -> bytes:
    """Decompress one raw Snappy block.

    Implemented here rather than taken from a binding because iWork needs only
    the decompressor, and that half of Snappy is small: a varint length preamble
    followed by literal and copy elements. Keeping it in Python means the format
    does not depend on a compiled wheel staying maintained for future Python
    releases, and Pages documents are small enough that the speed difference is
    immaterial.
    """
    expected, pos = read_varint(block, 0)
    out = bytearray()
    size = len(block)

    while pos < size:
        tag = block[pos]
        pos += 1
        kind = tag & 0x03

        if kind == _TAG_LITERAL:
            length = tag >> 2
            if length >= 60:
                # 60..63 mean the real length occupies the next (length - 59) bytes.
                extra = length - 59
                if pos + extra > size:
                    raise DocumentLoadError("Truncated Snappy literal length.")
                length = int.from_bytes(block[pos : pos + extra], "little")
                pos += extra
            length += 1
            if pos + length > size:
                raise DocumentLoadError("Truncated Snappy literal.")
            out += block[pos : pos + length]
            pos += length
            continue

        if kind == _TAG_COPY_1B:
            length = 4 + ((tag >> 2) & 0x07)
            if pos >= size:
                raise DocumentLoadError("Truncated Snappy copy offset.")
            offset = ((tag >> 5) << 8) | block[pos]
            pos += 1
        else:
            width = 2 if kind == _TAG_COPY_2B else 4
            length = (tag >> 2) + 1
            if pos + width > size:
                raise DocumentLoadError("Truncated Snappy copy offset.")
            offset = int.from_bytes(block[pos : pos + width], "little")
            pos += width

        if offset == 0 or offset > len(out):
            raise DocumentLoadError("Snappy copy offset outside the output window.")

        start = len(out) - offset
        if offset >= length:
            out += out[start : start + length]
        else:
            # Overlapping copy: the source keeps advancing as the output grows,
            # which is how Snappy encodes repeated runs, so copy byte by byte.
            for index in range(length):
                out.append(out[start + index])

    if len(out) != expected:
        raise DocumentLoadError(
            f"Snappy block decoded to {len(out)} bytes, expected {expected}."
        )
    return bytes(out)


def decompress(data: bytes) -> bytes:
    """Concatenate the decompressed chunks of one ``.iwa`` member."""
    out = bytearray()
    pos = 0
    while pos < len(data):
        if pos + _HEADER_LEN > len(data):
            raise DocumentLoadError("Truncated IWA chunk header.")
        tag = data[pos]
        length = int.from_bytes(data[pos + 1 : pos + _HEADER_LEN], "little")
        pos += _HEADER_LEN
        if tag != _SNAPPY_TAG:
            raise DocumentLoadError(
                f"Unsupported IWA chunk compression tag 0x{tag:02x}."
            )
        block = data[pos : pos + length]
        if len(block) != length:
            raise DocumentLoadError("Truncated IWA chunk payload.")
        pos += length
        # Raw Snappy: the chunk carries no stream framing or checksum.
        out += decompress_snappy_block(block)
        if len(out) > _MAX_STREAM_BYTES:
            raise DocumentLoadError(
                f"IWA stream expands beyond {_MAX_STREAM_BYTES} bytes."
            )
    return bytes(out)


def iter_objects(data: bytes) -> Iterator[IWAObject]:
    """Yield every archived object in one decompressed ``.iwa`` stream."""
    stream = decompress(data)
    pos = 0
    while pos < len(stream):
        info_len, pos = read_varint(stream, pos)
        if pos + info_len > len(stream):
            raise DocumentLoadError("Truncated TSP.ArchiveInfo in IWA stream.")
        info = read_fields(stream[pos : pos + info_len])
        pos += info_len

        identifier = info.get(1, [0])[0]
        if not isinstance(identifier, int):
            raise DocumentLoadError("Malformed object identifier in IWA stream.")

        for message_info in info.get(2, []):
            if not isinstance(message_info, bytes):
                continue
            message = read_fields(message_info)
            message_type = message.get(1, [0])[0]
            payload_len = message.get(3, [0])[0]
            if not isinstance(message_type, int) or not isinstance(payload_len, int):
                raise DocumentLoadError("Malformed MessageInfo in IWA stream.")
            if pos + payload_len > len(stream):
                raise DocumentLoadError("Truncated object payload in IWA stream.")
            yield IWAObject(identifier, message_type, stream[pos : pos + payload_len])
            pos += payload_len
