# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend for mainframe EBCDIC data files.

EBCDIC files carry no self-describing structure: the byte stream is a sequence
of fixed-width records whose fields are only meaningful together with the COBOL
copybook that produced them. The copybook is therefore supplied as an
`EbcdicLayout` through `EbcdicBackendOptions`, and this backend turns each
record schema into one table of the resulting `DoclingDocument`.

Decoding is pure Python: character data goes through the standard library
EBCDIC codecs, and the COBOL numeric usages are unpacked from their nibbles.
"""

import codecs
import logging
import re
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Callable, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    TableCell,
    TableData,
)
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import (
    EbcdicBackendOptions,
    EbcdicField,
    EbcdicFieldType,
    EbcdicLayout,
    EbcdicRecordLayout,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_MIME_TYPE = "application/x-ebcdic"

# Unicode general category Cc, i.e. the C0 and C1 control ranges. EBCDIC fields
# are padded and delimited with bytes that decode into these.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Sign nibbles of packed and zoned decimals: 0xb and 0xd are negative, every
# other value (0xa, 0xc, 0xe, 0xf and unsigned digits) is positive.
_NEGATIVE_SIGNS = frozenset({0xB, 0xD})

_DecodedValue = Union[str, int, Decimal]


class EbcdicDecodeError(DocumentLoadError):
    """A field could not be decoded with the configured layout."""


class _FieldDecoder:
    """Decode single EBCDIC fields into Python values."""

    def __init__(self, encoding: str, strip_control_characters: bool) -> None:
        try:
            self._decode_text = codecs.getdecoder(encoding)
        except LookupError as exc:
            raise DocumentLoadError(f"Unknown EBCDIC codec {encoding!r}.") from exc
        self._strip_control_characters = strip_control_characters
        self._decoders: dict[EbcdicFieldType, Callable[[bytes], _DecodedValue]] = {
            EbcdicFieldType.STRING: self._string,
            EbcdicFieldType.INTEGER: lambda data: self._binary(data, signed=True),
            EbcdicFieldType.UNSIGNED_INTEGER: lambda data: self._binary(
                data, signed=False
            ),
            EbcdicFieldType.PACKED_DECIMAL: self._packed_decimal,
            EbcdicFieldType.ZONED_DECIMAL: self._zoned_decimal,
        }

    def decode(self, data: bytes, field: EbcdicField) -> _DecodedValue:
        """Decode the bytes of one field as described by its layout."""
        try:
            value = self._decoders[field.type](data)
        except (ArithmeticError, LookupError, UnicodeError, ValueError) as exc:
            raise EbcdicDecodeError(
                f"Cannot decode field {field.name!r} of type {field.type.value} "
                f"from {data.hex()!r}."
            ) from exc
        if isinstance(value, int) and field.scale:
            return Decimal(value).scaleb(-field.scale)
        return value

    def _string(self, data: bytes) -> str:
        text, _ = self._decode_text(data)
        if self._strip_control_characters:
            text = _CONTROL_CHARACTERS.sub("", text)
        return text.strip()

    @staticmethod
    def _binary(data: bytes, signed: bool) -> int:
        return int.from_bytes(data, byteorder="big", signed=signed)

    @staticmethod
    def _packed_decimal(data: bytes) -> int:
        """Unpack COMP-3: two digits per byte, sign in the trailing nibble."""
        nibbles = data.hex()
        digits = int(nibbles[:-1] or "0")
        return -digits if int(nibbles[-1], 16) in _NEGATIVE_SIGNS else digits

    @staticmethod
    def _zoned_decimal(data: bytes) -> int:
        """Unpack a signed display numeric: one digit per low nibble."""
        nibbles = [byte & 0x0F for byte in data]
        if any(nibble > 9 for nibble in nibbles):
            raise ValueError(f"{data.hex()} holds a non-decimal digit")
        digits = int("".join(str(nibble) for nibble in nibbles))
        return -digits if data[-1] >> 4 in _NEGATIVE_SIGNS else digits


class _RecordParser:
    """Split an EBCDIC byte stream into decoded records."""

    def __init__(self, layout: EbcdicLayout, decoder: _FieldDecoder) -> None:
        self._layout = layout
        self._decoder = decoder

    def parse(
        self, data: bytes, max_records: Union[int, None]
    ) -> dict[str, list[list[str]]]:
        """Return the decoded rows of every record schema, keyed by schema name."""
        rows: dict[str, list[list[str]]] = {
            record.name: [] for record in self._layout.records
        }
        end = len(data) - self._layout.footer_size
        offset = self._layout.header_size
        count = 0

        while offset < end and (max_records is None or count < max_records):
            record, size, offset = self._read_prefix(data, offset, end)
            body = self._take(data, offset, size, end, record.name)
            rows[record.name].append(self._decode_record(record, body))
            offset += size
            count += 1

        _log.debug("Decoded %d EBCDIC records", count)
        return rows

    def _read_prefix(
        self, data: bytes, offset: int, end: int
    ) -> tuple[EbcdicRecordLayout, int, int]:
        """Consume the record prefix and resolve the schema and body size."""
        layout = self._layout
        length: Union[int, None] = None
        record_type: Union[str, None] = None

        if (field := layout.record_length_field) is not None:
            chunk = self._take(data, offset, field.size, end, field.name)
            length = int(self._decoder.decode(chunk, field))
            offset += field.size
        if (field := layout.record_type_field) is not None:
            chunk = self._take(data, offset, field.size, end, field.name)
            record_type = str(self._decoder.decode(chunk, field))
            offset += field.size

        record = layout.select(record_type)
        if record is None:
            raise EbcdicDecodeError(
                f"No record layout matches record type {record_type!r}."
            )

        size = record.size if length is None else length - layout.prefix_size
        if size < 0:
            raise EbcdicDecodeError(
                f"Record length {length} is shorter than the "
                f"{layout.prefix_size}-byte record prefix."
            )
        return record, size, offset

    @staticmethod
    def _take(data: bytes, offset: int, size: int, end: int, name: str) -> bytes:
        if offset + size > end:
            raise EbcdicDecodeError(
                f"Input ends inside {name!r}: {end - offset} of {size} bytes left."
            )
        return data[offset : offset + size]

    def _decode_record(self, record: EbcdicRecordLayout, body: bytes) -> list[str]:
        values: list[str] = []
        offset = 0
        for field in record.fields:
            chunk = body[offset : offset + field.size]
            offset += field.size
            if field.type is not EbcdicFieldType.SKIP:
                values.append(str(self._decoder.decode(chunk, field)))
        return values


class EbcdicDocumentBackend(DeclarativeDocumentBackend):
    """Declarative backend converting EBCDIC data files to `DoclingDocument`.

    The layout is mandatory and comes from `EbcdicBackendOptions`, either
    inline as an `EbcdicLayout` or as a JSON file. Each record schema becomes
    one table whose header row holds the field names; `skip` fields are
    consumed but never emitted.
    """

    options: EbcdicBackendOptions
    content: bytes
    layout: EbcdicLayout

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Union[EbcdicBackendOptions, None] = None,
    ) -> None:
        if options is None:
            options = EbcdicBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        self.layout = self._resolve_layout()
        try:
            # Read from the argument rather than self.path_or_stream, which
            # unload() clears to None.
            self.content = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else path_or_stream.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise DocumentLoadError(
                "Could not initialize the EBCDIC backend for file with hash "
                f"{self.document_hash}."
            ) from exc

    def _resolve_layout(self) -> EbcdicLayout:
        if self.options.layout is not None:
            return self.options.layout
        if self.options.layout_file is None:
            raise DocumentLoadError(
                "The EBCDIC backend needs a layout: set either "
                "EbcdicBackendOptions.layout or EbcdicBackendOptions.layout_file."
            )
        try:
            return EbcdicLayout.model_validate_json(
                self.options.layout_file.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise DocumentLoadError(
                f"Could not read the EBCDIC layout {self.options.layout_file}."
            ) from exc

    @override
    def is_valid(self) -> bool:
        return bool(self.content)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.EBCDIC}

    @override
    def convert(self) -> DoclingDocument:
        """Parse the EBCDIC data into one table per record schema."""
        if not self.is_valid():
            raise DocumentLoadError(
                f"Cannot convert doc with {self.document_hash} because the "
                "backend failed to init."
            )

        origin = DocumentOrigin(
            filename=self.file.name or "file.ebc",
            mimetype=_MIME_TYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if self.layout.description:
            doc.add_text(label=DocItemLabel.TEXT, text=self.layout.description)

        decoder = _FieldDecoder(
            self.options.encoding, self.options.strip_control_characters
        )
        rows = _RecordParser(self.layout, decoder).parse(
            self.content, self.options.max_records
        )

        for record in self.layout.records:
            if not rows[record.name]:
                continue
            if len(self.layout.records) > 1:
                doc.add_heading(text=record.name)
            doc.add_table(data=_build_table(record, rows[record.name]))

        return doc


def _build_table(record: EbcdicRecordLayout, rows: list[list[str]]) -> TableData:
    """Lay the decoded records out as a table with a field-name header row."""
    header = [
        field.name for field in record.fields if field.type is not EbcdicFieldType.SKIP
    ]
    table = TableData(num_rows=len(rows) + 1, num_cols=len(header), table_cells=[])
    for row_idx, row in enumerate([header, *rows]):
        for col_idx, value in enumerate(row):
            table.table_cells.append(
                TableCell(
                    text=value,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=row_idx == 0,
                )
            )
    return table
