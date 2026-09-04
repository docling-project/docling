# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from pydantic import ValidationError

from docling.backend.ebcdic_backend import EbcdicDecodeError, EbcdicDocumentBackend
from docling.datamodel.backend_options import (
    EbcdicBackendOptions,
    EbcdicField,
    EbcdicFieldType,
    EbcdicLayout,
    EbcdicRecordLayout,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.document_converter import DocumentConverter, EbcdicFormatOption
from docling.exceptions import DocumentLoadError

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_export

GENERATE = GEN_TEST_DATA
ENCODING = "cp037"
SOURCES = Path("./tests/data/ebcdic/sources/")


def _text(value: str, size: int) -> bytes:
    return value.ljust(size).encode(ENCODING)


def _packed(value: int, size: int) -> bytes:
    """Encode a COMP-3 value: two digits per byte, sign in the last nibble."""
    return bytes.fromhex(f"{abs(value):0{size * 2 - 1}d}" + ("d" if value < 0 else "c"))


def _zoned(value: int, size: int) -> bytes:
    """Encode a signed display numeric: the sign lands in the last zone nibble."""
    digits = f"{abs(value):0{size}d}".encode(ENCODING)
    return digits[:-1] + bytes([(digits[-1] & 0x0F) | (0xD0 if value < 0 else 0xC0)])


def _convert(data: bytes, options: EbcdicBackendOptions) -> ConversionResult:
    converter = DocumentConverter(
        format_options={InputFormat.EBCDIC: EbcdicFormatOption(backend_options=options)}
    )
    return converter.convert(DocumentStream(name="data.ebc", stream=BytesIO(data)))


def _backend(data: bytes, options: EbcdicBackendOptions) -> EbcdicDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=BytesIO(data),
        format=InputFormat.EBCDIC,
        filename="data.ebc",
        backend=EbcdicDocumentBackend,
        backend_options=options,
    )
    return EbcdicDocumentBackend(in_doc, BytesIO(data), options)


@pytest.fixture
def employee_layout() -> EbcdicLayout:
    return EbcdicLayout(
        description="employee master",
        records=[
            EbcdicRecordLayout(
                name="employee",
                fields=[
                    EbcdicField(name="name", size=10),
                    EbcdicField(name="filler", size=2, type=EbcdicFieldType.SKIP),
                    EbcdicField(
                        name="wages",
                        size=5,
                        type=EbcdicFieldType.PACKED_DECIMAL,
                        scale=2,
                    ),
                    EbcdicField(
                        name="balance",
                        size=6,
                        type=EbcdicFieldType.ZONED_DECIMAL,
                        scale=2,
                    ),
                    EbcdicField(name="id", size=4, type=EbcdicFieldType.INTEGER),
                ],
            )
        ],
    )


def _employee(name: str, wages: int, balance: int, ident: int) -> bytes:
    return (
        _text(name, 10)
        + b"\x00\x00"
        + _packed(wages, 5)
        + _zoned(balance, 6)
        + ident.to_bytes(4, "big", signed=True)
    )


def _converter(layout_file: Path) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.EBCDIC: EbcdicFormatOption(
                backend_options=EbcdicBackendOptions(layout_file=layout_file)
            )
        }
    )


@pytest.mark.parametrize("source", sorted(SOURCES.glob("*.ebc")), ids=lambda p: p.stem)
def test_e2e_ebcdic_conversions(source: Path):
    """Convert the mainframe samples and compare against the groundtruth."""
    gt_path = source.parent.parent / "groundtruth" / source.name

    result = _converter(source.with_suffix(".layout.json")).convert(source)
    doc: DoclingDocument = result.document

    # Markdown alone is the groundtruth here: these documents are nothing but
    # tables of decoded field values, so the serialized DoclingDocument would
    # add megabytes of cell scaffolding without covering anything the tables do
    # not already show. Structure is asserted separately, below.
    pred_md = doc.export_to_markdown(escape_html=False, compact_tables=True)
    assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
        "export to md"
    )


def test_ola013k_splits_records_across_schemas():
    """The multi-schema sample fans out into one table per record type."""
    source = SOURCES / "ola013k.ebc"

    result = _converter(source.with_suffix(".layout.json")).convert(source)

    doc = result.document
    assert len(doc.tables) == 4
    # Each schema carries the same five records, but its own set of columns.
    assert [table.data.num_rows for table in doc.tables] == [6, 6, 6, 6]
    assert [table.data.num_cols for table in doc.tables] == [209, 307, 360, 489]


def test_single_schema_decodes_cobol_field_types(employee_layout):
    data = _employee("Ada", 123456, -98765, 7) + _employee("Grace", 900000, 12345, -2)

    result = _convert(data, EbcdicBackendOptions(layout=employee_layout))

    tables = result.document.tables
    assert len(tables) == 1
    rows = tables[0].data.grid
    # Skip fields are consumed but never emitted as a column.
    assert [cell.text for cell in rows[0]] == ["name", "wages", "balance", "id"]
    assert [cell.text for cell in rows[1]] == ["Ada", "1234.56", "-987.65", "7"]
    assert [cell.text for cell in rows[2]] == ["Grace", "9000.00", "123.45", "-2"]


def test_max_records_stops_early(employee_layout):
    data = _employee("Ada", 1, 1, 1) * 5

    result = _convert(data, EbcdicBackendOptions(layout=employee_layout, max_records=2))

    assert result.document.tables[0].data.num_rows == 3  # header row plus 2 records


def test_multi_schema_variable_length_records(tmp_path: Path):
    layout = {
        "description": "orders",
        "header_size": 4,
        "footer_size": 3,
        "record_length_field": {
            "name": "length",
            "size": 2,
            "type": "unsigned_integer",
        },
        "record_type_field": {"name": "kind", "size": 1},
        "records": [
            {
                "name": "customer",
                "selector": "H",
                "fields": [{"name": "account", "size": 8}],
            },
            {
                "name": "line",
                "selector": "L",
                "fields": [
                    {"name": "sku", "size": 4},
                    {"name": "quantity", "size": 3, "type": "packed_decimal"},
                ],
            },
        ],
    }
    layout_file = tmp_path / "orders.json"
    layout_file.write_text(json.dumps(layout))

    def record(kind: str, body: bytes) -> bytes:
        return (3 + len(body)).to_bytes(2, "big") + kind.encode(ENCODING) + body

    data = (
        b"HDR\x00"
        + record("H", _text("ACME", 8))
        + record("L", _text("A100", 4) + _packed(12, 3))
        + record("L", _text("B200", 4) + _packed(-7, 3))
        + record("H", _text("GLOBEX", 8))
        + b"EOF"
    )

    result = _convert(data, EbcdicBackendOptions(layout_file=layout_file))

    accounts, lines = result.document.tables
    assert [cell.text for cell in accounts.data.grid[1]] == ["ACME"]
    assert [cell.text for cell in accounts.data.grid[2]] == ["GLOBEX"]
    assert [row[1].text for row in lines.data.grid[1:]] == ["12", "-7"]


def test_layout_is_required():
    with pytest.raises(DocumentLoadError, match="needs a layout"):
        _backend(b"\x00" * 8, EbcdicBackendOptions())


def test_truncated_record_is_reported(employee_layout):
    truncated = _employee("Ada", 1, 1, 1)[:-1]

    with pytest.raises(EbcdicDecodeError, match="Input ends inside 'employee'"):
        _backend(truncated, EbcdicBackendOptions(layout=employee_layout)).convert()


def test_unknown_record_type_is_reported():
    layout = EbcdicLayout(
        record_type_field=EbcdicField(name="kind", size=1),
        records=[
            EbcdicRecordLayout(
                name="a", selector="A", fields=[EbcdicField(name="value", size=2)]
            ),
            EbcdicRecordLayout(
                name="b", selector="B", fields=[EbcdicField(name="value", size=2)]
            ),
        ],
    )

    with pytest.raises(EbcdicDecodeError, match="No record layout matches"):
        _backend(
            _text("Z", 1) + _text("xy", 2), EbcdicBackendOptions(layout=layout)
        ).convert()


def test_multi_schema_layout_requires_a_record_type_field():
    fields = [EbcdicField(name="value", size=2)]

    with pytest.raises(ValidationError, match="record_type_field is required"):
        EbcdicLayout(
            records=[
                EbcdicRecordLayout(name="a", selector="A", fields=fields),
                EbcdicRecordLayout(name="b", selector="B", fields=fields),
            ]
        )


def test_record_names_must_be_unique():
    """Rows are bucketed by record name, so duplicates would merge two schemas.

    `name` defaults to "record", so a two-schema layout that sets only the
    selectors used to convert cleanly and hand every table both schemas' rows.
    """
    fields = [EbcdicField(name="value", size=2)]

    with pytest.raises(ValidationError, match="record names must be unique"):
        EbcdicLayout(
            record_type_field=EbcdicField(name="kind", size=1),
            records=[
                EbcdicRecordLayout(selector="A", fields=fields),
                EbcdicRecordLayout(selector="B", fields=fields),
            ],
        )


def test_record_names_must_be_unique_from_a_layout_file(tmp_path: Path):
    """A copybook usually arrives as JSON, so the check has to hold on that path."""
    layout_file = tmp_path / "layout.json"
    layout_file.write_text(
        json.dumps(
            {
                "record_type_field": {"name": "kind", "size": 1},
                "records": [
                    {"selector": "A", "fields": [{"name": "value", "size": 2}]},
                    {"selector": "B", "fields": [{"name": "value", "size": 2}]},
                ],
            }
        )
    )

    with pytest.raises(DocumentLoadError, match="Could not read the EBCDIC layout"):
        _backend(
            _text("A", 1) + _text("xy", 2),
            EbcdicBackendOptions(layout_file=layout_file),
        )


def test_a_single_record_keeps_the_default_name():
    """The default name is only a problem when it collides; one schema is fine."""
    layout = EbcdicLayout(
        records=[EbcdicRecordLayout(fields=[EbcdicField(name="v", size=2)])]
    )

    assert layout.records[0].name == "record"


def test_layout_sources_are_mutually_exclusive(employee_layout, tmp_path: Path):
    with pytest.raises(ValidationError, match="not both"):
        EbcdicBackendOptions(
            layout=employee_layout, layout_file=tmp_path / "orders.json"
        )
