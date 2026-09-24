# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import warnings
from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
pytestmark = pytest.mark.cross_platform


def get_csv_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/csv/sources/")

    # List all CSV files in the directory and its subdirectories
    return sorted(directory.rglob("*.csv"))


def get_csv_path(name: str):
    # Return the matching CSV file path
    return Path(f"./tests/data/csv/sources/{name}.csv")


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.CSV])

    return converter


def test_e2e_valid_csv_conversions():
    valid_csv_paths = get_csv_paths()
    converter = get_converter()

    for csv_path in valid_csv_paths:
        print(f"converting {csv_path}")

        gt_path = csv_path.parent.parent / "groundtruth" / csv_path.name
        if csv_path.stem in (
            "csv-too-few-columns",
            "csv-too-many-columns",
            "csv-inconsistent-header",
        ):
            with pytest.warns(UserWarning, match="Inconsistent column lengths"):
                conv_result: ConversionResult = converter.convert(csv_path)
        else:
            conv_result: ConversionResult = converter.convert(csv_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(
            pred_doc=doc,
            gtfile=str(gt_path) + ".json",
            generate=GENERATE,
        ), "export to json"


def test_e2e_invalid_csv_conversions():
    csv_too_few_columns = get_csv_path("csv-too-few-columns")
    csv_too_many_columns = get_csv_path("csv-too-many-columns")
    csv_inconsistent_header = get_csv_path("csv-inconsistent-header")
    converter = get_converter()

    print(f"converting {csv_too_few_columns}")
    with pytest.warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_too_few_columns)

    print(f"converting {csv_too_many_columns}")
    with pytest.warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_too_many_columns)

    print(f"converting {csv_inconsistent_header}")
    with pytest.warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_inconsistent_header)


def test_quoted_newline_in_first_field():
    """A quoted field spanning several lines must not break delimiter sniffing.

    Reading a single line split the field mid-quote, so the sniffer saw an
    unterminated quote and the conversion failed outright.
    """
    csv_bytes = b'"line one\nstill line one";b;c\n1;2;3\n'
    conv_result = get_converter().convert(
        DocumentStream(name="quoted.csv", stream=BytesIO(csv_bytes)),
        raises_on_error=True,
    )
    table = conv_result.document.tables[0]
    assert table.data.num_cols == 3
    assert table.data.table_cells[0].text == "line one\nstill line one"


@pytest.mark.parametrize(
    "csv_bytes",
    [
        b'id,quote\n1,"He said ""no"" twice"\n',
        b'id;quote\n1;"He said ""no"" twice"\n',
        b'"id","quote"\n"1","He said ""no"" twice"\n',
        # Sniffed from the fallback sample; the doubled quote is past its end.
        b'"id\nnumber";quote\n' + b"1;x\n" * 1100 + b'2;"He said ""no"" twice"\n',
    ],
    ids=["comma", "semicolon", "quoted-header", "fallback-sample"],
)
def test_doubled_quote_is_an_escaped_quote(csv_bytes):
    """A doubled quote inside a quoted field is one literal quote (RFC 4180).

    The sniffer only sets doublequote when its sample already contains one,
    and the sample is usually just the header, so `""` in the data rows was
    read as a closing quote and the rest of the field kept stray quotes.
    """
    conv_result = get_converter().convert(
        DocumentStream(name="quotes.csv", stream=BytesIO(csv_bytes)),
        raises_on_error=True,
    )
    table = conv_result.document.tables[0]
    assert table.data.num_cols == 2
    assert table.data.table_cells[-1].text == 'He said "no" twice'


def test_backslash_escaped_quote_still_converts():
    """A backslash escape is not RFC 4180, but such files still have to convert.

    MySQL's SELECT ... INTO OUTFILE writes `\\"` by default. Reading doubled
    quotes as escapes makes those rows malformed, so the row is read unescaped
    instead of failing the whole document.
    """
    csv_bytes = b'id,quote\n1,"He said \\"no\\" twice"\n'
    conv_result = get_converter().convert(
        DocumentStream(name="backslash.csv", stream=BytesIO(csv_bytes)),
        raises_on_error=True,
    )
    table = conv_result.document.tables[0]
    assert table.data.num_cols == 2
    assert table.data.table_cells[-1].text.startswith("He said ")


def test_empty_csv():
    """Regression test: converting an empty CSV file should not raise an IndexError."""
    conv_result = get_converter().convert(
        DocumentStream(name="empty.csv", stream=BytesIO(b"")),
        raises_on_error=True,
    )
    doc = conv_result.document
    assert doc is not None
    # The empty CSV should result in an empty document (no tables and no texts).
    assert len(getattr(doc, "tables", [])) == 0
    assert len(getattr(doc, "texts", [])) == 0


def test_utf8_bom_is_not_part_of_the_first_cell(tmp_path):
    """A leading UTF-8 BOM must not survive into the first header cell.

    Excel and Google Sheets write a BOM when exporting "CSV UTF-8". Decoding
    with plain utf-8 kept it, so the first header came back as U+FEFF followed
    by "Name", and matching on that header silently missed the column. Both the
    stream and the file path are covered, since each decodes separately.
    """
    csv_bytes = "\ufeffName,Age\nAlice,30\n".encode()
    converter = get_converter()

    stream_doc = converter.convert(
        DocumentStream(name="bom.csv", stream=BytesIO(csv_bytes)),
        raises_on_error=True,
    ).document

    csv_file = tmp_path / "bom.csv"
    csv_file.write_bytes(csv_bytes)
    file_doc = converter.convert(csv_file, raises_on_error=True).document

    for doc in (stream_doc, file_doc):
        cells = doc.tables[0].data.table_cells
        assert cells[0].text == "Name"
        assert cells[1].text == "Age"


def test_malformed_quoted_csv_is_a_load_error():
    """An unclosed quote used to escape convert() as csv.Error after sniffing."""
    conv_result = get_converter().convert(
        DocumentStream(name="bad.csv", stream=BytesIO(b'"unclosed quote,1,2\n')),
        raises_on_error=False,
    )
    assert conv_result.status == ConversionStatus.FAILURE


def test_blank_lines_do_not_become_empty_rows():
    """A blank line is not a record, so it must not add a row of empty cells.

    `csv.reader` yields an empty list for a blank line. Those were kept, so a file
    ending in a newline pair -- which plenty of exporters write -- gained a trailing
    empty row, and a blank line between records gained one in the middle. The empty
    row also made the row lengths non-uniform, raising a spurious "Inconsistent
    column lengths" warning on a perfectly uniform file.
    """
    converter = get_converter()
    cases = {
        "trailing": b"a,b\n1,2\n\n",
        "two trailing": b"a,b\n1,2\n\n\n",
        "trailing crlf": b"a,b\r\n1,2\r\n\r\n",
        "leading": b"\na,b\n1,2\n",
    }

    for name, csv_bytes in cases.items():
        with warnings.catch_warnings():
            # -- a uniform file must not warn about inconsistent column lengths --
            warnings.simplefilter("error")
            doc = converter.convert(
                DocumentStream(name=f"{name}.csv", stream=BytesIO(csv_bytes)),
                raises_on_error=True,
            ).document

        table_data = doc.tables[0].data
        assert table_data.num_rows == 2, name
        assert [cell.text for cell in table_data.table_cells] == ["a", "b", "1", "2"], (
            name
        )


def test_blank_line_between_records_is_dropped():
    """The records on either side of a blank line stay adjacent."""
    doc = (
        get_converter()
        .convert(
            DocumentStream(name="gap.csv", stream=BytesIO(b"a,b\n1,2\n\n3,4\n")),
            raises_on_error=True,
        )
        .document
    )

    table_data = doc.tables[0].data
    assert table_data.num_rows == 3
    assert [cell.text for cell in table_data.table_cells] == [
        "a",
        "b",
        "1",
        "2",
        "3",
        "4",
    ]


def test_row_of_empty_fields_is_kept():
    """A line of delimiters is a real record of empty fields, unlike a blank line."""
    doc = (
        get_converter()
        .convert(
            DocumentStream(name="empties.csv", stream=BytesIO(b"a,b\n,\n")),
            raises_on_error=True,
        )
        .document
    )

    table_data = doc.tables[0].data
    assert table_data.num_rows == 2
    assert [cell.text for cell in table_data.table_cells] == ["a", "b", "", ""]


def test_file_of_only_blank_lines_is_empty():
    """Dropping every row must leave an empty document, not an empty table."""
    doc = (
        get_converter()
        .convert(
            DocumentStream(name="blank.csv", stream=BytesIO(b"\n\n\n")),
            raises_on_error=True,
        )
        .document
    )

    assert len(doc.tables) == 0
