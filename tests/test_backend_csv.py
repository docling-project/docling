# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import DocumentStream, InputFormat
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
