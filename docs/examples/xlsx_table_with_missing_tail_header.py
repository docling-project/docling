from pathlib import Path

import pandas as pd

from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.datamodel.backend_options import MsExcelBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter, ExcelFormatOption


def get_excel_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/xlsx/")

    # List all Excel files in the directory and its subdirectories
    excel_files = sorted(directory.rglob("*.xlsx")) + sorted(directory.rglob("*.xlsm"))
    return excel_files


if __name__ == "__main__":
    # Get a test Excel file
    path = next(
        (
            item
            for item in get_excel_paths()
            if item.stem == "xlsx_06_table_with_missing_headers_tail"
        ),
        None,
    )

    if path is None:
        print("xlsx_06 not found!")

    # Create converter with merge_headless_columns_in_pages=True
    options = MsExcelBackendOptions(
        merge_headless_columns_in_pages=True,
        merge_headless_columns_str="Missing Header",
    )
    format_options = {InputFormat.XLSX: ExcelFormatOption(backend_options=options)}
    converter = DocumentConverter(
        allowed_formats=[InputFormat.XLSX], format_options=format_options
    )

    conv_result: ConversionResult = converter.convert(path)
    doc: DoclingDocument = conv_result.document

    # With merge_headless_columns_in_pages=True, the two tables should be merged into one
    tables = list(doc.tables)

    assert len(tables) == 1, f"Should have 1 table, got {len(tables)}"

    table = tables[0]
    table_df: pd.DataFrame = table.export_to_dataframe(doc=conv_result.document)
    print(table_df.to_markdown())
    element_csv_file = "./example_output.csv"
    table_df.to_csv(element_csv_file)

    # Verify the table dimensions
    assert table.data.num_rows == 4, (
        f"Table should have 4 rows, got {table.data.num_rows}"
    )
    assert table.data.num_cols == 6, (
        f"Table should have 6 rows, got {table.data.num_cols}"
    )
