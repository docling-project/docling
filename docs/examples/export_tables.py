# %% [markdown]
# Extract tables from a PDF and export them as CSV and HTML.
#
# What this example does
# - Converts a PDF and iterates detected tables.
# - Prints each table as Markdown to stdout, and saves CSV/HTML to `scratch/`.
#
# Prerequisites
# - Install Docling and `pandas`.
#
# How to run
# - From the repo root: `python docs/examples/export_tables.py`.
# - Outputs are written to `scratch/`.
#
# Input document
# - Defaults to `tests/data/pdf/2206.01062.pdf`. Change `input_doc_path` as needed.
#
# Notes
# - `table.export_to_dataframe()` returns a pandas DataFrame for convenient export/processing.
# - Printing via `DataFrame.to_markdown()` may require the optional `tabulate` package
#   (`pip install tabulate`). If unavailable, skip the print or use `to_csv()`.

# %%

import logging
import time
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    doc_converter = DocumentConverter()

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as CSV
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)

        # Save the table as HTML
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

    end_time = time.time() - start_time

    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")


if __name__ == "__main__":
    main()
