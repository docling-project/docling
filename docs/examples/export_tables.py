# %% [markdown]
# Extract tables from a PDF and export them as CSV and HTML using docling-serve.
#
# What this example does
# - Converts a PDF using docling-serve API and iterates detected tables.
# - Prints each table as Markdown to stdout, and saves CSV/HTML to `scratch/`.
#
# Prerequisites
# - Install Docling service client and `pandas`.
# - Set API_KEY and SERVICE_URL for your docling-serve instance.
#
# How to run
# - From the repo root: `python docs/examples/tableexport.py`.
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
import os

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

_log = logging.getLogger(__name__)

# Configure your SERVICE_URL, and your API_KEY should your service need authentication.
SERVICE_URL = os.environ["DOCLING_SERVICE_URL"]
API_KEY = os.environ.get("DOCLING_SERVICE_API_KEY")

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../docling/tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    start_time = time.time()

    # Use DoclingServiceClient instead of DocumentConverter
    with DoclingServiceClient(url=SERVICE_URL, api_key=API_KEY) as client:
        conv_res = client.convert(
            source=input_doc_path,
            options=ConvertDocumentsOptions(
                do_table_structure=True,  # Ensure table structure detection is enabled
                to_formats=[OutputFormat.JSON],  # JSON format needed for table access
            ),
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        doc_filename = conv_res.input.file.stem

        # Export tables
        for table_ix, table in enumerate(conv_res.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe(doc=conv_res.document)
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
    