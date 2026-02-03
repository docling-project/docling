# %% [markdown]
# Extract chart data from a PDF and export the result as split-page HTML with layout.
#
# What this example does
# - Converts a PDF with chart extraction enrichment enabled.
# - Iterates detected pictures and prints extracted chart data as CSV to stdout.
# - Saves the converted document as split-page HTML with layout to `scratch/`.
#
# Prerequisites
# - Install Docling with the `granite_vision` extra (for chart extraction model).
# - Install `pandas`.
#
# How to run
# - From the repo root: `python docs/examples/chart_extraction.py`.
# - Outputs are written to `scratch/`.
#
# Input document
# - Defaults to `docs/examples/data/chart_document.pdf`. Change `input_doc_path`
#   as needed.
#
# Notes
# - Enabling `do_chart_extraction` automatically enables picture classification.
# - Supported chart types: bar chart, pie chart, line chart.

# %%

import logging
import time
from pathlib import Path

import pandas as pd
from docling_core.types.doc import ImageRefMode, PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path(__file__).parent / "data/chart_document.pdf"
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure the PDF pipeline with chart extraction enabled.
    # This automatically enables picture classification as well.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_chart_extraction = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    doc_filename = conv_res.input.file.stem

    # Iterate over document items and print extracted chart data.
    for item, _level in conv_res.document.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        if item.meta is None:
            continue

        # Check if the picture was classified as a chart.
        if item.meta.classification is not None:
            chart_type = item.meta.classification.get_main_prediction().class_name
        else:
            continue

        # Check if chart data was extracted.
        if item.meta.tabular_chart is None:
            continue

        table_data = item.meta.tabular_chart.chart_data
        print(f"## Chart type: {chart_type}")
        print(f"   Size: {table_data.num_rows} rows x {table_data.num_cols} cols")

        # Build a DataFrame from the extracted table cells for display.
        grid: list[list[str]] = [
            [""] * table_data.num_cols for _ in range(table_data.num_rows)
        ]
        for cell in table_data.table_cells:
            grid[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text

        df = pd.DataFrame(grid)
        print(df.to_csv(index=False, header=False))

    # Export the full document as split-page HTML with layout.
    html_filename = output_dir / f"{doc_filename}.html"
    conv_res.document.save_as_html(
        html_filename,
        image_mode=ImageRefMode.EMBEDDED,
        split_page_view=True,
    )
    _log.info(f"Saved split-page HTML to {html_filename}")

    elapsed = time.time() - start_time
    _log.info(f"Document converted and exported in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
