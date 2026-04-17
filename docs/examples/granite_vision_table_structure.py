# %% [markdown]
# Extract tables from a PDF using Granite Vision for table structure recognition.
#
# What this example does
# - Converts a PDF using the Granite Vision VLM for table structure extraction
#   instead of the default TableFormer model.
# - Prints each detected table as Markdown to stdout.
#
# Prerequisites
# - Install Docling with VLM support: `pip install docling[vlm]`
# - A CUDA GPU is recommended; CPU works but is significantly slower.
#
# How to run
# - From the repo root: `python docs/examples/granite_vision_table_structure.py`
#
# Input document
# - Defaults to `tests/data/pdf/2206.01062.pdf`. Change `input_doc_path` as needed.
#
# Notes
# - The Granite Vision model (`ibm-granite/granite-4.0-3b-vision`) is downloaded
#   automatically from HuggingFace on first run.
# - The model outputs table structure in OTSL (Open Table Structure Language) format,
#   which Docling parses into structured table cells.

# %%

import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    GraniteVisionTableStructureOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    # Configure pipeline to use Granite Vision for table structure
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = GraniteVisionTableStructureOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)
    elapsed = time.time() - start_time

    for table_ix, table in enumerate(conv_res.document.tables):
        table_df = table.export_to_dataframe(doc=conv_res.document)
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())
        print()

    _log.info(
        f"Document converted in {elapsed:.2f} seconds "
        f"({len(conv_res.document.tables)} tables found)."
    )


if __name__ == "__main__":
    main()
