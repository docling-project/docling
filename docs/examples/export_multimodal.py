# %% [markdown]
# Export multimodal page data (image bytes, text, segments) to a Parquet file.
#
# What this example does
# - Converts a PDF and assembles per-page multimodal records: image, cells, text, segments.
# - Normalizes records to a pandas DataFrame and writes a timestamped `.parquet` in `scratch/`.
#
# Prerequisites
# - Install Docling and `pandas`. Optional: `datasets` and `Pillow` for the commented demo.
#
# How to run
# - From the repo root: `python docs/examples/export_multimodal.py`.
# - Output parquet is written to `scratch/`.
#
# Key options
# - `IMAGE_RESOLUTION_SCALE`: page rendering scale (1 ~ 72 DPI).
# - `PdfPipelineOptions.generate_page_images`: keep page images for export.
#
# Requirements
# - Writing Parquet requires an engine such as `pyarrow` or `fastparquet`
#   (`pip install pyarrow` is the most common choice).
#
# Input document
# - Defaults to `tests/data/pdf/2206.01062.pdf`. Change `input_doc_path` as needed.
#
# Notes
# - The commented block at the bottom shows how to load the Parquet with HF Datasets
#   and reconstruct images from raw bytes.

# %%

import datetime
import logging
import time
from pathlib import Path

import pandas as pd

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    # Keep page images so they can be exported to the multimodal rows.
    # Use PdfPipelineOptions.images_scale to control the render scale (1 ~ 72 DPI).
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (
        content_text,
        content_md,
        content_dt,
        page_cells,
        page_segments,
        page,
    ) in generate_multimodal_pages(conv_res):
        dpi = page._default_image_scale * 72

        rows.append(
            {
                "document": conv_res.input.file.name,
                "hash": conv_res.input.document_hash,
                "page_hash": create_hash(
                    conv_res.input.document_hash + ":" + str(page.page_no - 1)
                ),
                "image": {
                    "width": page.image.width,
                    "height": page.image.height,
                    "bytes": page.image.tobytes(),
                },
                "cells": page_cells,
                "contents": content_text,
                "contents_md": content_md,
                "contents_dt": content_dt,
                "segments": page_segments,
                "extra": {
                    "page_num": page.page_no + 1,
                    "width_in_points": page.size.width,
                    "height_in_points": page.size.height,
                    "dpi": dpi,
                },
            }
        )

    # Generate one parquet from all documents
    df_result = pd.json_normalize(rows)
    now = datetime.datetime.now()
    output_filename = output_dir / f"multimodal_{now:%Y-%m-%d_%H%M%S}.parquet"
    df_result.to_parquet(output_filename)

    end_time = time.time() - start_time

    _log.info(
        f"Document converted and multimodal pages generated in {end_time:.2f} seconds."
    )

    # This block demonstrates how the file can be opened with the HF datasets library
    # from datasets import Dataset
    # from PIL import Image
    # multimodal_df = pd.read_parquet(output_filename)

    # # Convert pandas DataFrame to Hugging Face Dataset and load bytes into image
    # dataset = Dataset.from_pandas(multimodal_df)
    # def transforms(examples):
    #     examples["image"] = Image.frombytes('RGB', (examples["image.width"], examples["image.height"]), examples["image.bytes"], 'raw')
    #     return examples
    # dataset = dataset.map(transforms)


if __name__ == "__main__":
    main()
