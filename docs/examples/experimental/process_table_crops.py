"""Run Docling on an image using the experimental TableCrops layout model."""

from __future__ import annotations

from pathlib import Path

import docling
from docling.datamodel.document import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.experimental.datamodel.table_crops_layout_options import (
    TableCropsLayoutOptions,
)
from docling.experimental.models.table_crops_layout_model import TableCropsLayoutModel
from docling.models.factories import get_layout_factory


def main() -> None:
    # Go up 3 levels to escape 'docs' and reach the repo root
    sample_image = Path(__file__).resolve().parents[3] / "tests" / "data" / "2305.03393v1-table_crop.png"

    pipeline_options = ThreadedPdfPipelineOptions(
        layout_options=TableCropsLayoutOptions(),
        do_table_structure=True,
        generate_page_images=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        },
    )

    conv_res = converter.convert(sample_image)

    print(conv_res.document.tables[0].export_to_markdown())


if __name__ == "__main__":
    main()
