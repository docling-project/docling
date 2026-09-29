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
# - For LM Studio: `python docs/examples/chart_extraction.py --lmstudio`
# - Outputs are written to `scratch/`.
#
# Input document
# - Defaults to `docs/examples/data/chart_document.pdf`. Change `input_doc_path`
#   as needed.
#
# Notes
# - Setting `do_chart_extraction=True` automatically enables picture classification.
# - Supported chart types: bar chart, pie chart, line chart.
# - The default preset uses the local Transformers runtime (granite_vision_v4).
#   Pass --lmstudio to use the GGUF model served by LM Studio instead.

# %%

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc import ImageRefMode, PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.chart_extraction_options import ChartExtractionVlmEngineOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def make_chart_options(lmstudio: bool = False) -> ChartExtractionVlmEngineOptions:
    """Return chart extraction options for the requested backend.

    Args:
        lmstudio: When True, use the granite-vision-4.1-4b GGUF model served
            by LM Studio on its default local endpoint
            (http://localhost:1234/v1/chat/completions).
            When False (default), run the HuggingFace model locally via
            Transformers.

    Returns:
        A :class:`ChartExtractionVlmEngineOptions` configured for the chosen
        backend.  Both paths use the same ``granite_vision_v4`` preset so the
        model identifier, prompt tokens, and output flags are identical.
    """
    if lmstudio:
        return ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API_LMSTUDIO,
                # LM Studio default endpoint — change if you moved it
                url="http://localhost:1234/v1/chat/completions",
            ),
        )
    # Default: local Transformers runtime
    return ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Chart extraction example")
    parser.add_argument(
        "--lmstudio",
        action="store_true",
        help=(
            "Use the granite-vision-4.1-4b GGUF model served by LM Studio "
            "(http://localhost:1234/v1/chat/completions) instead of the local "
            "Transformers runtime."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data/chart_document.pdf",
        help="Path to the input PDF (default: docs/examples/data/chart_document.pdf)",
    )
    args = parser.parse_args()

    input_doc_path: Path = args.input
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_options = make_chart_options(lmstudio=args.lmstudio)
    backend = "LM Studio" if args.lmstudio else "Transformers (local)"
    _log.info(f"Chart extraction backend: {backend}")
    _log.info(f"  model  : {chart_options.model_spec.name}")
    _log.info(f"  engine : {chart_options.engine_options.engine_type.value}")
    _log.info(
        f"  outputs: csv={chart_options.chart2csv}  summary={chart_options.chart2summary}  code={chart_options.chart2code}"
    )

    # Configure the PDF pipeline with chart extraction enabled.
    # This automatically enables picture classification as well.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_chart_extraction = True
    pipeline_options.chart_extraction_options = chart_options
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.enable_remote_services = args.lmstudio

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

        chart_df = pd.DataFrame(grid)
        print(chart_df.to_csv(index=False, header=False))

    # Export the full document as split-page HTML with layout.
    html_filename = output_dir / f"{doc_filename}.html"
    ser = HTMLDocSerializer(
        doc=conv_res.document,
        params=HTMLParams(
            image_mode=ImageRefMode.EMBEDDED,
            output_style=HTMLOutputStyle.SPLIT_PAGE,
        ),
    )
    visualizer = LayoutVisualizer()
    visualizer.params.show_label = False
    ser_res = ser.serialize(
        visualizer=visualizer,
    )
    with open(html_filename, "w") as fw:
        fw.write(ser_res.text)
    _log.info(f"Saved split-page HTML to {html_filename}")

    elapsed = time.time() - start_time
    _log.info(f"Document converted and exported in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
