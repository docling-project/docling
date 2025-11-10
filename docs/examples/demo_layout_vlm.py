#!/usr/bin/env python3
"""Demo script for the new ThreadedLayoutVlmPipeline.

This script demonstrates the usage of the new pipeline that combines
layout model preprocessing with VLM processing in a threaded manner.
"""

import argparse
import logging
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.vlm_model_specs import (
    GRANITEDOCLING_MLX,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.experimental.datamodel.threaded_layout_vlm_pipeline_options import (
    ThreadedLayoutVlmPipelineOptions,
)
from docling.experimental.pipeline.threaded_layout_vlm_pipeline import (
    ThreadedLayoutVlmPipeline,
)

_log = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Demo script for the new ThreadedLayoutVlmPipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="tests/data/pdf/2203.01017v2.pdf",
        help="Path to a PDF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/",
        help="Output directory for converted files",
    )
    return parser.parse_args()


# Can be used to read multiple pdf files under a folder
# def _get_docs(input_doc_path):
#     """Yield DocumentStream objects from list of input document paths"""
#     for path in input_doc_path:
#         buf = BytesIO(path.read_bytes())
#         stream = DocumentStream(name=path.name, stream=buf)
#         yield stream


def demo_threaded_layout_vlm_pipeline(input_doc_path: Path, out_dir_layout_aware: Path):
    """Demonstrate the threaded layout+VLM pipeline."""

    # Configure pipeline options
    print("Configuring pipeline options...")
    pipeline_options_layout_aware = ThreadedLayoutVlmPipelineOptions(
        # VLM configuration - defaults to GRANITEDOCLING_TRANSFORMERS
        vlm_options=GRANITEDOCLING_MLX,
        # Layout configuration - defaults to DOCLING_LAYOUT_HERON
        # Batch sizes for parallel processing
        layout_batch_size=2,
        vlm_batch_size=1,
        # Queue configuration
        queue_max_size=10,
        batch_timeout_seconds=1.0,
        # Layout coordinate injection
        include_layout_coordinates=True,
        coordinate_precision=1,
        # Image processing
        images_scale=2.0,
        generate_page_images=True,
    )

    # Create converter with the new pipeline
    print("Initializing DocumentConverter (this may take a while - loading models)...")
    doc_converter_layout_enhanced = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedLayoutVlmPipeline,
                pipeline_options=pipeline_options_layout_aware,
            )
        }
    )

    result_layout_aware = doc_converter_layout_enhanced.convert(
        input_doc_path, raises_on_error=False
    )

    for conv_result in result_layout_aware:
        if conv_result.status == ConversionStatus.FAILURE:
            _log.error(f"Conversion failed: {conv_result.status}")
            continue

        doc_filename = conv_result.input.file.stem
        conv_result.document.save_as_json(out_dir_layout_aware / f"{doc_filename}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        args = _parse_args()
        print(f"Parsed arguments: input={args.input}, output={args.output}")

        input_path = Path(args.input)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_path}")

        out_dir_layout_aware = Path(args.output) / "layout_aware/"

        out_dir_layout_aware.mkdir(parents=True, exist_ok=True)

        demo_threaded_layout_vlm_pipeline(input_path, out_dir_layout_aware)
        _log.info("Script completed successfully!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise
