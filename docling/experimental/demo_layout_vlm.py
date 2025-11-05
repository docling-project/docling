#!/usr/bin/env python3
"""Demo script for the new ThreadedLayoutVlmPipeline.

This script demonstrates the usage of the new pipeline that combines
layout model preprocessing with VLM processing in a threaded manner.
"""

from pathlib import Path
import argparse
import logging
from io import BytesIO


from docling.datamodel.base_models import InputFormat
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS, GRANITEDOCLING_VLLM
from docling.experimental.datamodel.threaded_layout_vlm_pipeline_options import ThreadedLayoutVlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.experimental.pipeline.threaded_layout_vlm_pipeline import ThreadedLayoutVlmPipeline
from docling.datamodel.base_models import ConversionStatus, DocumentStream
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions


_log = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description='Demo script for the new ThreadedLayoutVlmPipeline')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing PDF files'
    )    
    parser.add_argument(
        '--output',
        type=str,
        default='../results/',
        help='Output directory for converted files'
    )
    return parser.parse_args()


def _get_docs(input_doc_paths):
    '''Yield DocumentStream objects from list of input document paths'''
    for path in input_doc_paths:
        buf = BytesIO(path.read_bytes())
        stream = DocumentStream(name=path.name, stream=buf)
        yield stream


def demo_threaded_layout_vlm_pipeline(input_doc_paths: list[Path], out_dir_layout_aware: Path, out_dir_classic_vlm: Path):
    """Demonstrate the threaded layout+VLM pipeline."""

    # Configure pipeline options
    print("Configuring pipeline options...")
    pipeline_options_layout_aware = ThreadedLayoutVlmPipelineOptions(
        # VLM configuration - defaults to GRANITEDOCLING_TRANSFORMERS
        vlm_options=GRANITEDOCLING_TRANSFORMERS,
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

    pipeline_options_classic_vlm = VlmPipelineOptions(vlm_otpions=GRANITEDOCLING_VLLM)

    # Create converter with the new pipeline
    print("Initializing DocumentConverter (this may take a while - loading models)...")
    doc_converter_layout_enhanced = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedLayoutVlmPipeline,
                pipeline_options=pipeline_options_layout_aware
            )
        }
    )
    doc_converter_classic_vlm = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options_classic_vlm,
            ),
        }
    )

    print(f"Starting conversion of {len(input_doc_paths)} document(s)...")
    result_layout_aware = doc_converter_layout_enhanced.convert_all(list(_get_docs(input_doc_paths)), raises_on_error=False)
    result_without_layout = doc_converter_classic_vlm.convert_all(list(_get_docs(input_doc_paths)), raises_on_error=False)

    for conv_result in result_layout_aware:
        if conv_result.status == ConversionStatus.FAILURE:
            _log.error(f"Conversion failed: {conv_result.status}")
            continue

        doc_filename = conv_result.input.file.stem
        conv_result.document.save_as_doctags(out_dir_layout_aware / f"{doc_filename}.dt")

    for conv_result in result_without_layout:
        if conv_result.status == ConversionStatus.FAILURE:
            _log.error(f"Conversion failed: {conv_result.status}")
            continue

        doc_filename = conv_result.input.file.stem
        conv_result.document.save_as_doctags(out_dir_classic_vlm / f"{doc_filename}.dt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("Starting script...")
        args = _parse_args()
        print(f"Parsed arguments: input={args.input}, output={args.output}")

        base_path = Path(args.input)

        print(f"Searching for PDFs in: {base_path}")
        input_doc_paths = sorted(list(base_path.rglob("*.*")))
        input_doc_paths = [e for e in input_doc_paths if e.name.endswith(".pdf") or e.name.endswith(".PDF")]

        if not input_doc_paths:
            _log.error(f"ERROR: No PDF files found in {base_path}")

        print(f"Found {len(input_doc_paths)} PDF file(s):")

        out_dir_layout_aware = Path(args.output) / "layout_aware" / "model_output" / "layout" / "doc_tags"
        out_dir_classic_vlm = Path(args.output) / "classic_vlm" / "model_output" / "layout" / "doc_tags"
        out_dir_layout_aware.mkdir(parents=True, exist_ok=True)
        out_dir_classic_vlm.mkdir(parents=True, exist_ok=True)

        _log.info("Calling demo_threaded_layout_vlm_pipeline...")
        demo_threaded_layout_vlm_pipeline(input_doc_paths, out_dir_layout_aware, out_dir_classic_vlm)
        _log.info("Script completed successfully!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
