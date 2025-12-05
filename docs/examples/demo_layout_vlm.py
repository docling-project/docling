#!/usr/bin/env python3
"""Demo script for the new ThreadedLayoutVlmPipeline.

This script demonstrates the usage of the experimental ThreadedLayoutVlmPipeline pipeline
that combines layout model preprocessing with VLM processing in a threaded manner.
"""

import argparse
import logging
import traceback
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
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
        description="Demo script for the experimental ThreadedLayoutVlmPipeline"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="tests/data/pdf/code_and_formula.pdf",
        help="Path to a PDF file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scratch/demo_layout_vlm/",
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


def openai_compatible_vlm_options(
    model: str,
    prompt: str,
    format: ResponseFormat,
    hostname_and_port,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: str = "",
    skip_special_tokens=False,
):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    options = ApiVlmOptions(
        url=f"http://{hostname_and_port}/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
        params=dict(
            model=model,
            max_tokens=max_tokens,
            skip_special_tokens=skip_special_tokens,  # needed for VLLM
        ),
        headers=headers,
        prompt=prompt,
        timeout=90,
        scale=2.0,
        temperature=temperature,
        response_format=format,
    )

    return options


def demo_threaded_layout_vlm_pipeline(
    input_doc_path: Path, out_dir_layout_aware: Path, use_api_vlm: bool
):
    """Demonstrate the threaded layout+VLM pipeline."""

    vlm_options = GRANITEDOCLING_TRANSFORMERS.model_copy()

    if use_api_vlm:
        vlm_options = openai_compatible_vlm_options(
            model="granite-docling-258m-mlx",  # For VLLM use "ibm-granite/granite-docling-258M"
            hostname_and_port="localhost:1234",  # LM studio defaults to port 1234, VLLM to 8000
            prompt="Convert this page to docling.",
            format=ResponseFormat.DOCTAGS,
            api_key="",
        )
    vlm_options.track_input_prompt = True

    # Configure pipeline options
    print("Configuring pipeline options...")
    pipeline_options_layout_aware = ThreadedLayoutVlmPipelineOptions(
        # VLM configuration - defaults to GRANITEDOCLING_TRANSFORMERS
        vlm_options=vlm_options,
        # Layout configuration - defaults to DOCLING_LAYOUT_HERON
        # Batch sizes for parallel processing
        layout_batch_size=2,
        vlm_batch_size=1,
        # Queue configuration
        queue_max_size=10,
        # Image processing
        images_scale=vlm_options.scale,
        generate_page_images=True,
        enable_remote_services=use_api_vlm,
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
        source=input_doc_path, raises_on_error=False
    )

    if result_layout_aware.status == ConversionStatus.FAILURE:
        _log.error(f"Conversion failed: {result_layout_aware.status}")

    doc_filename = result_layout_aware.input.file.stem
    result_layout_aware.document.save_as_json(
        out_dir_layout_aware / f"{doc_filename}.json"
    )

    result_layout_aware.document.save_as_html(
        out_dir_layout_aware / f"{doc_filename}.html", split_page_view=True
    )
    for page in result_layout_aware.pages:
        _log.info("Page %s of VLM response:", page.page_no)
        if page.predictions.vlm_response:
            _log.info(page.predictions.vlm_response)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        args = _parse_args()
        _log.info(
            f"Parsed arguments: input={args.input_file}, output={args.output_dir}"
        )

        input_path = Path(args.input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_path}")

        out_dir_layout_aware = Path(args.output_dir) / "layout_aware/"
        out_dir_layout_aware.mkdir(parents=True, exist_ok=True)

        use_api_vlm = False  # Set to False to use inline VLM model

        demo_threaded_layout_vlm_pipeline(input_path, out_dir_layout_aware, use_api_vlm)
    except Exception:
        traceback.print_exc()
        raise
