# %% [markdown]
# What this example does
# - Run a batch conversion on a parquet file with an image column.
#
# Requirements
# - Python 3.9+
# - Install Docling: `pip install docling`
#
# How to run
# - `python docs/examples/parquet_images.py FILE`
#
# The parquet file should be in the format similar to the ViDoRe V3 dataset.
# https://huggingface.co/collections/vidore/vidore-benchmark-v3
#
# For example:
# - https://huggingface.co/datasets/vidore/vidore_v3_hr/blob/main/corpus/test-00000-of-00001.parquet
#
# ### Start models with vllm
# ```console
# vllm serve ibm-granite/granite-docling-258M \
#   --host 127.0.0.1 --port 8000 \
#   --max-num-seqs 512 \
#   --max-num-batched-tokens 8192 \
#   --enable-chunked-prefill \
#   --gpu-memory-utilization 0.9
# ```
# %%

import io
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import pyarrow.parquet as pq
import typer
from PIL import Image

from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PipelineOptions,
    RapidOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.pipeline.base_pipeline import ConvertPipeline
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.accelerator_utils import decide_device


def process_document(
    images: list[Image.Image], chunk_idx: int, doc_converter: DocumentConverter
):
    """Builds a tall image and sends it through Docling."""

    print(f"\n--- Processing chunk {chunk_idx} with {len(images)} images ---")

    # Convert images to mode RGB (TIFF pages must match)
    rgb_images = [im.convert("RGB") for im in images]

    # First image is the base frame
    first = rgb_images[0]
    rest = rgb_images[1:]

    # Create multi-page TIFF using PIL frames
    buf = io.BytesIO()
    first.save(
        buf,
        format="TIFF",
        save_all=True,
        append_images=rest,
        compression="tiff_deflate",  # good compression, optional
    )
    buf.seek(0)

    # Docling conversion
    doc_stream = DocumentStream(name=f"doc_{chunk_idx}.tiff", stream=buf)

    start_time = time.time()
    conv_result = doc_converter.convert(doc_stream)
    runtime = time.time() - start_time

    assert conv_result.status == ConversionStatus.SUCCESS

    pages = len(conv_result.pages)
    print(
        f"Chunk {chunk_idx} converted in {runtime:.2f} sec ({pages / runtime:.2f} pages/s)."
    )


def run(
    filename: Annotated[Path, typer.Argument()] = Path(
        "docs/examples/data/vidore_v3_hr-slice.parquet"
    ),
    doc_size: int = 192,
    batch_size: int = 64,
    pipeline: Literal["standard", "vlm", "legacy"] = "standard",
):
    acc_opts = AcceleratorOptions()
    device = decide_device(acc_opts.device)

    ocr_options = RapidOcrOptions()
    if "cuda" in device:
        ocr_options = RapidOcrOptions(backend="torch")

    # On Python 3.14 we only have torch
    if sys.version_info >= (3, 14):
        ocr_options = RapidOcrOptions(backend="torch")

    if pipeline == "standard":
        pipeline_cls: type[ConvertPipeline] = StandardPdfPipeline
        pipeline_options: PipelineOptions = PdfPipelineOptions(
            ocr_options=ocr_options,
            ocr_batch_size=batch_size,
            layout_batch_size=batch_size,
            table_batch_size=4,
        )
    elif pipeline == "legacy":
        settings.perf.page_batch_size = batch_size
        pipeline_cls: type[ConvertPipeline] = LegacyStandardPdfPipeline
        pipeline_options: PipelineOptions = PdfPipelineOptions(
            ocr_options=ocr_options,
            ocr_batch_size=batch_size,
            layout_batch_size=batch_size,
            table_batch_size=4,
        )
    elif pipeline == "vlm":
        settings.perf.page_batch_size = batch_size
        pipeline_cls = VlmPipeline
        vlm_options = ApiVlmOptions(
            url="http://localhost:8000/v1/chat/completions",
            params=dict(
                model=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.repo_id,
                max_tokens=4096,
                skip_special_tokens=True,
            ),
            prompt=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.prompt,
            timeout=90,
            scale=1.0,
            temperature=0.0,
            concurrency=batch_size,
            stop_strings=["</doctag>", "<|end_of_text|>"],
            response_format=ResponseFormat.DOCTAGS,
        )
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,  # required when using a remote inference service.
        )
    else:
        raise RuntimeError(f"Pipeline {pipeline} not available.")

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=pipeline_cls,
                pipeline_options=pipeline_options,
            )
        }
    )

    start_time = time.time()
    doc_converter.initialize_pipeline(InputFormat.IMAGE)
    init_runtime = time.time() - start_time
    print(f"Pipeline initialized in {init_runtime:.2f} seconds.")

    # ------------------------------------------------------------
    # Open parquet file in streaming mode
    # ------------------------------------------------------------
    pf = pq.ParquetFile(filename)

    image_buffer = []  # holds up to doc_size images
    chunk_idx = 0

    # ------------------------------------------------------------
    # Stream batches from parquet
    # ------------------------------------------------------------
    for batch in pf.iter_batches(batch_size=batch_size, columns=["image"]):
        col = batch.column("image")

        # Extract Python objects (PIL images)
        # Arrow stores them as Python objects inside an ObjectArray
        for i in range(len(col)):
            img_dict = col[i].as_py()  # {"bytes": ..., "path": ...}
            pil_image = Image.open(io.BytesIO(img_dict["bytes"]))
            image_buffer.append(pil_image)

            # If enough images gathered → process one doc
            if len(image_buffer) == doc_size:
                process_document(image_buffer, chunk_idx, doc_converter)
                image_buffer.clear()
                chunk_idx += 1

    # ------------------------------------------------------------
    # Process trailing images (last partial chunk)
    # ------------------------------------------------------------
    if image_buffer:
        process_document(image_buffer, chunk_idx, doc_converter)


if __name__ == "__main__":
    typer.run(run)
