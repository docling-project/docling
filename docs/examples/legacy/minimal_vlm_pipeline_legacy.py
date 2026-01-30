# %% [markdown]
# Minimal VLM pipeline example (LEGACY VERSION - for backward compatibility testing)
#
# **NOTE:** This is the legacy version using `vlm_model_specs` directly.
# For the new preset-based approach, see `minimal_vlm_pipeline.py`.
# This file is kept to validate backward compatibility with the old API.
#
# What this example does
# - Runs the VLM-powered pipeline on a PDF (by URL) and prints Markdown output.
# - Shows two setups: default (Transformers/GraniteDocling) and macOS MPS/MLX.
# - Uses the LEGACY vlm_model_specs approach (still supported for backward compatibility)
#
# Prerequisites
# - Install Docling with VLM extras and the appropriate backend (Transformers or MLX).
# - Ensure your environment can download model weights (e.g., from Hugging Face).
#
# How to run
# - From the repository root, run: `python docs/examples/minimal_vlm_pipeline_legacy.py`.
# - The script prints the converted Markdown to stdout.
#
# Notes
# - `source` may be a local path or a URL to a PDF.
# - The second section demonstrates macOS MPS acceleration via MLX (`vlm_model_specs.GRANITEDOCLING_MLX`).
# - For the NEW preset-based approach, see `docs/examples/minimal_vlm_pipeline.py`.

# %%

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Convert a public arXiv PDF; replace with a local path if preferred.
source = "https://arxiv.org/pdf/2501.17887"

###### USING SIMPLE DEFAULT VALUES
# - GraniteDocling model
# - Using the transformers framework

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())


###### USING MACOS MPS ACCELERATOR
# Demonstrates using MLX on macOS with MPS acceleration (macOS only).
# For more options see the `compare_vlm_models.py` example.

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
