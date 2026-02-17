# %% [markdown]
# Minimal VLM pipeline example: convert a PDF using a vision-language model.
#
# What this example does
# - Runs the VLM-powered pipeline on a PDF (by URL) and prints Markdown output.
# - Shows three setups: default (no config), using presets, and runtime overrides.
# - Demonstrates both the simplest approach and the NEW preset-based system.
#
# Prerequisites
# - Install Docling with VLM extras and the appropriate backend (Transformers or MLX).
# - Ensure your environment can download model weights (e.g., from Hugging Face).
#
# How to run
# - From the repository root, run: `python docs/examples/minimal_vlm_pipeline.py`.
# - The script prints the converted Markdown to stdout.
#
# Notes
# - `source` may be a local path or a URL to a PDF.
# - For the LEGACY approach (backward compatibility), see `docs/examples/minimal_vlm_pipeline_legacy.py`.
# - For more preset examples and runtime options, see `docs/examples/vlm_presets_and_runtimes.py`.

# %%

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import (
    MlxVlmEngineOptions,
    VlmEngineType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Convert a public arXiv PDF; replace with a local path if preferred.
source = "https://arxiv.org/pdf/2501.17887"

###### EXAMPLE 1: USING DEFAULT SETTINGS (SIMPLEST)
# - No configuration needed
# - Uses default VLM model (GraniteDocling)
# - Auto-selects the best runtime for your platform

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())


###### EXAMPLE 2: USING PRESETS (RECOMMENDED)
# - Uses the "granite_docling" preset explicitly
# - Same as default but more explicit and configurable
# - Auto-selects the best runtime for your platform (Transformers by default)

vlm_options = VlmConvertOptions.from_preset("granite_docling")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())


###### EXAMPLE 3: USING PRESETS WITH RUNTIME OVERRIDE (ADVANCED)
# Demonstrates using the same preset but overriding the runtime to use MLX
# on macOS with MPS acceleration. The preset automatically uses the MLX-specific
# model variant when available.

vlm_options = VlmConvertOptions.from_preset(
    "granite_docling",
    engine_options=MlxVlmEngineOptions(),
)

# The preset automatically selects the MLX-optimized model variant
print(f"Using model: {vlm_options.model_spec.get_repo_id(VlmEngineType.MLX)}")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
