# %% [markdown]
# Picture Description with Inline VLM Models
#
# What this example does
# - Demonstrates picture description in standard PDF pipeline
# - Shows default preset, changing presets, and manual configuration without presets
# - Enriches documents with AI-generated image captions
#
# Prerequisites
# - Install Docling with VLM extras: `pip install docling[vlm]`
# - Ensure your environment can download model weights
#
# How to run
# - From the repository root: `python docs/examples/picture_description_inline.py`
#
# Notes
# - This uses the standard PDF pipeline (not VlmPipeline)
# - For API-based picture description, see `pictures_description_api.py`
# - For legacy PictureDescriptionVlmOptions approach, see `picture_description_inline_legacy.py`

# %%

import logging
import os
from pathlib import Path

from docling_core.types.doc import PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    PictureDescriptionVlmRuntimeOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_runtime_options import AutoInlineVlmRuntimeOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO)

# Test document with images
input_doc_path = Path("tests/data/pdf/2206.01062.pdf")

# Check if running in CI
IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")

###### EXAMPLE 1: Using default VLM for picture description (SmolVLM)

print("=" * 60)
print("Example 1: Default picture description (SmolVLM preset)")
print("=" * 60)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
# When no picture_description_options is set, it uses the default (SmolVLM)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert(input_doc_path)

# Print picture descriptions
for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        print(
            f"Picture {element.self_ref}\n"
            f"Caption: {element.caption_text(doc=result.document)}\n"
            f"Meta: {element.meta}"
        )


###### EXAMPLE 2: Change to Granite Vision preset (skipped in CI)

if not IS_CI:
    print("\n" + "=" * 60)
    print("Example 2: Using Granite Vision preset")
    print("=" * 60)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = (
        PictureDescriptionVlmRuntimeOptions.from_preset("granite_vision")
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(input_doc_path)

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"Picture {element.self_ref}\n"
                f"Caption: {element.caption_text(doc=result.document)}\n"
                f"Meta: {element.meta}"
            )
else:
    print("\n" + "=" * 60)
    print("Example 2: Skipped (running in CI environment)")
    print("=" * 60)


###### EXAMPLE 3: Without presets - manually configuring model and runtime

print("\n" + "=" * 60)
print("Example 3: Manual configuration without presets")
print("=" * 60)

# You can manually configure the model spec and runtime options without using presets

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = PictureDescriptionVlmRuntimeOptions(
    model_spec=VlmModelSpec(
        name="SmolVLM-256M-Custom",
        default_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt="Provide a detailed technical description of this image, focusing on any diagrams, charts, or technical content.",
        response_format=ResponseFormat.PLAINTEXT,
    ),
    runtime_options=AutoInlineVlmRuntimeOptions(),
    prompt="Provide a detailed technical description of this image, focusing on any diagrams, charts, or technical content.",
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert(input_doc_path)

for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        print(
            f"Picture {element.self_ref}\n"
            f"Caption: {element.caption_text(doc=result.document)}\n"
            f"Meta: {element.meta}"
        )


# %% [markdown]
# ## Summary
#
# This example shows three approaches:
# 1. **Default**: No configuration needed, uses SmolVLM preset automatically
# 2. **Preset-based**: Use `from_preset()` to select a different model (e.g., granite_vision)
# 3. **Manual configuration**: Manually create VlmModelSpec and runtime options without presets
#
# Available presets: smolvlm, granite_vision, pixtral, qwen
#
# For API-based picture description (vLLM, LM Studio, watsonx.ai), see `pictures_description_api.py`
# For the legacy approach using PictureDescriptionVlmOptions, see `picture_description_inline_legacy.py`
