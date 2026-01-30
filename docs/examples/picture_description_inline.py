# %% [markdown]
# Picture Description with Inline VLM Models
#
# What this example does
# - Demonstrates picture description in standard PDF pipeline
# - Shows default preset, changing presets, and legacy repo_id approach
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
# - For legacy approach, see `picture_description_inline_legacy.py`

# %%

import logging
from pathlib import Path

from docling_core.types.doc import PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    PictureDescriptionVlmRuntimeOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO)

# Test document with images
input_doc_path = Path("tests/data/pdf/2206.01062.pdf")

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


###### EXAMPLE 2: Change to Granite Vision preset

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


###### EXAMPLE 3: Without presets - using HF repo_id directly with custom prompt

print("\n" + "=" * 60)
print("Example 3: Using repo_id directly (legacy approach)")
print("=" * 60)

# This demonstrates the legacy approach for backward compatibility
# You can specify the HuggingFace repo_id directly and customize the prompt

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
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
# 3. **Legacy repo_id**: Directly specify HuggingFace repo_id with custom prompt
#
# Available presets: smolvlm, granite_vision, pixtral, qwen
#
# For API-based picture description (vLLM, LM Studio, watsonx.ai), see `pictures_description_api.py`
