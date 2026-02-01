# %% [markdown]
# Picture Description with Legacy VLM Options
#
# This example demonstrates the LEGACY approach using PictureDescriptionVlmOptions
# with direct repo_id specification (no preset system).
#
# For the NEW approach with preset support, see: picture_description_inline.py
#
# What this example does:
# - Uses the legacy PictureDescriptionVlmOptions with direct repo_id
# - Shows backward compatibility with the old implementation
# - Demonstrates the PictureDescriptionVlmModel (not the runtime-based version)
#
# Prerequisites:
# - Install Docling with VLM extras: `pip install docling[vlm]`
#
# How to run:
# - From the repository root: `python docs/examples/legacy/picture_description_inline_legacy.py`

# %%

from pathlib import Path

from docling_core.types.doc import PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# %%
# Example 1: Legacy approach with direct repo_id specification

IMAGE_RESOLUTION_SCALE = 2.0

input_doc_path = Path("./tests/data/pdf/2206.01062.pdf")

# Configure pipeline with legacy VLM options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

# Legacy: Direct repo_id specification (no preset system)
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    prompt="Describe this image in a few sentences.",
    scale=IMAGE_RESOLUTION_SCALE,
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

result = doc_converter.convert(input_doc_path)

# Print picture descriptions
print("\n" + "=" * 80)
print("PICTURE DESCRIPTIONS (Legacy Approach)")
print("=" * 80)

for item, _ in result.document.iterate_items():
    if isinstance(item, PictureItem):
        print(
            f"Picture {item.self_ref}\n"
            f"Caption: {item.caption_text(doc=result.document)}\n"
            f"Meta: {item.meta}"
        )

# %%
# Example 2: Legacy approach with custom prompt

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

# Legacy: Custom prompt with direct repo_id
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    prompt="What is shown in this image? Provide a detailed technical description.",
    scale=IMAGE_RESOLUTION_SCALE,
    generation_config={
        "max_new_tokens": 300,
        "do_sample": False,
    },
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

result = doc_converter.convert(input_doc_path)

print("\n" + "=" * 80)
print("PICTURE DESCRIPTIONS (Legacy with Custom Prompt)")
print("=" * 80)

for item, _level in result.document.iterate_items():
    if isinstance(item, PictureItem):
        print(
            f"Picture {item.self_ref}\n"
            f"Caption: {item.caption_text(doc=result.document)}\n"
            f"Meta: {item.meta}"
        )

print("\n" + "=" * 80)
print("NOTE: This is the LEGACY approach.")
print("For the NEW preset-based approach, see: picture_description_inline.py")
print("=" * 80)
