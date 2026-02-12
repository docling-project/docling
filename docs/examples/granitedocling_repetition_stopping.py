# %% [markdown]
# Experimental VLM pipeline with custom repetition stopping criteria (LEGACY).
#
# **NOTE:** This example uses the LEGACY vlm_model_specs approach because
# custom_stopping_criteria is a feature of the old InlineVlmOptions system.
# This feature is not yet migrated to the new preset/runtime system.
#
# This script demonstrates the use of custom stopping criteria that detect
# repetitive location coordinate patterns in generated text and stop generation
# when such patterns are found.
#
# What this example does
# - Uses the GraniteDocling model with custom repetition stopping criteria injected
# - Processes a PDF document or image and monitors for repetitive coordinate patterns
# - Stops generation early when repetitive patterns are detected


# %%

import logging

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.utils.generation_utils import (
    DocTagsRepetitionStopper,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# Set up logging to see when repetition stopping is triggered
logging.basicConfig(level=logging.INFO)

# Replace with a local path if preferred.
# source = "https://ibm.biz/docling-page-with-table" # Example that shows no repetitions.
source = "tests/data_scanned/old_newspaper.png"  # Example that creates repetitions.
print(f"Processing document: {source}")

###### USING GRANITEDOCLING WITH CUSTOM REPETITION STOPPING (LEGACY)

## Using standard Huggingface Transformers (most portable, slowest)
custom_vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.model_copy()

# Uncomment this to use MLX-accelerated version on Apple Silicon
# custom_vlm_options = vlm_model_specs.GRANITEDOCLING_MLX.model_copy() # use this for Apple Silicon


# Create custom VLM options with repetition stopping criteria
custom_vlm_options.custom_stopping_criteria = [
    DocTagsRepetitionStopper(N=32)
]  # check for repetitions for every 32 new tokens decoded.

pipeline_options = VlmPipelineOptions(
    vlm_options=custom_vlm_options,
)

converter = DocumentConverter(
    format_options={
        InputFormat.IMAGE: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())


###### ALTERNATIVE: USING A REMOTE VLM INFERENCE SERVICE (e.g., VLLM) - LEGACY

# from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
#
# custom_vlm_options = ApiVlmOptions(
#     url="http://localhost:8000/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
#     params=dict(
#         model=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.repo_id,
#         max_tokens=8192,
#         seed=42,
#     ),
#     response_format=ResponseFormat.DOCTAGS,
#     headers={
#         # "Authorization": "Bearer YOUR_API_KEY",  # if needed
#     },
#     prompt=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.prompt,
#     timeout=90,
#     # Note: Custom stopping criteria work differently with API runtimes
#     # They are applied client-side after receiving tokens from the API
#     custom_stopping_criteria=[DocTagsRepetitionStopper(N=32)],
# )
#
# pipeline_options = VlmPipelineOptions(
#     vlm_options=custom_vlm_options,
#     enable_remote_services=True, # required when using a remote inference service.
# )
#
# converter = DocumentConverter(
#     format_options={
#         InputFormat.IMAGE: PdfFormatOption(
#             pipeline_cls=VlmPipeline,
#             pipeline_options=pipeline_options,
#         ),
#     }
# )
#
# doc = converter.convert(source=source).document
# print(doc.export_to_markdown())
