# %% [markdown]
# Experimental VLM pipeline with custom repetition stopping criteria.
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

###### USING GRANITEDOCLING WITH CUSTOM REPETITION STOPPING

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

## Using a remote VLM inference service (for example VLLM) - uncomment to use

# custom_vlm_options = ApiVlmOptions(
#     url="http://localhost:8000/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
#     params=dict(
#         model=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.repo_id,
#         max_tokens=8192,
#         skip_special_tokens=True,  # needed for VLLM
#     ),
#     headers={
#         "Authorization": "Bearer YOUR_API_KEY",
#     },
#     prompt=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.prompt,
#     timeout=90,
#     scale=2.0,
#     temperature=0.0,
#     response_format=ResponseFormat.DOCTAGS,
#     custom_stopping_criteria=[
#         DocTagsRepetitionStopper(N=1)
#     ],  # check for repetitions for every new chunk of the response stream
# )


# pipeline_options = VlmPipelineOptions(
#     vlm_options=custom_vlm_options,
#     enable_remote_services=True, # required when using a remote inference service.
# )

# converter = DocumentConverter(
#     format_options={
#         InputFormat.IMAGE: PdfFormatOption(
#             pipeline_cls=VlmPipeline,
#             pipeline_options=pipeline_options,
#         ),
#     }
# )

# doc = converter.convert(source=source).document

# print(doc.export_to_markdown())
