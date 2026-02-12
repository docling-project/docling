# %% [markdown]
# Describe pictures using VLM models via API runtimes
#
# What this example does
# - Demonstrates using presets with API runtimes (LM Studio, watsonx.ai)
# - Shows that API is just a runtime choice, not a different options class
# - Explains pre-configured API types and custom API configuration
#
# Prerequisites
# - Install Docling and `python-dotenv` if loading env vars from a `.env` file.
# - For LM Studio: ensure LM Studio is running with a VLM model loaded
# - For watsonx.ai: set `WX_API_KEY` and `WX_PROJECT_ID` in the environment.
#
# How to run
# - From the repo root: `python docs/examples/pictures_description_api.py`.
# - watsonx.ai example runs automatically if credentials are available
#
# Notes
# - The NEW runtime system unifies API and local inference
# - For legacy approach, see `pictures_description_api_legacy.py`

# %%

import logging
import os
from pathlib import Path

import requests
from docling_core.types.doc import PictureItem
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmEngineOptions,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    VlmEngineType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def run_lm_studio_example(input_doc_path: Path):
    """Example 1: Using Granite Vision preset with LM Studio API runtime."""
    print("=" * 70)
    print("Example 1: Granite Vision with LM Studio (pre-configured API type)")
    print("=" * 70)

    # Start LM Studio with granite-vision model loaded
    # The preset is pre-configured for LM Studio API type
    picture_desc_options = PictureDescriptionVlmEngineOptions.from_preset(
        "granite_vision",
        engine_options=ApiVlmEngineOptions(
            runtime_type=VlmEngineType.API_LMSTUDIO,
            # url is pre-configured for LM Studio (http://localhost:1234/v1/chat/completions)
            # model name is pre-configured from the preset
            timeout=90,
        ),
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = picture_desc_options
    pipeline_options.enable_remote_services = True  # Required for API runtimes

    print("\nOther API types are also pre-configured:")
    print("- VlmEngineType.API_OLLAMA: http://localhost:11434/v1/chat/completions")
    print("- VlmEngineType.API_OPENAI: https://api.openai.com/v1/chat/completions")
    print("- VlmEngineType.API: Generic API endpoint (you specify the URL)")
    print("\nEach preset has pre-configured model names for these API types.")
    print("For example, granite_vision preset knows:")
    print('- Ollama model name: "ibm/granite3.3-vision:2b"')
    print('- LM Studio model name: "granite-vision-3.3-2b"')
    print("- OpenAI model name: would use the HuggingFace repo_id\n")

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert(input_doc_path)

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"Picture {element.self_ref}\n"
                f"Caption: {element.caption_text(doc=result.document)}\n"
                f"Meta: {element.meta}\n"
            )


def run_watsonx_example(input_doc_path: Path):
    """Example 2: Using Granite Vision preset with watsonx.ai."""
    print("\n" + "=" * 70)
    print("Example 2: Granite Vision with watsonx.ai (custom API configuration)")
    print("=" * 70)

    # Check if running in CI environment
    if os.environ.get("CI"):
        print("Skipping watsonx.ai example in CI environment")
        return

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("WX_API_KEY")
    project_id = os.environ.get("WX_PROJECT_ID")

    # Check if credentials are available
    if not api_key or not project_id:
        print("WARNING: watsonx.ai credentials not found.")
        print(
            "Set WX_API_KEY and WX_PROJECT_ID environment variables to run this example."
        )
        print("Skipping watsonx.ai example.\n")
        return

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        return res.json()["access_token"]

    # For watsonx.ai, we need to provide custom URL, headers, and params
    picture_desc_options = PictureDescriptionVlmEngineOptions.from_preset(
        "granite_vision",
        engine_options=ApiVlmEngineOptions(
            runtime_type=VlmEngineType.API,  # Generic API type
            url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            headers={
                "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
            },
            params={
                # Note: Granite Vision models are no longer available on watsonx.ai (they are model on demand)
                # "model_id": "ibm/granite-vision-3-3-2b",
                "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
                "project_id": project_id,
                "parameters": {"max_new_tokens": 400},
            },
            timeout=60,
        ),
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = picture_desc_options
    pipeline_options.enable_remote_services = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert(input_doc_path)

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"Picture {element.self_ref}\n"
                f"Caption: {element.caption_text(doc=result.document)}\n"
                f"Meta: {element.meta}\n"
            )


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    # Run LM Studio example
    run_lm_studio_example(input_doc_path)

    # Run watsonx.ai example (skips if in CI or credentials not found)
    run_watsonx_example(input_doc_path)


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Key Concepts
#
# ### Pre-configured API Types
# The new runtime system has pre-configured API types:
# - **API_OLLAMA**: Ollama server (port 11434)
# - **API_LMSTUDIO**: LM Studio server (port 1234)
# - **API_OPENAI**: OpenAI API
# - **API**: Generic API endpoint (you provide URL)
#
# Each preset knows the appropriate model names for these API types.
#
# ### Custom API Configuration
# For services like watsonx.ai that need custom configuration:
# - Use `VlmEngineType.API` (generic)
# - Provide custom `url`, `headers`, and `params`
# - The preset still provides the base model configuration
#
# ### Same Preset, Different Runtime
# You can use the same preset (e.g., "granite_vision") with:
# - Local Transformers runtime (see `picture_description_inline.py`)
# - Local MLX runtime (macOS)
# - LM Studio API runtime (this example)
# - watsonx.ai API runtime (this example)
# - Any other API endpoint
#
# This makes it easy to develop locally and deploy to production!
