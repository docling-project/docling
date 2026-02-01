# %% [markdown]
# Use the VLM pipeline with remote API models (LM Studio, Ollama, VLLM, watsonx.ai).
#
# What this example does
# - Demonstrates using presets with API runtimes (LM Studio, Ollama, VLLM, watsonx.ai)
# - Shows that API is just a runtime choice, not a different options class
# - Explains pre-configured API types and custom API configuration
#
# Prerequisites
# - Install Docling with VLM extras and `python-dotenv` if using environment files.
# - For local APIs: run LM Studio, Ollama, or VLLM locally.
# - For cloud APIs: set required environment variables (see watsonx.ai example).
#
# How to run
# - From the repo root: `python docs/examples/vlm_pipeline_api_model.py`.
# - Each example checks its own prerequisites and skips if not available.
#
# Notes
# - The NEW runtime system unifies API and local inference
# - For legacy approach, see legacy examples in docs/examples/legacy/

# %%

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_runtime_options import (
    ApiVlmRuntimeOptions,
    VlmRuntimeType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def check_and_load_lmstudio_model(model_name: str) -> bool:
    """Check if model is loaded in LM Studio and attempt to load if not.

    Args:
        model_name: The model name to check/load

    Returns:
        True if model is loaded or successfully loaded, False otherwise
    """
    try:
        # Check if model is already loaded
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            models = response.json().get("data", [])
            loaded_models = [m.get("id") for m in models]
            if model_name in loaded_models:
                print(f"✓ Model '{model_name}' is already loaded in LM Studio")
                return True

            # Try to load the model using LM Studio API
            print(f"Attempting to load model '{model_name}' in LM Studio...")

            load_response = requests.post(
                "http://localhost:1234/api/v1/models/load",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_name,
                },
                timeout=60,
            )

            if load_response.status_code == 200:
                print(f"✓ Successfully loaded model '{model_name}'")
                return True
            else:
                print(f"✗ Failed to load model: HTTP {load_response.status_code}")
                print("  Please load the model manually in LM Studio:")
                print(f"    lms load {model_name}")
                return False
        return False
    except requests.exceptions.Timeout:
        print("✗ Timeout while trying to load model")
        return False
    except Exception as e:
        print(f"✗ Error checking/loading model: {e}")
        return False


def check_and_pull_ollama_model(model_name: str) -> bool:
    """Check if model exists in Ollama and attempt to pull if not.

    Args:
        model_name: The model name to check/pull

    Returns:
        True if model exists or successfully pulled, False otherwise
    """
    try:
        # Check if model exists
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            # Check for exact match or with :latest tag
            if model_name in model_names or f"{model_name}:latest" in model_names:
                print(f"✓ Model '{model_name}' is already available in Ollama")
                return True

            # Try to pull the model using Ollama API
            print(f"Attempting to pull model '{model_name}' in Ollama...")
            print("This may take a few minutes...")

            # Ollama pull API endpoint
            pull_response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300,
            )

            if pull_response.status_code == 200:
                # Stream the response to show progress
                for line in pull_response.iter_lines():
                    if line:
                        import json

                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if status:
                                print(f"  {status}", end="\r")
                        except json.JSONDecodeError:
                            pass
                print()  # New line after progress
                print(f"✓ Successfully pulled model '{model_name}'")
                return True
            else:
                print(f"✗ Failed to pull model: HTTP {pull_response.status_code}")
                return False
        return False
    except requests.exceptions.Timeout:
        print("✗ Timeout while trying to pull model (this can take a while)")
        print("Please try pulling manually: ollama pull", model_name)
        return False
    except Exception as e:
        print(f"✗ Error checking/pulling model: {e}")
        return False


def run_lmstudio_example(input_doc_path: Path) -> bool:
    """Example 1: Using Granite-Docling preset with LM Studio API runtime.

    Returns:
        True if example ran successfully, False if skipped
    """
    print("=" * 70)
    print("Example 1: Granite-Docling with LM Studio (pre-configured API type)")
    print("=" * 70)
    print("\nPrerequisites:")
    print("- Start LM Studio: lms server start")
    print("- Model will be loaded automatically if not already loaded")
    print("  (or manually: lms load granite-docling-258m-mlx)")
    print()

    # Check if LM Studio is running
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code != 200:
            print("WARNING: LM Studio server not responding correctly")
            print("Skipping LM Studio example.\n")
            return False
    except requests.exceptions.RequestException:
        print("WARNING: LM Studio server not running at http://localhost:1234")
        print("Skipping LM Studio example.\n")
        return False

    # Check and load the model
    # Note: LM Studio uses a different model ID than the HuggingFace repo
    model_name = "granite-docling-258m-mlx"
    if not check_and_load_lmstudio_model(model_name):
        print("Skipping LM Studio example.\n")
        return False

    # Use granite_docling preset with LM Studio API runtime
    # The preset is pre-configured for LM Studio API type
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        runtime_options=ApiVlmRuntimeOptions(
            runtime_type=VlmRuntimeType.API_LMSTUDIO,
            # url is pre-configured for LM Studio (http://localhost:1234/v1/chat/completions)
            # model name is pre-configured from the preset
            timeout=90,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,  # Required for API runtimes
    )

    print("\nOther API types are also pre-configured:")
    print("- VlmRuntimeType.API_OLLAMA: http://localhost:11434/v1/chat/completions")
    print("- VlmRuntimeType.API_OPENAI: https://api.openai.com/v1/chat/completions")
    print("- VlmRuntimeType.API: Generic API endpoint (you specify the URL)")
    print("\nEach preset has pre-configured model names for these API types.\n")

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())
    return True


def run_ollama_example(input_doc_path: Path) -> bool:
    """Example 2: Using Granite-Docling preset with Ollama.

    Returns:
        True if example ran successfully, False if skipped
    """
    print("\n" + "=" * 70)
    print("Example 2: Granite-Docling with Ollama (pre-configured API type)")
    print("=" * 70)
    print("\nPrerequisites:")
    print("- Install Ollama: https://ollama.ai")
    print("- Pull model: ollama pull ibm/granite-docling:258m")
    print()

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("WARNING: Ollama server not responding correctly")
            print("Skipping Ollama example.\n")
            return False
    except requests.exceptions.RequestException:
        print("WARNING: Ollama server not running at http://localhost:11434")
        print("Skipping Ollama example.\n")
        return False

    # Check and pull the model
    model_name = "ibm/granite-docling:258m"
    if not check_and_pull_ollama_model(model_name):
        print("Skipping Ollama example.\n")
        return False

    # Use granite_docling preset with Ollama API runtime
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        runtime_options=ApiVlmRuntimeOptions(
            runtime_type=VlmRuntimeType.API_OLLAMA,
            # url is pre-configured for Ollama (http://localhost:11434/v1/chat/completions)
            # model name is pre-configured from the preset
            timeout=90,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())
    return True


def run_vllm_example(input_doc_path: Path) -> bool:
    """Example 3: Using Granite-Docling preset with VLLM server.

    Returns:
        True if example ran successfully, False if skipped
    """
    print("\n" + "=" * 70)
    print("Example 3: Granite-Docling with VLLM (generic API configuration)")
    print("=" * 70)
    print("\nPrerequisites:")
    print("- Start VLLM server:")
    print("  vllm serve ibm-granite/granite-docling-258M --revision untied")
    print()

    # Check if VLLM is running
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        if response.status_code != 200:
            print("WARNING: VLLM server not responding correctly")
            print("Skipping VLLM example.\n")
            return False
    except requests.exceptions.RequestException:
        print("WARNING: VLLM server not running at http://localhost:8000")
        print("Skipping VLLM example.\n")
        return False

    # Use granite_docling preset with generic API runtime
    # For VLLM, we need to provide custom URL and params
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        runtime_options=ApiVlmRuntimeOptions(
            runtime_type=VlmRuntimeType.API,  # Generic API type
            url="http://localhost:8000/v1/chat/completions",
            params={
                "model": "ibm-granite/granite-docling-258M",
                "max_tokens": 4096,
                "skip_special_tokens": True,
            },
            timeout=90,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())
    return True


def run_watsonx_example(input_doc_path: Path) -> bool:
    """Example 4: Using preset with watsonx.ai (custom API configuration).

    Returns:
        True if example ran successfully, False if skipped
    """
    print("\n" + "=" * 70)
    print("Example 4: Granite-Docling with watsonx.ai (custom API configuration)")
    print("=" * 70)

    # Check if running in CI environment
    if os.environ.get("CI"):
        print("Skipping watsonx.ai example in CI environment")
        return False

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
        return False

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        return res.json()["access_token"]

    print("\nNote: Granite-Docling models are not currently available on watsonx.ai")
    print("Using Llama 3.2 Vision model instead")
    print("The preset still provides the prompt and response format configuration\n")

    # Use granite_docling preset but override the model for watsonx.ai
    vlm_options = VlmConvertOptions.from_preset(
        "granite_docling",
        runtime_options=ApiVlmRuntimeOptions(
            runtime_type=VlmRuntimeType.API,  # Generic API type
            url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            headers={
                "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
            },
            params={
                "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
                "project_id": project_id,
                "parameters": {"max_new_tokens": 4096},
            },
            timeout=60,
        ),
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())
    return True


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2305.03393v1-pg9.pdf"

    # Track which examples ran
    results = {
        "LM Studio": run_lmstudio_example(input_doc_path),
        "Ollama": run_ollama_example(input_doc_path),
        "VLLM": run_vllm_example(input_doc_path),
        "watsonx.ai": run_watsonx_example(input_doc_path),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ran = [name for name, success in results.items() if success]
    skipped = [name for name, success in results.items() if not success]

    if ran:
        print(f"\n✓ Examples that ran successfully ({len(ran)}):")
        for name in ran:
            print(f"  - {name}")

    if skipped:
        print(f"\n⊘ Examples that were skipped ({len(skipped)}):")
        for name in skipped:
            reason = "Server not running"
            if name == "watsonx.ai":
                if os.environ.get("CI"):
                    reason = "Running in CI environment"
                else:
                    reason = "Credentials not found (WX_API_KEY, WX_PROJECT_ID)"
            print(f"  - {name}: {reason}")

    print()


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
# - Use `VlmRuntimeType.API` (generic)
# - Provide custom `url`, `headers`, and `params`
# - The preset still provides the base model configuration (prompt, response format)
#
# ### Same Preset, Different Runtime
# You can use the same preset (e.g., "granite_docling") with:
# - Local Transformers runtime (see other examples)
# - Local MLX runtime (macOS)
# - LM Studio API runtime (this example)
# - Ollama API runtime (this example)
# - VLLM API runtime (this example)
# - watsonx.ai API runtime (this example)
# - Any other API endpoint
#
# This makes it easy to develop locally and deploy to production!
#
# ### Available Presets for VlmConvert
# - **granite_docling**: IBM Granite Docling 258M (DocTags format)
# - **smoldocling**: SmolDocling 256M (DocTags format)
# - **deepseek_ocr**: DeepSeek OCR (Markdown format)
# - **granite_vision**: IBM Granite Vision (Markdown format)
# - **pixtral**: Pixtral (Markdown format)
# - **got_ocr**: GOT-OCR (Markdown format)
# - **phi4**: Phi-4 (Markdown format)
# - **qwen**: Qwen (Markdown format)
# - **gemma_12b**: Gemma 12B (Markdown format)
# - **gemma_27b**: Gemma 27B (Markdown format)
# - **dolphin**: Dolphin (Markdown format)

# %%
