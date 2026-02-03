# %% [markdown]
# Use the VLM pipeline with remote API models (LM Studio, Ollama, watsonx.ai).
#
# What this example does
# - Shows how to configure `ApiVlmOptions` for different VLM providers.
# - Converts a single PDF page using the VLM pipeline and prints Markdown.
#
# Prerequisites
# - Install Docling with VLM extras and `python-dotenv` if using environment files.
# - For local APIs: run LM Studio (HTTP server) or Ollama locally.
# - For cloud APIs: set required environment variables (see below).
# - Requires `requests` for HTTP calls and `python-dotenv` if loading env vars from `.env`.
#
# How to run
# - From the repo root: `python docs/examples/vlm_pipeline_api_model.py`.
# - The script prints the converted Markdown to stdout.
#
# Choosing a provider
# - Uncomment exactly one `pipeline_options.vlm_options = ...` block below.
# - Keep `enable_remote_services=True` to permit calling remote APIs.
#
# Notes
# - LM Studio default endpoint: `http://localhost:1234/v1/chat/completions`.
# - Ollama default endpoint: `http://localhost:11434/v1/chat/completions`.
# - watsonx.ai requires `WX_API_KEY` and `WX_PROJECT_ID` in env/`.env`.

# %%

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests
from docling_core.types.doc.page import SegmentedPage
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

### Example of ApiVlmOptions definitions

#### Using LM Studio or VLLM (OpenAI-compatible APIs)


def openai_compatible_vlm_options(
    model: str,
    prompt: str,
    format: ResponseFormat,
    hostname_and_port,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: str = "",
    skip_special_tokens=False,
):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    options = ApiVlmOptions(
        url=f"http://{hostname_and_port}/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
        params=dict(
            model=model,
            max_tokens=max_tokens,
            skip_special_tokens=skip_special_tokens,  # needed for VLLM
        ),
        headers=headers,
        prompt=prompt,
        timeout=90,
        scale=2.0,
        temperature=temperature,
        response_format=format,
    )
    return options


#### Using LM Studio with OlmOcr model


def lms_olmocr_vlm_options(model: str):
    class OlmocrVlmOptions(ApiVlmOptions):
        def build_prompt(self, page: SegmentedPage | None) -> str:
            if page is None:
                return self.prompt.replace("#RAW_TEXT#", "")

            anchor = [
                f"Page dimensions: {int(page.dimension.width)}x{int(page.dimension.height)}"
            ]

            for text_cell in page.textline_cells:
                if not text_cell.text.strip():
                    continue
                bbox = text_cell.rect.to_bounding_box().to_bottom_left_origin(
                    page.dimension.height
                )
                anchor.append(f"[{int(bbox.l)}x{int(bbox.b)}] {text_cell.text}")

            for image_cell in page.bitmap_resources:
                bbox = image_cell.rect.to_bounding_box().to_bottom_left_origin(
                    page.dimension.height
                )
                anchor.append(
                    f"[Image {int(bbox.l)}x{int(bbox.b)} to {int(bbox.r)}x{int(bbox.t)}]"
                )

            if len(anchor) == 1:
                anchor.append(
                    f"[Image 0x0 to {int(page.dimension.width)}x{int(page.dimension.height)}]"
                )

            # Original prompt uses cells sorting. We are skipping it for simplicity.

            raw_text = "\n".join(anchor)

            return self.prompt.replace("#RAW_TEXT#", raw_text)

        def decode_response(self, text: str) -> str:
            # OlmOcr trained to generate json response with language, rotation and other info
            try:
                generated_json = json.loads(text)
            except json.decoder.JSONDecodeError:
                return ""

            return generated_json["natural_text"]

    options = OlmocrVlmOptions(
        url="http://localhost:1234/v1/chat/completions",
        params=dict(
            model=model,
        ),
        prompt=(
            "Below is the image of one page of a document, as well as some raw textual"
            " content that was previously extracted for it. Just return the plain text"
            " representation of this document as if you were reading it naturally.\n"
            "Do not hallucinate.\n"
            "RAW_TEXT_START\n#RAW_TEXT#\nRAW_TEXT_END"
        ),
        timeout=90,
        scale=1.0,
        max_size=1024,  # from OlmOcr pipeline
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


#### Using Ollama


def ollama_vlm_options(model: str, prompt: str):
    options = ApiVlmOptions(
        url="http://localhost:11434/v1/chat/completions",  # the default Ollama endpoint
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


#### Using a cloud service like IBM watsonx.ai


def watsonx_vlm_options(model: str, prompt: str):
    load_dotenv()
    api_key = os.environ.get("WX_API_KEY")
    project_id = os.environ.get("WX_PROJECT_ID")

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        api_out = res.json()
        print(f"{api_out=}")
        return api_out["access_token"]

    options = ApiVlmOptions(
        url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
        params=dict(
            model_id=model,
            project_id=project_id,
            parameters=dict(
                max_new_tokens=400,
            ),
        ),
        headers={
            "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
        },
        prompt=prompt,
        timeout=60,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


### Usage and conversion


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2305.03393v1-pg9.pdf"

    # Configure the VLM pipeline. Enabling remote services allows HTTP calls to
    # locally hosted APIs (LM Studio, Ollama) or cloud services.
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # required when calling remote VLM endpoints
    )

    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.

    # One possibility is self-hosting the model, e.g., via LM Studio, Ollama or VLLM.
    #
    # e.g. with VLLM, serve granite-docling with these commands:
    # > vllm serve ibm-granite/granite-docling-258M --revision untied
    #
    # with LM Studio, serve granite-docling with these commands:
    # > lms server start
    # > lms load ibm-granite/granite-docling-258M-mlx

    # Example using the Granite-Docling model with LM Studio or VLLM:
    pipeline_options.vlm_options = openai_compatible_vlm_options(
        model="granite-docling-258m-mlx",  # For VLLM use "ibm-granite/granite-docling-258M"
        hostname_and_port="localhost:1234",  # LM studio defaults to port 1234, VLLM to 8000
        prompt="Convert this page to docling.",
        format=ResponseFormat.DOCTAGS,
        api_key="",
    )

    # Example using the OlmOcr (dynamic prompt) model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_olmocr_vlm_options(
    #     model="hf.co/lmstudio-community/olmOCR-7B-0225-preview-GGUF",
    # )

    # Example using the Granite Vision model with Ollama:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = ollama_vlm_options(
    #     model="granite3.2-vision:2b",
    #     prompt="OCR the full page to markdown.",
    # )

    # Another possibility is using online services, e.g., watsonx.ai.
    # Using watsonx.ai requires setting env variables WX_API_KEY and WX_PROJECT_ID
    # (see the top-level docstring for details). You can use a .env file as well.
    # (uncomment the following lines)
    # pipeline_options.vlm_options = watsonx_vlm_options(
    #     model="ibm/granite-vision-3-2-2b", prompt="OCR the full page to markdown."
    # )

    # Create the DocumentConverter and launch the conversion.
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


if __name__ == "__main__":
    main()

# %%
