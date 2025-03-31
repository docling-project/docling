import logging
import os
from pathlib import Path

import requests
from docling_core.types.doc import PictureItem
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def vllm_local_options(model: str):
    options = PictureDescriptionApiOptions(
        url="http://localhost:8000/v1/chat/completions",
        params=dict(
            model=model,
            seed=42,
            max_completion_tokens=200,
        ),
        prompt="Describe the image in three sentences. Be consise and accurate.",
        timeout=90,
    )
    return options


def watsonx_vlm_options():
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

    options = PictureDescriptionApiOptions(
        url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
        params=dict(
            model_id="meta-llama/llama-3-2-11b-vision-instruct",
            project_id=project_id,
            parameters=dict(
                max_new_tokens=400,
            ),
        ),
        headers={
            "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
        },
        prompt="Describe the image in three sentences. Be consise and accurate.",
        timeout=60,
    )
    return options


def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path("./tests/data/pdf/2206.01062.pdf")

    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )
    pipeline_options.do_picture_description = True

    # The PictureDescriptionApiOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.
    #
    # One possibility is self-hosting model, e.g. via VLLM.
    # $ vllm serve MODEL_NAME
    # Then PictureDescriptionApiOptions can point to the localhost endpoint.
    #
    # Example for the Granite Vision model: (uncomment the following lines)
    # pipeline_options.picture_description_options = vllm_local_options(
    #     model="ibm-granite/granite-vision-3.1-2b-preview"
    # )
    #
    # Example for the SmolVLM model: (uncomment the following lines)
    pipeline_options.picture_description_options = vllm_local_options(
        model="HuggingFaceTB/SmolVLM-256M-Instruct"
    )
    #
    # Another possibility is using online services, e.g. watsonx.ai.
    # Using requires setting the env variables WX_API_KEY and WX_PROJECT_ID.
    # Uncomment the following line for this option:
    # pipeline_options.picture_description_options = watsonx_vlm_options()

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
                f"Annotations: {element.annotations}"
            )


if __name__ == "__main__":
    main()
