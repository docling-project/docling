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

#### Using LM Studio


def lms_vlm_options(model: str, prompt: str, format: ResponseFormat):
    options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",  # the default LM Studio
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=format,
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


#### Using Ollama with OlmOcr


def ollama_olmocr_vlm_options(model: str):
    def _dynamic_olmocr_prompt(page: Optional[SegmentedPage]):
        if page is None:
            return (
                "Below is the image of one page of a document. Just return the plain text"
                " representation of this document as if you were reading it naturally.\n"
                "Do not hallucinate.\n"
            )

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

        # Original prompt uses cells sorting. We are skipping it in this demo.

        base_text = "\n".join(anchor)

        return (
            f"Below is the image of one page of a document, as well as some raw textual"
            f" content that was previously extracted for it. Just return the plain text"
            f" representation of this document as if you were reading it naturally.\n"
            f"Do not hallucinate.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
        )

    options = ApiVlmOptions(
        url="http://localhost:11434/v1/chat/completions",  # the default Ollama endpoint
        params=dict(
            model=model,
        ),
        prompt=_dynamic_olmocr_prompt,
        timeout=90,
        scale=1.0,
        max_size=1024,  # from OlmOcr pipeline
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

    # input_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    input_doc_path = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )

    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.

    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.

    # Example using the SmolDocling model with LM Studio:
    # (uncomment the following lines)
    pipeline_options.vlm_options = lms_vlm_options(
        model="smoldocling-256m-preview-mlx-docling-snap",
        prompt="Convert this page to docling.",
        format=ResponseFormat.DOCTAGS,
    )

    # Example using the Granite Vision model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_vlm_options(
    #     model="granite-vision-3.2-2b",
    #     prompt="OCR the full page to markdown.",
    #     format=ResponseFormat.MARKDOWN,
    # )

    # Example using the Granite Vision model with Ollama:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = ollama_vlm_options(
    #     model="granite3.2-vision:2b",
    #     prompt="OCR the full page to markdown.",
    # )

    # Example using the OlmOcr (dynamic prompt) model with Ollama:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = ollama_olmocr_vlm_options(
    #     model="hf.co/lmstudio-community/olmOCR-7B-0225-preview-GGUF:Q8_0",
    # )

    # Another possibility is using online services, e.g. watsonx.ai.
    # Using requires setting the env variables WX_API_KEY and WX_PROJECT_ID.
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
