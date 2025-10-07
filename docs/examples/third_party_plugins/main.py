"""
Example: Docling pipeline using the third-party picture description plugin
located in docs/examples/third_party_plugins/api_usage.

Prerequisites:
- Ensure you have Docling installed in the same Python environment
- Install this example plugin in editable mode:
    pip install -e docs/examples/third_party_plugins
- Optionally set environment variables for the API backend:
    OPENAI_COMPATIBLE_API_URL, OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_API_HEADER_NAME
    (or provide url/headers directly below)

Run:
    python docs/examples/third_party_plugins/main.py
"""

import os
from typing import Dict

# Import the options class from the installed example plugin package
from api_usage.datamodel.pipeline_options.picture_description_api_model_with_usage import (
    PictureDescriptionApiOptionsWithUsage,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    # Resolve a simple OpenAI-compatible backend from environment variables
    url = os.getenv(
        "OPENAI_COMPATIBLE_API_URL", "http://localhost:8000/v1/chat/completions"
    )
    key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    header_name = os.getenv("OPENAI_COMPATIBLE_API_HEADER_NAME", "api-key")
    headers: Dict[str, str] = {header_name: key} if key else {}

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.allow_external_plugins = True

    # Enable image processing for paginated PDF processing
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2  # higher resolution thumbnails
    pipeline_options.do_picture_description = True

    # Enable remote services (required for external API calls)
    pipeline_options.enable_remote_services = True

    # Configure picture description via the example plugin options
    pipeline_options.picture_description_options = (
        PictureDescriptionApiOptionsWithUsage(
            url=url,
            headers=headers,
            params={"model": "gpt-4o-mini", "temperature": 0},
            prompt="Describe the image clearly and concisely in a few sentences.",
            timeout=45.0,
            concurrency=2,
            # If your server returns token usage in a dict under 'usage', you can
            # extract a specific field and make it the generated text:
            token_extract_key="usage",
        )
    )

    # Create converter with the configured options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document (local path or URL)
    source = os.getenv("SOURCE_DOCUMENT", "https://arxiv.org/pdf/2408.09869")
    print(f"\nConverting source: {source}\n")

    result = converter.convert(source)
    doc = result.document

    # Print the markdown result
    print(doc.export_to_markdown())

    # Print token usage for each picture annotation (if provided by backend)
    for idx, pic in enumerate(doc.pictures):
        print(f"\nPicture #{idx}:")
        if not getattr(pic, "annotations", None):
            print("  (no annotations)")
            continue

        for ann_idx, ann in enumerate(pic.annotations):
            token_usage = getattr(ann, "token_usage", None)
            ann_text = getattr(ann, "text", None)
            print(
                f"  Annotation {ann_idx}: text={ann_text!r} token_usage={token_usage!r}"
            )


if __name__ == "__main__":
    main()
