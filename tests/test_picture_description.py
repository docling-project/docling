import logging
from pathlib import Path

import pytest
import requests
from docling_core.types.doc.document import PictureDescriptionData

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configure logging at the top of the file
logging.basicConfig(level=logging.DEBUG)
_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
LOCAL_VISION_MODEL = "ibm-granite/granite-vision-3.2-2b"
# LOCAL_VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
API_VISION_MODEL = "granite3.2-vision:2b"
REMOTE_CHAT_API_URL = "http://localhost:8321/v1/openai/v1/chat/completions"  # for llama-stack OpenAI API interface
DOC_SOURCE = "https://www.allspringglobal.com/globalassets/assets/regulatory/summary-prospectus/emerging-markets-equity-summ.pdf"
PROMPT = (
    "Please describe the image using the text above as additional context. "
    "Additionally, if only the image contains a chart (like bar chat, pie chat, line chat, etc.), "
    "please try to extract a list of data points (percentages, numbers, etc) that are depicted in the chart. "
    "Also, based on the type of information extracted, "
    "when applicable try to summarize it using bullet points or even a tabular representation using markdown if possible."
)


def is_api_available(url: str, timeout: int = 3) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout) as e:
        _log.debug(f"API endpoint {url} is not reachable: {e!s}")
        return False


def process_document(pipeline_options: PdfPipelineOptions):
    # Initialize document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert test document
    _log.info(f"Converting {DOC_SOURCE} with VLM API")
    conversion_result = doc_converter.convert(source=DOC_SOURCE)

    # Basic conversion checks
    assert conversion_result.status == ConversionStatus.SUCCESS
    doc = conversion_result.document
    assert doc is not None

    # Verify pictures were processed
    assert len(doc.pictures) > 0

    # Check each picture for descriptions
    for picture in doc.pictures:
        # Not every picture has a annotations (eg. some pictures are too small (based on the threshold param (5% of the page area by default))
        # and gets ignored by the conversion Pipeline)
        if len(picture.annotations) > 0:
            # Get the description
            descriptions = [
                ann
                for ann in picture.annotations
                if isinstance(ann, PictureDescriptionData)
            ]
            assert len(descriptions) > 0

            # Verify each description is non-empty
            for desc in descriptions:
                assert isinstance(desc.text, str)
                assert len(desc.text) > 0
                _log.info(
                    f"\nPicture ref: {picture.get_ref().cref}, page #{picture.prov[0].page_no}"
                )
                _log.info(f"\tGenerated description: {desc.text}")
        else:
            _log.info(
                f"Picture {picture.get_ref().cref} has no annotations (too small?)"
            )


@pytest.mark.skipif(
    not is_api_available(REMOTE_CHAT_API_URL),
    reason="Remote API endpoint is not accessible",
)
def test_picture_description_context_api_integration():
    """Test that the context windows functionality works correctly in the picture description pipeline using a VLM served via API"""
    # Setup pipeline options with context windows
    pipeline_options = PdfPipelineOptions(
        images_scale=IMAGE_RESOLUTION_SCALE,
        do_picture_description=True,
        generate_picture_images=True,
        enable_remote_services=True,
        picture_description_options=PictureDescriptionApiOptions(
            url=REMOTE_CHAT_API_URL,
            params=dict(model=API_VISION_MODEL),
            text_context_window_size_before_picture=2,  # Get 2 text items before
            text_context_window_size_after_picture=1,  # Get 1 text item after
            prompt=PROMPT,
            timeout=90,
        ),
    )

    process_document(pipeline_options)


def test_picture_description_context_vlm_integration():
    """Test that the context windows functionality works correctly in the picture description pipeline"""
    # Setup pipeline options with context windows
    pipeline_options = PdfPipelineOptions(
        images_scale=IMAGE_RESOLUTION_SCALE,
        generate_page_images=True,
        do_picture_description=True,
        generate_picture_images=True,
        picture_description_options=PictureDescriptionVlmOptions(
            repo_id=LOCAL_VISION_MODEL,
            text_context_window_size_before_picture=2,  # Get 2 text items before
            text_context_window_size_after_picture=1,  # Get 1 text item after
            prompt=PROMPT,
        ),
    )

    process_document(pipeline_options)


def test_picture_description_no_context_vlm_integration():
    """Test that the picture description works without context windows"""
    # Setup pipeline options without context windows
    pipeline_options = PdfPipelineOptions(
        images_scale=IMAGE_RESOLUTION_SCALE,
        do_picture_description=True,
        generate_picture_images=True,
        picture_description_options=PictureDescriptionVlmOptions(
            repo_id=LOCAL_VISION_MODEL,
            text_context_window_size_before_picture=0,  # No text context
            text_context_window_size_after_picture=0,  # No text context
            prompt=PROMPT,
        ),
    )

    process_document(pipeline_options)
