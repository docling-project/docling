# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for picture crops produced by VLM page finalization.

Doctags bounding boxes are expressed in the pixels of the page image, which is
rendered at images_scale. The crop must not scale the bbox a second time.
"""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc.base import Size
from PIL import Image as PILImage, ImageDraw

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.ml_vlm

IMAGE_SIZE = (200, 400)
RED_BOX = (40, 80, 120, 160)


@pytest.fixture
def pipeline() -> VlmPipeline:
    """VlmPipeline instance without running __init__ (no model download)."""
    pipe = VlmPipeline.__new__(VlmPipeline)
    pipe.force_backend_text = False
    pipe.pipeline_options = MagicMock()
    pipe.pipeline_options.generate_page_images = False
    pipe.pipeline_options.generate_picture_images = True
    return pipe


@pytest.mark.parametrize("images_scale", [1.0, 2.0])
def test_picture_crop_matches_bbox_in_page_image(
    pipeline: VlmPipeline, images_scale: float
) -> None:
    width, height = IMAGE_SIZE
    page_image = PILImage.new("RGB", (width, height), "white")
    ImageDraw.Draw(page_image).rectangle(
        (RED_BOX[0], RED_BOX[1], RED_BOX[2] - 1, RED_BOX[3] - 1), fill="red"
    )

    page = Page(page_no=1)
    page.size = Size(width=width / images_scale, height=height / images_scale)
    page.predictions = PagePredictions(
        vlm_response=VlmPrediction(
            text="<doctag><picture><loc_100><loc_100><loc_300><loc_200></picture></doctag>"
        )
    )
    page._image_cache = {images_scale: page_image}
    page._default_image_scale = images_scale
    pipeline.pipeline_options.images_scale = images_scale

    document = pipeline._doctags_page_document(
        page.predictions.vlm_response.text, page_image
    )
    pipeline._finalize_page_output(document, page)

    crop = document.pictures[0].image.pil_image
    assert crop.size == (80, 80)
    assert crop.getcolors() == [(80 * 80, (255, 0, 0))]
