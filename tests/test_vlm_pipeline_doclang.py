# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""DocLang responses in VlmPipeline are mapped from the DocLang grid to the page."""

from pathlib import PurePath
from types import SimpleNamespace

import pytest
from docling_core.types.doc import CoordOrigin, PictureItem, Size, TextItem
from PIL import Image as PILImage

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

_PAGE = Size(width=612, height=792)

# A picture over the bottom-right quarter and a text line at the top-left.
_DOCLANG = (
    '<doclang version="0.7">'
    "<picture>"
    '<location value="256"/><location value="256"/>'
    '<location value="512"/><location value="512"/>'
    "</picture>"
    "<text>"
    '<location value="0"/><location value="0"/>'
    '<location value="128"/><location value="64"/>'
    "Hello</text>"
    "</doclang>"
)


def _pipeline(images_scale: float) -> VlmPipeline:
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        generate_page_images=False,
        generate_picture_images=True,
        images_scale=images_scale,
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="test",
            response_format=ResponseFormat.DOCLANG,
            inference_framework=InferenceFramework.TRANSFORMERS,
            scale=images_scale,
        ),
    )
    pipeline.force_backend_text = False
    return pipeline


def _page(images_scale: float) -> Page:
    page = Page(
        page_no=1,
        size=_PAGE,
        predictions=PagePredictions(vlm_response=VlmPrediction(text=_DOCLANG)),
    )
    image = PILImage.new(
        "RGB",
        (round(_PAGE.width * images_scale), round(_PAGE.height * images_scale)),
        "white",
    )
    page._image_cache = {images_scale: image}
    page._default_image_scale = images_scale
    return page


@pytest.mark.parametrize("images_scale", [1.0, 2.0])
def test_doclang_locations_are_scaled_to_the_page(images_scale: float) -> None:
    pipeline = _pipeline(images_scale)
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")), errors=[]
    )
    page = _page(images_scale)
    assert page.image is not None

    doc = pipeline._finalize_page_document(conv_res, page)

    assert not conv_res.errors
    assert doc.pages[1].size == _PAGE

    picture = doc.pictures[0]
    assert isinstance(picture, PictureItem)
    bbox = picture.prov[0].bbox
    assert bbox.coord_origin == CoordOrigin.TOPLEFT
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx((306, 396, 612, 792))

    # The crop is the bottom-right quarter of the page image.
    assert picture.image is not None
    assert picture.image.size == Size(
        width=306 * images_scale, height=396 * images_scale
    )

    text = next(item for item in doc.texts if isinstance(item, TextItem))
    bbox = text.prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx((0, 0, 153, 99))
