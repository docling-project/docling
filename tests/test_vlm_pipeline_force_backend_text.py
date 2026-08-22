"""Tests for per-page VLM finalization with force_backend_text.

When force_backend_text=True on a multi-page DocTags (SmolDocling) document,
each text element must be re-extracted from its OWN page's backend, using that
page's size. A regression made every element use the last page's backend and
height, so all pages received the final page's text.

Related: https://github.com/docling-project/docling/pull/1371
"""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import TextItem
from docling_core.types.doc.base import Size
from PIL import Image as PILImage

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.ml_vlm


def _make_page(page_no: int, height: int, backend_text: str) -> tuple[Page, MagicMock]:
    """Build a Page whose backend returns a page-specific string for any rect."""
    page = Page(page_no=page_no)
    page.size = Size(width=100, height=height)
    page.predictions = PagePredictions(
        vlm_response=VlmPrediction(
            text=(
                "<doctag><text><loc_10><loc_10><loc_90><loc_20>"
                f"model text {page_no}</text></doctag>"
            )
        )
    )
    page._image_cache = {1.0: PILImage.new("RGB", (100, height), "white")}

    backend = MagicMock()
    backend.get_text_in_rect.return_value = backend_text
    page._backend = backend
    return page, backend


@pytest.fixture
def pipeline() -> VlmPipeline:
    """VlmPipeline instance without running __init__ (no model download)."""
    pipe = VlmPipeline.__new__(VlmPipeline)
    pipe.force_backend_text = True
    pipe.pipeline_options = MagicMock()
    pipe.pipeline_options.images_scale = 1.0
    pipe.pipeline_options.generate_page_images = False
    pipe.pipeline_options.generate_picture_images = False
    return pipe


def test_finalize_page_output_uses_each_pages_backend(
    pipeline: VlmPipeline,
) -> None:
    """Each page's text must come from that page's backend, not the last page's."""
    page1, backend1 = _make_page(1, height=200, backend_text="backend text page 1")
    page2, backend2 = _make_page(2, height=400, backend_text="backend text page 2")

    texts = []
    for page in (page1, page2):
        response = page.predictions.vlm_response
        image = page.image
        assert response is not None
        assert image is not None
        document = pipeline._doctags_page_document(response.text, image)
        pipeline._finalize_page_output(document, page)
        texts.extend(
            item.text
            for item, _level in document.iterate_items()
            if isinstance(item, TextItem)
        )

    assert texts == ["backend text page 1", "backend text page 2"]

    # The bug routed every element through the last page's backend, so page 1's
    # backend was never queried while page 2's was queried for both elements.
    assert backend1.get_text_in_rect.call_count == 1
    assert backend2.get_text_in_rect.call_count == 1
