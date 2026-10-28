# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Page backends stay alive for enrichment only so later crops can be rendered.

Issue #1886: formula, code, picture, and chart enrichment set ``keep_backend``.
The page backend is then kept until the conversion ends, and memory grows with
the page count. The crops only need a cached page image.
"""

from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

from docling_core.types.doc import BoundingBox, DoclingDocument, NodeItem, Size
from PIL import Image

from docling.datamodel.base_models import ItemAndImageEnrichmentElement, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import (
    ProcessingResult,
    StandardPdfPipeline,
    ThreadedItem,
)


class _PageBackend:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.unloaded = False
        self.unload_calls = 0

    def is_valid(self) -> bool:
        return not self.unloaded

    def unload(self) -> None:
        self.unload_calls += 1
        self.unloaded = True

    def get_page_image(
        self, scale: float = 1.0, cropbox: BoundingBox | None = None
    ) -> Image.Image:
        image = Image.new(
            "RGB", (round(self.width * scale), round(self.height * scale)), "white"
        )
        image.putpixel((0, 0), (255, 0, 0))
        if cropbox is None:
            return image
        box = (
            cropbox.to_top_left_origin(page_height=self.height)
            .scaled(scale=scale)
            .as_tuple()
        )
        return image.crop(box)


class _ImageEnrichment(BaseItemAndImageEnrichmentModel):
    def __init__(self, scale: float, *, enabled: bool = True) -> None:
        self.images_scale = scale
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return False

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        return ()


def _pipeline(*, release_after_cache: bool, keep_backend: bool) -> StandardPdfPipeline:
    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline.keep_images = False
    pipeline.keep_backend = keep_backend
    pipeline._release_backend_after_image_cache = release_after_cache
    pipeline.enrichment_pipe = [_ImageEnrichment(1.67)]
    pipeline.pipeline_options = SimpleNamespace(generate_parsed_pages=True)
    return pipeline


def _page() -> tuple[Page, _PageBackend]:
    backend = _PageBackend(100, 80)
    page = Page(page_no=1, size=Size(width=100, height=80))
    page._backend = backend
    page._image_cache[1.0] = Image.new("RGB", (100, 80), "blue")
    return page, backend


def test_cached_page_image_is_cropped_without_a_backend() -> None:
    page, backend = _page()
    page._backend = backend
    cropbox = BoundingBox(l=10, t=5, r=40, b=25)
    page.get_image(scale=2.0)
    expected = page.get_image(scale=2.0, cropbox=cropbox)
    assert expected is not None

    page._backend = None
    cropped = page.get_image(scale=2.0, cropbox=cropbox)

    assert cropped is not None
    assert cropped.size == expected.size
    assert cropped.getpixel((0, 0)) == (255, 255, 255)


def test_enrichment_releases_the_page_backend_after_caching_its_scale() -> None:
    pipeline = _pipeline(release_after_cache=True, keep_backend=True)
    page, backend = _page()
    item = ThreadedItem(
        payload=page,
        run_id=1,
        page_no=1,
        conv_res=cast(ConversionResult, SimpleNamespace()),
    )

    pipeline._release_page_resources(item)

    assert backend.unloaded
    assert page._backend is None
    assert 1.0 not in page._image_cache
    assert 1.67 in page._image_cache
    cropbox = BoundingBox(l=10, t=5, r=40, b=25)
    cropped = page.get_image(scale=1.67, cropbox=cropbox)
    assert cropped is not None
    assert cropped.size != page._image_cache[1.67].size
    assert cropped.getpixel((0, 0)) == (255, 255, 255)

    pipeline._integrate_results(
        cast(
            ConversionResult,
            SimpleNamespace(pages=[page], errors=[], status=None),
        ),
        ProcessingResult(pages=[page], total_expected=1),
    )
    assert 1.67 in page._image_cache
    still_cropped = page.get_image(scale=1.67, cropbox=cropbox)
    assert still_cropped is not None
    assert still_cropped.size == cropped.size


def test_invalid_page_backend_is_dropped_without_unload() -> None:
    pipeline = _pipeline(release_after_cache=True, keep_backend=True)
    page, backend = _page()
    backend.unloaded = True
    item = ThreadedItem(
        payload=page,
        run_id=1,
        page_no=1,
        conv_res=cast(ConversionResult, SimpleNamespace()),
    )

    pipeline._release_page_resources(item)

    assert backend.unload_calls == 0
    assert page._backend is None


def test_explicit_keep_backend_is_left_alive() -> None:
    pipeline = _pipeline(release_after_cache=False, keep_backend=True)
    page, backend = _page()
    item = ThreadedItem(
        payload=page,
        run_id=1,
        page_no=1,
        conv_res=cast(ConversionResult, SimpleNamespace()),
    )

    pipeline._release_page_resources(item)

    assert backend.unloaded is False
    assert page._backend is backend
    assert page._image_cache == {}
