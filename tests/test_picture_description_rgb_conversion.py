"""Tests that PictureDescriptionBaseModel converts non-RGB images to RGB.

When documents contain images with non-RGB modes (e.g. RGBA for PNGs with
transparency, L for grayscale, P for palette/indexed color), the picture
description model must convert them to RGB before calling _annotate_images.
This is required because transformers processors and VLM engines only accept
3-channel RGB input.
"""

from collections.abc import Iterable
from typing import ClassVar, List, Type

from docling_core.types.doc import DoclingDocument, PictureItem
from PIL import Image

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel


class _TestOptions(PictureDescriptionBaseOptions):
    """Minimal concrete options subclass used only in tests."""

    kind: ClassVar[str] = "test"


class _RecordingPictureDescriptionModel(PictureDescriptionBaseModel):
    """Test double that records the image modes it receives.

    Instead of running a real VLM, this spy subclass captures what image
    modes arrive at _annotate_images so tests can assert on them.
    """

    def __init__(self) -> None:
        # Bypass the full parent __init__ (which loads heavy ML models) and
        # set the state attributes directly — a standard unit-testing technique.
        self.enabled = True
        self.options = _TestOptions()
        self.provenance = "test"
        self.received_modes: List[str] = []

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return _TestOptions

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        for image in images:
            self.received_modes.append(image.mode)
            yield "test description"


def _make_element(mode: str, size: tuple = (100, 100)) -> ItemAndImageEnrichmentElement:
    """Create an ItemAndImageEnrichmentElement with a synthetic image.

    The PictureItem has no provenance (prov=[]) so the base model skips the
    page-area-threshold check and processes the image unconditionally.

    self_ref is a JSON-pointer string required by docling-core to identify
    nodes in the document tree (e.g. "#/pictures/0").
    """
    img = Image.new(mode, size)
    item = PictureItem(self_ref="#/pictures/0")
    return ItemAndImageEnrichmentElement(item=item, image=img)


def test_rgba_image_converted_to_rgb() -> None:
    """RGBA images (PNG with alpha channel) must be converted to RGB."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    list(model(doc=doc, element_batch=[_make_element("RGBA")]))
    assert model.received_modes == ["RGB"]


def test_grayscale_image_converted_to_rgb() -> None:
    """Grayscale (L mode) images must be converted to RGB."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    list(model(doc=doc, element_batch=[_make_element("L")]))
    assert model.received_modes == ["RGB"]


def test_palette_image_converted_to_rgb() -> None:
    """Palette/indexed-color (P mode) images must be converted to RGB."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    list(model(doc=doc, element_batch=[_make_element("P")]))
    assert model.received_modes == ["RGB"]


def test_rgb_image_unchanged() -> None:
    """Already-RGB images must pass through without any mode change."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    list(model(doc=doc, element_batch=[_make_element("RGB")]))
    assert model.received_modes == ["RGB"]


def test_mixed_modes_all_converted_to_rgb() -> None:
    """A batch with several different image modes must all arrive as RGB."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    elements = [
        _make_element("RGBA"),
        _make_element("RGB"),
        _make_element("L"),
        _make_element("P"),
    ]
    list(model(doc=doc, element_batch=elements))
    assert model.received_modes == ["RGB", "RGB", "RGB", "RGB"]
