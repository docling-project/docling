# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test that PictureDescriptionBaseModel converts non-RGB images to RGB."""

from collections.abc import Iterable
from typing import ClassVar, List, Type

import pytest
from docling_core.types.doc import DoclingDocument, PictureItem
from PIL import Image

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel

pytestmark = pytest.mark.ml_vlm


class _TestOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[str] = "test"


class _RecordingPictureDescriptionModel(PictureDescriptionBaseModel):
    """Spy subclass that records image modes arriving at _annotate_images."""

    def __init__(self) -> None:
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


def _make_element(mode: str) -> ItemAndImageEnrichmentElement:
    img = Image.new(mode, (100, 100))
    item = PictureItem(self_ref="#/pictures/0")
    return ItemAndImageEnrichmentElement(item=item, image=img)


def test_rgba_image_converted_to_rgb() -> None:
    """RGBA images must be converted to RGB before picture description."""
    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")
    list(model(doc=doc, element_batch=[_make_element("RGBA")]))
    assert model.received_modes == ["RGB"]


def test_normalize_image_palette_trns_no_warning() -> None:
    """normalize_image_to_pil converts P+tRNS without PIL warning."""
    import warnings

    from docling.models.inference_engines.vlm._utils import normalize_image_to_pil

    img = Image.new("P", (100, 100))
    img.info["transparency"] = bytes([0] * 256)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_image_to_pil(img)
        palette_warnings = [x for x in w if "Palette images" in str(x.message)]
        assert result.mode == "RGB"
        assert len(palette_warnings) == 0, (
            f"PIL palette transparency warning should not fire: {palette_warnings}"
        )


def test_palette_trns_image_converted_without_warning() -> None:
    """Palette (P) images with per-entry tRNS must not trigger PIL UserWarning."""
    import warnings

    img = Image.new("P", (100, 100))
    img.info["transparency"] = bytes([0] * 256)
    item = PictureItem(self_ref="#/pictures/0")
    element = ItemAndImageEnrichmentElement(item=item, image=img)

    model = _RecordingPictureDescriptionModel()
    doc = DoclingDocument(name="test")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(model(doc=doc, element_batch=[element]))
        palette_warnings = [x for x in w if "Palette images" in str(x.message)]
        assert model.received_modes == ["RGB"]
        assert len(palette_warnings) == 0, (
            f"PIL palette transparency warning should not fire: {palette_warnings}"
        )
