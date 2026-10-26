# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test that picture enrichment models receive RGB images.

Transparent pixels must reach the models as white, not as the black colour
that a plain ``convert("RGB")`` keeps under the alpha channel.
"""

from collections.abc import Iterable
from typing import ClassVar, Dict, List, Type

import pytest
from docling_core.types.doc import DoclingDocument, PictureItem
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.picture_classification_options import (
    DocumentPictureClassifierOptions,
)
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngine,
    ImageClassificationEngineInput,
    ImageClassificationEngineOutput,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
)

pytestmark = pytest.mark.ml_vlm


class _TestOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[str] = "test"


class _RecordingPictureDescriptionModel(PictureDescriptionBaseModel):
    """Spy subclass that records the images arriving at _annotate_images."""

    def __init__(self) -> None:
        self.enabled = True
        self.options = _TestOptions()
        self.provenance = "test"
        self.received_modes: List[str] = []
        self.received_images: List[Image.Image] = []

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return _TestOptions

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        for image in images:
            self.received_modes.append(image.mode)
            self.received_images.append(image)
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


class _RecordingClassificationEngine(BaseImageClassificationEngine):
    """Spy engine that records the images arriving at predict_batch."""

    def __init__(self) -> None:
        self.received_images: List[Image.Image] = []

    def initialize(self) -> None:
        pass

    def predict_batch(
        self, input_batch: List[ImageClassificationEngineInput]
    ) -> List[ImageClassificationEngineOutput]:
        self.received_images.extend(item.image for item in input_batch)
        return [
            ImageClassificationEngineOutput(label_ids=[0], scores=[1.0])
            for _ in input_batch
        ]

    def get_label_mapping(self) -> Dict[int, str]:
        return {0: "test"}


def _transparent_rgba() -> Image.Image:
    return Image.new("RGBA", (10, 10), (0, 0, 0, 0))


def _transparent_palette() -> Image.Image:
    img = Image.new("P", (10, 10), 0)
    img.putpalette([0, 0, 0] * 256)
    img.info["transparency"] = bytes([0] * 256)
    return img


_TRANSPARENT_IMAGES = pytest.mark.parametrize(
    "make_image",
    [_transparent_rgba, _transparent_palette],
    ids=["rgba", "palette_trns"],
)


@_TRANSPARENT_IMAGES
def test_picture_description_flattens_transparency_onto_white(make_image) -> None:
    item = PictureItem(self_ref="#/pictures/0")
    element = ItemAndImageEnrichmentElement(item=item, image=make_image())
    model = _RecordingPictureDescriptionModel()

    list(model(doc=DoclingDocument(name="test"), element_batch=[element]))

    assert [img.getpixel((0, 0)) for img in model.received_images] == [(255, 255, 255)]


@_TRANSPARENT_IMAGES
def test_picture_classifier_flattens_transparency_onto_white(make_image) -> None:
    classifier = DocumentPictureClassifier(
        enabled=False,
        artifacts_path=None,
        options=DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        ),
        accelerator_options=AcceleratorOptions(),
    )
    engine = _RecordingClassificationEngine()
    classifier.enabled = True
    classifier.engine = engine
    classifier._classes = engine.get_label_mapping()
    item = PictureItem(self_ref="#/pictures/0")
    element = ItemAndImageEnrichmentElement(item=item, image=make_image())

    list(classifier(doc=DoclingDocument(name="test"), element_batch=[element]))

    assert [img.getpixel((0, 0)) for img in engine.received_images] == [(255, 255, 255)]


@_TRANSPARENT_IMAGES
def test_normalize_image_flattens_transparency_onto_white(make_image) -> None:
    from docling.models.inference_engines.vlm._utils import normalize_image_to_pil

    result = normalize_image_to_pil(make_image())

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
