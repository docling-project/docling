"""Test that PictureDescriptionBaseModel converts non-RGB images to RGB."""

from collections.abc import Iterable
from typing import ClassVar, List, Type

from docling_core.types.doc import DoclingDocument, PictureItem
from PIL import Image

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel


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
