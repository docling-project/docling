from collections.abc import Iterable
from types import SimpleNamespace
from typing import ClassVar, List, Type

import pytest
from docling_core.types.doc import (
    DoclingDocument,
    ImageRef,
    PictureItem,
    ProvenanceItem,
)
from docling_core.types.doc.base import BoundingBox, Size
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmEngineOptions,
    PipelineOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.pipeline.base_pipeline import BasePipeline


class _TestOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[str] = "test"


class _ConfiguredPictureDescriptionModel(PictureDescriptionBaseModel):
    def __init__(self, options: PictureDescriptionBaseOptions) -> None:
        super().__init__(
            enabled=True,
            enable_remote_services=False,
            artifacts_path=None,
            options=options,
            accelerator_options=AcceleratorOptions(),
        )

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return _TestOptions

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        for _image in images:
            yield "test description"


class _BatchRecordingPictureDescriptionModel(_ConfiguredPictureDescriptionModel):
    def __init__(self, options: PictureDescriptionBaseOptions) -> None:
        super().__init__(options)
        self.batch_sizes: List[int] = []

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[PictureItem]:
        element_list = list(element_batch)
        self.batch_sizes.append(len(element_list))
        for element in element_list:
            assert isinstance(element.item, PictureItem)
            yield element.item


class _PictureDescriptionPipeline(BasePipeline):
    def _build_document(self, conv_res):
        return conv_res

    def _determine_status(self, conv_res):
        return conv_res.status

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return PipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend) -> bool:
        return True


def _make_picture_doc(*, count: int, embed_images: bool = True) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    for _ in range(count):
        image = (
            ImageRef.from_pil(Image.new("RGB", (20, 20), "red"), dpi=72)
            if embed_images
            else None
        )
        doc.add_picture(image=image)
    return doc


def test_picture_description_options_control_batch_size_and_scale() -> None:
    model = _ConfiguredPictureDescriptionModel(_TestOptions(batch_size=3, scale=1.5))

    assert model.elements_batch_size == 3
    assert model.images_scale == 1.5


def test_picture_description_batch_size_controls_pipeline_chunking() -> None:
    pipeline = _PictureDescriptionPipeline(PipelineOptions())
    model = _BatchRecordingPictureDescriptionModel(_TestOptions(batch_size=2))
    pipeline.enrichment_pipe = [model]
    conv_res = SimpleNamespace(
        document=_make_picture_doc(count=5),
        timings={},
        status="success",
    )

    pipeline._enrich_document(conv_res)

    assert model.batch_sizes == [2, 2, 1]


def test_picture_description_scale_is_used_for_cropping() -> None:
    model = _ConfiguredPictureDescriptionModel(_TestOptions(scale=1.5))
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=100, height=100))
    picture = doc.add_picture(
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=10, t=10, r=30, b=30),
            charspan=(0, 0),
        )
    )

    class _PageSpy:
        def __init__(self):
            self.page_no = 1
            self.calls = []

        def get_image(self, *, scale, cropbox):
            self.calls.append({"scale": scale, "cropbox": cropbox})
            return Image.new("RGB", (5, 5), "blue")

    page = _PageSpy()
    conv_res = SimpleNamespace(document=doc, pages=[page])

    prepared = model.prepare_element(conv_res=conv_res, element=picture)

    assert prepared is not None
    assert page.calls[0]["scale"] == 1.5


def test_picture_description_embedded_images_keep_original_size() -> None:
    model = _ConfiguredPictureDescriptionModel(_TestOptions(scale=1.5))
    doc = _make_picture_doc(count=1, embed_images=True)

    prepared = model.prepare_element(
        conv_res=SimpleNamespace(document=doc, pages=[]), element=doc.pictures[0]
    )

    assert prepared is not None
    assert prepared.image.size == (20, 20)


def test_picture_description_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _TestOptions(batch_size=0)


def test_picture_description_scale_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _TestOptions(scale=0)


def test_picture_description_preset_batch_size_must_be_positive() -> None:
    options = PictureDescriptionVlmEngineOptions.from_preset("smolvlm", batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        _ConfiguredPictureDescriptionModel(options)


def test_picture_description_preset_scale_must_be_positive() -> None:
    options = PictureDescriptionVlmEngineOptions.from_preset("smolvlm", scale=0)

    with pytest.raises(ValueError, match="scale"):
        _ConfiguredPictureDescriptionModel(options)
