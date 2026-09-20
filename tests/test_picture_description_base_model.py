# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
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
from docling.datamodel.base_models import (
    FailureCategory,
    ItemAndImageEnrichmentElement,
    VlmStopReason,
)
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmEngineOptions,
    PipelineOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.api_image_request import ApiImageRequestResult

pytestmark = pytest.mark.ml_vlm


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


class _UsagePictureDescriptionModel(_ConfiguredPictureDescriptionModel):
    def _annotate_images(
        self, images: Iterable[Image.Image]
    ) -> Iterable[ApiImageRequestResult]:
        for _image in images:
            yield ApiImageRequestResult(
                text="test description",
                num_tokens=42,
                stop_reason=VlmStopReason.END_OF_SEQUENCE,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 32,
                    "total_tokens": 42,
                },
            )


class _FailingPictureDescriptionModel(_ConfiguredPictureDescriptionModel):
    def _annotate_images(
        self, images: Iterable[Image.Image]
    ) -> Iterable[ApiImageRequestResult]:
        for _image in images:
            yield ApiImageRequestResult(
                text="",
                num_tokens=0,
                stop_reason=VlmStopReason.INFERENCE_ERROR,
                error="HTTP 400: Unsupported parameter: temperature",
            )


class _RaisingAfterFailurePictureDescriptionModel(_FailingPictureDescriptionModel):
    """The first batch fails at the provider, the second raises (a bug or a
    connection reset the model does not catch)."""

    def __init__(self, options: PictureDescriptionBaseOptions) -> None:
        super().__init__(options)
        self.batches = 0

    def _annotate_images(
        self, images: Iterable[Image.Image]
    ) -> Iterable[ApiImageRequestResult]:
        self.batches += 1
        if self.batches > 1:
            raise RuntimeError("boom")
        yield from super()._annotate_images(images)


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
        errors=[],
    )

    pipeline._enrich_document(conv_res)

    assert model.batch_sizes == [2, 2, 1]


def test_picture_description_failed_request_is_recorded_not_stored() -> None:
    """A failed API call leaves the picture without a description and hands the
    failure to the pipeline instead of storing an empty text (#4009)."""
    model = _FailingPictureDescriptionModel(_TestOptions())
    doc = _make_picture_doc(count=1)
    image = Image.new("RGB", (20, 20), "red")

    results = list(
        model(
            doc=doc,
            element_batch=[
                ItemAndImageEnrichmentElement(item=doc.pictures[0], image=image)
            ],
        )
    )

    assert len(results) == 1
    assert results[0].meta is None or results[0].meta.description is None
    errors = model.collect_errors()
    assert len(errors) == 1
    assert errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert "HTTP 400: Unsupported parameter: temperature" in errors[0].error_message
    assert model.collect_errors() == []


def test_pipeline_collects_picture_description_failures_into_conv_res() -> None:
    pipeline = _PictureDescriptionPipeline(PipelineOptions())
    pipeline.enrichment_pipe = [_FailingPictureDescriptionModel(_TestOptions())]
    conv_res = SimpleNamespace(
        document=_make_picture_doc(count=2),
        timings={},
        status="success",
        errors=[],
    )

    pipeline._enrich_document(conv_res)

    assert len(conv_res.errors) == 2
    assert all(
        error.category == FailureCategory.INFERENCE_FAILURE for error in conv_res.errors
    )


def test_pipeline_collects_failures_even_when_a_later_batch_raises() -> None:
    """A conversion that raises takes the failures recorded so far with it; they
    must not surface in the next conversion that runs on the same thread."""
    pipeline = _PictureDescriptionPipeline(PipelineOptions())
    model = _RaisingAfterFailurePictureDescriptionModel(_TestOptions(batch_size=1))
    pipeline.enrichment_pipe = [model]
    conv_res = SimpleNamespace(
        document=_make_picture_doc(count=2),
        timings={},
        status="success",
        errors=[],
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._enrich_document(conv_res)

    assert len(conv_res.errors) == 1
    assert model.collect_errors() == []


def test_picture_description_failures_are_kept_apart_per_thread() -> None:
    """Pipelines share their model instances. Two conversions running at the
    same time (docling-serve's local engine does that) must each collect only
    their own failures."""
    model = _FailingPictureDescriptionModel(_TestOptions())
    image = Image.new("RGB", (20, 20), "red")
    both_failed = threading.Barrier(2, timeout=5)

    def _convert(count: int) -> int:
        doc = _make_picture_doc(count=count)
        list(
            model(
                doc=doc,
                element_batch=[
                    ItemAndImageEnrichmentElement(item=picture, image=image)
                    for picture in doc.pictures
                ],
            )
        )
        both_failed.wait()  # the other conversion has recorded its failures too
        return len(model.collect_errors())

    with ThreadPoolExecutor(max_workers=2) as pool:
        collected = sorted(pool.map(_convert, [1, 3]))

    assert collected == [1, 3]


def test_picture_description_stores_usage_payload_on_description_meta() -> None:
    model = _UsagePictureDescriptionModel(_TestOptions())
    doc = _make_picture_doc(count=1)
    image = Image.new("RGB", (20, 20), "red")

    results = list(
        model(
            doc=doc,
            element_batch=[
                ItemAndImageEnrichmentElement(item=doc.pictures[0], image=image)
            ],
        )
    )

    assert len(results) == 1
    picture = results[0]
    assert picture.meta is not None
    assert picture.meta.description is not None
    assert picture.meta.description.text == "test description"
    assert picture.meta.description.get_custom_part()["docling__usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 32,
        "total_tokens": 42,
    }


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
    with pytest.raises(ValueError, match="batch_size"):
        PictureDescriptionVlmEngineOptions.from_preset("smolvlm", batch_size=0)


def test_picture_description_preset_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="scale"):
        PictureDescriptionVlmEngineOptions.from_preset("smolvlm", scale=0)
