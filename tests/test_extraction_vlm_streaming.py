# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace

import pytest
from docling_core.types.doc import Size
from PIL.Image import Image

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.extraction import PageScope
from docling.datamodel.extraction_options import (
    NU_EXTRACT_2B_TRANSFORMERS,
    ChannelSelection,
)
from docling.datamodel.settings import DocumentLimits
from docling.models.extraction.prompt_utils import prepare_legacy_target
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Tracker:
    def __init__(self) -> None:
        self.live_pages = 0
        self.live_images = 0
        self.page_high_water = 0
        self.image_high_water = 0
        self.render_scales: list[float] = []


class _Image(Image):
    def __init__(self, page_no: int, tracker: _Tracker) -> None:
        super().__init__()
        self.page_no = page_no
        self._tracker = tracker
        self._closed = False
        tracker.live_images += 1
        tracker.image_high_water = max(tracker.image_high_water, tracker.live_images)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._tracker.live_images -= 1


class _PageBackend(PdfPageBackend):
    def __init__(self, page_no: int, tracker: _Tracker, valid: bool = True) -> None:
        self._page_no = page_no
        self._tracker = tracker
        self._valid = valid
        self._unloaded = False
        tracker.live_pages += 1
        tracker.page_high_water = max(tracker.page_high_water, tracker.live_pages)

    @property
    def page_no(self) -> int:
        return self._page_no

    def get_text_in_rect(self, bbox):
        return ""

    def get_segmented_page(self):
        return None

    def get_text_cells(self):
        return [SimpleNamespace(text=f"page {self.page_no}")]

    def get_bitmap_rects(self, scale: float = 1):
        return []

    def get_page_image(self, scale: float = 1, cropbox=None):
        self._tracker.render_scales.append(scale)
        return _Image(self.page_no, self._tracker)

    def get_size(self) -> Size:
        return Size(width=100, height=100)

    def is_valid(self) -> bool:
        return self._valid

    def unload(self) -> None:
        if not self._unloaded:
            self._unloaded = True
            self._tracker.live_pages -= 1


class _StreamingBackend(PdfDocumentBackend):
    supports_random_page_access = False

    def __init__(
        self,
        page_nos: list[int],
        tracker: _Tracker,
        invalid_page_nos: set[int] | None = None,
    ) -> None:
        self._page_nos = page_nos
        self._tracker = tracker
        self._invalid_page_nos = invalid_page_nos or set()

    def is_valid(self) -> bool:
        return True

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise AssertionError("streaming extraction must not call load_page()")

    def page_count(self) -> int:
        return max(self._page_nos)

    def iter_pages(self):
        for page_no in self._page_nos:
            yield _PageBackend(
                page_no,
                self._tracker,
                valid=page_no not in self._invalid_page_nos,
            )

    def unload(self) -> None:
        return None


class _Model:
    def __init__(
        self,
        *,
        failed_page_nos: set[int] | None = None,
        truncated_page_nos: set[int] | None = None,
    ) -> None:
        self._failed_page_nos = failed_page_nos or set()
        self._truncated_page_nos = truncated_page_nos or set()

    def process(self, requests, target):
        content = requests[0][0]
        page_no = (
            content.image.page_no
            if content.type == "image"
            else int(content.text.split()[-1])
        )
        image = content.image if content.type == "image" else None
        if image is not None:
            assert not image._closed
            assert image._tracker.live_pages == image._tracker.live_images == 1
        if page_no in self._failed_page_nos:
            raise RuntimeError(f"page {page_no} failed")
        yield VlmPrediction(
            text=json.dumps({"page": page_no}),
            stop_reason=(
                VlmStopReason.LENGTH
                if page_no in self._truncated_page_nos
                else VlmStopReason.END_OF_SEQUENCE
            ),
        )
        if image is not None:
            assert (
                not image._closed
            )  # Also live when the consumer resumes this lazy iterator.


def _run_pipeline(
    *,
    page_nos: list[int],
    page_range: tuple[int, int],
    invalid_page_nos: set[int] | None = None,
    failed_page_nos: set[int] | None = None,
    truncated_page_nos: set[int] | None = None,
    document_timeout: float | None = None,
    scale: float = 1.0,
    max_size: int | None = None,
    channel: ChannelSelection = ChannelSelection.AUTO,
    model=None,
    target="{}",
):
    tracker = _Tracker()
    backend = _StreamingBackend(page_nos, tracker, invalid_page_nos)
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=document_timeout,
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS.model_copy(
            update={"scale": scale, "max_size": max_size}
        ),
        input_channels=channel,
        markdown_params=None,
    )
    pipeline.vlm_model = model or _Model(
        failed_page_nos=failed_page_nos,
        truncated_page_nos=truncated_page_nos,
    )
    ext_res = SimpleNamespace(
        input=SimpleNamespace(
            _backend=backend,
            limits=DocumentLimits(page_range=page_range),
        ),
        items=[],
        errors=[],
        status=ConversionStatus.PENDING,
    )

    pipeline._extract_data(ext_res, target=target)
    return pipeline, ext_res, tracker


def test_extraction_streams_out_of_order_pages_with_bounded_resources() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=list(range(80, 0, -1)),
        page_range=(5, 75),
        invalid_page_nos={41},
        truncated_page_nos={40},
    )

    assert [page.scope.page_no for page in ext_res.items] == list(range(5, 76))
    assert all(
        page.extracted_data == {"page": page.scope.page_no}
        for page in ext_res.items
        if page.scope.page_no != 41
    )
    assert (
        ext_res.items[36]
        .errors[0]
        .error_message.startswith("Page 41 backend is not valid")
    )
    assert pipeline._determine_status(ext_res) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0
    assert tracker.page_high_water == tracker.image_high_water == 1


def test_extraction_records_failed_page_by_absolute_number_and_continues() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=[2, 9, 5, 7, 6, 8],
        page_range=(5, 9),
        failed_page_nos={7},
    )

    assert [page.scope.page_no for page in ext_res.items] == [5, 6, 7, 8, 9]
    assert [error.error_message for error in ext_res.items[2].errors] == [
        "page 7 failed"
    ]
    assert pipeline._determine_status(ext_res) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0


def test_extraction_timeout_keeps_partial_result_and_releases_page() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=[9, 5, 7, 6, 8],
        page_range=(5, 9),
        document_timeout=0.0,
    )

    assert [page.scope.page_no for page in ext_res.items] == [5, 6, 7, 8, 9]
    assert all(page.extracted_data is None for page in ext_res.items)
    assert [error.category for error in ext_res.errors] == [FailureCategory.TIMEOUT]
    assert pipeline._determine_status(ext_res) == ConversionStatus.FAILURE
    assert tracker.live_pages == tracker.live_images == 0


def test_extraction_applies_max_image_size_before_rendering() -> None:
    _, _, tracker = _run_pipeline(
        page_nos=[1],
        page_range=(1, 1),
        scale=2.0,
        max_size=50,
    )

    assert tracker.render_scales == [0.5]


def test_invalid_json_is_a_failed_extraction() -> None:
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    ext_res = SimpleNamespace(items=[], errors=[], status=ConversionStatus.PENDING)
    ext_res.items.append(
        pipeline._prediction_to_item(
            PageScope(page_no=1),
            [VlmPrediction(text="not json", stop_reason=VlmStopReason.END_OF_SEQUENCE)],
            prepare_legacy_target("{}", NU_EXTRACT_2B_TRANSFORMERS.model_spec),
        )
    )

    assert (
        ext_res.items[0]
        .errors[0]
        .error_message.startswith("Model returned invalid JSON")
    )
    assert pipeline._determine_status(ext_res) == ConversionStatus.FAILURE


@pytest.mark.parametrize("channel", list(ChannelSelection))
def test_native_page_chunks_stream_for_every_channel(channel) -> None:
    pipeline, result, tracker = _run_pipeline(
        page_nos=[1, 2, 3], page_range=(2, 3), channel=channel
    )
    assert [item.scope.page_no for item in result.items] == [2, 3]
    assert pipeline._determine_status(result) == ConversionStatus.SUCCESS
    assert tracker.page_high_water == 1
    assert tracker.image_high_water == (0 if channel == ChannelSelection.TEXT else 1)
    assert tracker.live_pages == tracker.live_images == 0


def test_chunk_generator_close_releases_current_page_and_image() -> None:
    import time

    tracker = _Tracker()
    backend = _StreamingBackend([1, 2], tracker)
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=None,
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS,
        input_channels=ChannelSelection.IMAGE,
        markdown_params=None,
    )
    result = SimpleNamespace(
        input=SimpleNamespace(_backend=backend, limits=DocumentLimits()),
        items=[],
        errors=[],
    )
    chunks = pipeline._iter_extraction_chunks(
        result, pipeline._prepare_target("{}"), time.monotonic()
    )
    chunk = next(chunks)
    assert chunk.scope == PageScope(page_no=1)
    assert tracker.live_pages == tracker.live_images == 1
    chunks.close()
    assert tracker.live_pages == tracker.live_images == 0
    assert tracker.page_high_water == tracker.image_high_water == 1


def test_missing_selected_page_is_a_scoped_failure() -> None:
    pipeline, result, tracker = _run_pipeline(page_nos=[1, 3], page_range=(1, 3))
    assert [item.scope.page_no for item in result.items] == [1, 2, 3]
    assert [error.error_message for error in result.items[1].errors] == [
        "Selected page is missing from backend"
    ]
    assert pipeline._determine_status(result) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0


def test_random_page_load_failure_does_not_hide_selected_pages() -> None:
    class Backend(_StreamingBackend):
        supports_random_page_access = True

        def load_page(self, index):
            if index == 1:
                raise RuntimeError("page loading failed")
            return _PageBackend(index + 1, self._tracker)

    tracker = _Tracker()
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=None,
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS,
        input_channels=ChannelSelection.IMAGE,
        markdown_params=None,
    )
    pipeline.vlm_model = _Model()
    result = SimpleNamespace(
        input=SimpleNamespace(
            _backend=Backend([1, 2, 3], tracker), limits=DocumentLimits()
        ),
        items=[],
        errors=[],
    )
    pipeline._extract_data(result, target="{}")
    assert [item.scope.page_no for item in result.items] == [1, 2, 3]
    assert [error.error_message for error in result.items[1].errors] == [
        "page loading failed"
    ]
    assert pipeline._determine_status(result) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0
    assert tracker.page_high_water == tracker.image_high_water == 1


@pytest.mark.parametrize(
    "answer,lazy_failure,state",
    [
        ("not json", False, "not_run"),
        ('{"total": null}', False, "failed"),
        ('{"total": 42}', True, "not_run"),
    ],
)
def test_owned_resources_and_raw_metadata_survive_parse_schema_and_lazy_failures(
    answer, lazy_failure, state
) -> None:
    from docling.datamodel.extraction import ExtractionTarget

    class Model:
        def process(self, requests, target):
            image = requests[0][0].image
            assert not image._closed
            yield VlmPrediction(
                text=answer,
                num_tokens=7,
                generation_time=0.5,
                usage={"tokens": 7},
                stop_reason=VlmStopReason.END_OF_SEQUENCE,
            )
            assert not image._closed
            if lazy_failure:
                raise RuntimeError("lazy inference failed")

    target = ExtractionTarget(
        output_schema={
            "type": "object",
            "properties": {"total": {"type": "number"}},
            "required": ["total"],
        }
    )
    pipeline, result, tracker = _run_pipeline(
        page_nos=[1, 2], page_range=(1, 2), model=Model(), target=target
    )
    assert pipeline._determine_status(result) == ConversionStatus.FAILURE
    assert all(
        item.raw_text == answer
        and item.inference_metadata.num_tokens == 7
        and item.inference_metadata.generation_time == 0.5
        and item.inference_metadata.usage == {"tokens": 7}
        and item.validation_status == state
        and item.extracted_data is None
        for item in result.items
    )
    assert tracker.page_high_water == tracker.image_high_water == 1
    assert tracker.live_pages == tracker.live_images == 0
    if lazy_failure:
        assert all(
            [error.error_message for error in item.errors] == ["lazy inference failed"]
            for item in result.items
        )
