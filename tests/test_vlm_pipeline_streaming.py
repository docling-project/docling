# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import PurePath
from types import SimpleNamespace

from docling_core.types.doc import Size, TextItem
from PIL import Image

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    Page,
    PagePredictions,
    VlmPrediction,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import DocumentLimits, settings
from docling.pipeline.vlm_pipeline import VlmPipeline


class _Tracker:
    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0


class _PageBackend(PdfPageBackend):
    def __init__(self, page_no: int, tracker: _Tracker, valid: bool = True) -> None:
        self._page_no = page_no
        self._tracker = tracker
        self._valid = valid
        self._unloaded = False
        tracker.live += 1
        tracker.high_water = max(tracker.high_water, tracker.live)

    @property
    def page_no(self) -> int:
        return self._page_no

    def get_text_in_rect(self, bbox):
        return f"backend page {self.page_no}"

    def get_segmented_page(self):
        return None

    def get_text_cells(self):
        return []

    def get_bitmap_rects(self, scale: float = 1):
        return []

    def get_page_image(self, scale: float = 1, cropbox=None):
        return Image.new("RGB", (100, 100), (self.page_no, 0, 0))

    def get_size(self) -> Size:
        return Size(width=100, height=100)

    def is_valid(self) -> bool:
        return self._valid

    def unload(self) -> None:
        if not self._unloaded:
            self._unloaded = True
            self._tracker.live -= 1


class _StreamingBackend(PdfDocumentBackend):
    supports_random_page_access = False

    def __init__(
        self,
        page_nos: list[int],
        tracker: _Tracker,
        failed_page_nos: set[int] | None = None,
    ) -> None:
        self._page_nos = page_nos
        self._tracker = tracker
        self._failed_page_nos = failed_page_nos or set()

    def is_valid(self) -> bool:
        return True

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise AssertionError("streaming VLM must not call load_page()")

    def page_count(self) -> int:
        return max(self._page_nos)

    def iter_pages(self):
        for page_no in self._page_nos:
            yield _PageBackend(
                page_no, self._tracker, valid=page_no not in self._failed_page_nos
            )

    def unload(self) -> None:
        return None


class _RandomAccessBackend(_StreamingBackend):
    supports_random_page_access = True

    def __init__(
        self,
        page_nos: list[int],
        tracker: _Tracker,
        failed_page_nos: set[int] | None = None,
    ) -> None:
        super().__init__(page_nos, tracker, failed_page_nos)
        self.load_calls: list[int] = []

    def load_page(self, page_no: int) -> PdfPageBackend:
        self.load_calls.append(page_no)
        return _PageBackend(
            page_no + 1,
            self._tracker,
            valid=page_no + 1 not in self._failed_page_nos,
        )

    def iter_pages(self):
        raise AssertionError("random-access VLM must use load_page()")


class _PredictDoctags:
    def __init__(self, tag: str = "text") -> None:
        self._tag = tag

    def __call__(self, conv_res, pages):
        for page in pages:
            assert page.image is not None
            page.predictions = PagePredictions(
                vlm_response=VlmPrediction(
                    text=(
                        f"<doctag><{self._tag}><loc_10><loc_10><loc_90><loc_90>"
                        f"model page {page.page_no}</{self._tag}></doctag>"
                    )
                )
            )
            yield page


def _run_pipeline(
    *,
    page_nos: list[int],
    force_backend_text: bool,
    generate_page_images: bool,
    generate_picture_images: bool,
    tag: str = "text",
    random_access: bool = False,
    failed_page_nos: set[int] | None = None,
    document_timeout: float | None = None,
):
    tracker = _Tracker()
    backend = (
        _RandomAccessBackend(page_nos, tracker, failed_page_nos)
        if random_access
        else _StreamingBackend(page_nos, tracker, failed_page_nos)
    )
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=document_timeout,
        force_backend_text=force_backend_text,
        generate_page_images=generate_page_images,
        generate_picture_images=generate_picture_images,
        images_scale=1.0,
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="test",
            response_format=ResponseFormat.DOCTAGS,
            inference_framework=InferenceFramework.TRANSFORMERS,
        ),
    )
    pipeline.force_backend_text = force_backend_text
    pipeline.build_pipe = [_PredictDoctags(tag)]
    conv_res = SimpleNamespace(
        input=SimpleNamespace(
            _backend=backend,
            file=PurePath("test.pdf"),
            limits=DocumentLimits(page_range=(5, 9)),
            page_count=9,
        ),
        errors=[],
        pages=[],
        status=ConversionStatus.STARTED,
        timings={},
    )
    pipeline._build_document(conv_res)
    return conv_res, tracker, backend


def test_vlm_streams_out_of_order_pages_and_releases_each_batch(monkeypatch) -> None:
    monkeypatch.setattr(settings.perf, "page_batch_size", 3)

    conv_res, tracker, _backend = _run_pipeline(
        page_nos=[2, 9, 5, 7, 6, 8],
        force_backend_text=True,
        generate_page_images=False,
        generate_picture_images=False,
    )

    assert [page.page_no for page in conv_res.pages] == [5, 6, 7, 8, 9]
    assert sorted(conv_res.document.pages) == [5, 6, 7, 8, 9]
    assert {
        item.prov[0].page_no: item.text
        for item, _level in conv_res.document.iterate_items()
        if isinstance(item, TextItem) and item.prov
    } == {page_no: f"backend page {page_no}" for page_no in range(5, 10)}
    assert tracker.live == 0
    assert tracker.high_water <= 3
    assert all(
        page._backend is None and not page._image_cache for page in conv_res.pages
    )
    assert all(page.image is None for page in conv_res.document.pages.values())


def test_vlm_uses_indexed_loading_for_random_access_backends(monkeypatch) -> None:
    monkeypatch.setattr(settings.perf, "page_batch_size", 2)

    conv_res, tracker, backend = _run_pipeline(
        page_nos=list(range(1, 10)),
        force_backend_text=False,
        generate_page_images=False,
        generate_picture_images=False,
        random_access=True,
    )

    assert isinstance(backend, _RandomAccessBackend)
    assert backend.load_calls == [4, 5, 6, 7, 8]
    assert [page.page_no for page in conv_res.pages] == [5, 6, 7, 8, 9]
    assert tracker.live == 0
    assert tracker.high_water <= 2


def test_vlm_preserves_successes_around_a_failed_page(monkeypatch) -> None:
    monkeypatch.setattr(settings.perf, "page_batch_size", 2)

    conv_res, tracker, _backend = _run_pipeline(
        page_nos=[9, 5, 7, 6, 8],
        force_backend_text=False,
        generate_page_images=False,
        generate_picture_images=False,
        failed_page_nos={7},
    )

    assert [page.page_no for page in conv_res.pages] == [5, 6, 8, 9]
    assert sorted(conv_res.document.pages) == [5, 6, 8, 9]
    assert [(error.page_no, error.category) for error in conv_res.errors] == [
        (7, FailureCategory.BACKEND_FAILURE)
    ]
    assert (
        VlmPipeline._determine_status(VlmPipeline.__new__(VlmPipeline), conv_res)
        == ConversionStatus.PARTIAL_SUCCESS
    )
    assert tracker.live == 0


def test_vlm_timeout_stops_iteration_and_releases_live_pages(monkeypatch) -> None:
    monkeypatch.setattr(settings.perf, "page_batch_size", 2)

    conv_res, tracker, _backend = _run_pipeline(
        page_nos=[9, 5, 7, 6, 8],
        force_backend_text=False,
        generate_page_images=False,
        generate_picture_images=False,
        document_timeout=0.0,
    )

    assert [page.page_no for page in conv_res.pages] == [5, 9]
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.TIMEOUT
    assert conv_res.status == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live == 0


def test_vlm_text_response_keeps_absolute_page_number_after_concatenation() -> None:
    tracker = _Tracker()
    page = Page(
        page_no=5,
        size=Size(width=100, height=100),
        predictions=PagePredictions(vlm_response=VlmPrediction(text="# Page five")),
    )
    page._backend = _PageBackend(5, tracker)
    page._default_image_scale = 1.0
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        generate_page_images=False,
        generate_picture_images=False,
        images_scale=1.0,
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="test",
            response_format=ResponseFormat.MARKDOWN,
            inference_framework=InferenceFramework.TRANSFORMERS,
        ),
    )
    pipeline.force_backend_text = False
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")), errors=[]
    )

    page_document = pipeline._finalize_page_document(conv_res, page)
    document = pipeline._concatenate_page_documents([(5, page_document)])
    pipeline._release_page_resources(page)

    assert sorted(document.pages) == [5]
    assert document.texts[0].text == "Page five"
    assert document.texts[0].prov[0].page_no == 5
    assert tracker.live == 0


def test_vlm_owns_requested_page_and_picture_images_after_release(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings.perf, "page_batch_size", 1)

    conv_res, tracker, _backend = _run_pipeline(
        page_nos=[5, 6],
        force_backend_text=False,
        generate_page_images=True,
        generate_picture_images=True,
        tag="picture",
    )

    assert tracker.live == 0
    assert all(
        page._backend is None and not page._image_cache for page in conv_res.pages
    )
    assert all(page.image is not None for page in conv_res.document.pages.values())
    assert conv_res.document.pictures
    assert all(picture.image is not None for picture in conv_res.document.pictures)
