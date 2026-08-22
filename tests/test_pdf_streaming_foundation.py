import itertools
import time
from types import MethodType, SimpleNamespace

import pytest
from docling_core.types.doc import DoclingDocument, Size, TextItem
from PIL import Image

import docling.experimental.pipeline.threaded_layout_vlm_pipeline as threaded_layout_module
from docling.backend.pdf_backend import (
    PdfDocumentBackend,
    PdfPageBackend,
    iter_pdf_page_backends,
)
from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DocumentLimits
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import (
    RunContext,
    StandardPdfPipeline,
    ThreadedPipelineStage,
    ThreadedQueue,
)


class _BackendTracker:
    def __init__(self) -> None:
        self.live = 0
        self.high_water = 0


class _TrackedPageBackend(PdfPageBackend):
    def __init__(self, page_no: int, tracker: _BackendTracker) -> None:
        self._page_no = page_no
        self._tracker = tracker
        self._unloaded = False
        tracker.live += 1
        tracker.high_water = max(tracker.high_water, tracker.live)

    @property
    def page_no(self) -> int:
        return self._page_no

    def get_text_in_rect(self, bbox):
        return ""

    def get_segmented_page(self):
        return None

    def get_text_cells(self):
        return []

    def get_bitmap_rects(self, scale: float = 1):
        return []

    def get_page_image(self, scale: float = 1, cropbox=None):
        raise AssertionError("synthetic pipeline must not render")

    def get_size(self) -> Size:
        return Size(width=100, height=200)

    def is_valid(self) -> bool:
        return True

    def unload(self) -> None:
        if not self._unloaded:
            self._unloaded = True
            self._tracker.live -= 1


class _StreamingBackend(PdfDocumentBackend):
    supports_random_page_access = False

    def __init__(self, page_nos: list[int], tracker: _BackendTracker) -> None:
        self._page_nos = page_nos
        self._tracker = tracker

    def is_valid(self) -> bool:
        return True

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise AssertionError("streaming backend must not use load_page()")

    def page_count(self) -> int:
        return len(self._page_nos)

    def iter_pages(self):
        for page_no in self._page_nos:
            yield _TrackedPageBackend(page_no, self._tracker)

    def unload(self) -> None:
        return None


class _SlowPassModel:
    def __call__(self, conv_res, pages):
        time.sleep(0.001)
        return pages


def _make_run_context(postprocess=None) -> RunContext:
    stage = ThreadedPipelineStage(
        name="pass",
        model=_SlowPassModel(),
        batch_size=1,
        batch_timeout=0.001,
        queue_max_size=2,
        shutdown_timeout=1.0,
        postprocess=postprocess,
    )
    output_queue = ThreadedQueue(2)
    stage.add_output_queue(output_queue)
    return RunContext(stages=[stage], first_stage=stage, output_queue=output_queue)


def _make_conversion_result(backend: PdfDocumentBackend, page_count: int):
    return SimpleNamespace(
        input=SimpleNamespace(
            _backend=backend,
            page_count=page_count,
            limits=DocumentLimits(page_range=(1, page_count)),
        ),
        pages=[],
        errors=[],
        timings={},
        status=None,
    )


def test_requested_page_iterator_unloads_filtered_and_abandoned_pages() -> None:
    tracker = _BackendTracker()
    backend = _StreamingBackend([4, 2, 3], tracker)
    assert list(iter_pdf_page_backends(backend, [])) == []
    assert tracker.high_water == 0

    page_iterator = iter_pdf_page_backends(backend, [2, 3])
    page = next(page_iterator)
    assert page.page_no == 2
    assert tracker.live == 1

    page_iterator.close()
    assert tracker.live == 0


def test_standard_pipeline_bounds_live_streaming_backends() -> None:
    page_count = 80
    tracker = _BackendTracker()
    backend = _StreamingBackend(list(range(page_count, 0, -1)), tracker)
    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline._run_seq = itertools.count(1)
    pipeline._page_sizes_by_no = {}
    pipeline.keep_images = False
    pipeline.keep_backend = False
    pipeline.pipeline_options = SimpleNamespace(
        heading_hierarchy_options=SimpleNamespace(enabled=False, use_bookmarks=False),
        document_timeout=None,
        stage_shutdown_timeout_seconds=1.0,
        generate_parsed_pages=False,
    )
    pipeline._create_run_ctx = MethodType(
        lambda self: _make_run_context(self._release_page_resources), pipeline
    )
    conv_res = _make_conversion_result(backend, page_count)

    pipeline._build_document(conv_res)

    assert [page.page_no for page in conv_res.pages] == list(range(1, page_count + 1))
    assert tracker.live == 0
    assert tracker.high_water <= 6


def test_legacy_pipeline_warns_and_delegates_to_standard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    options = PdfPipelineOptions()
    monkeypatch.setattr(
        StandardPdfPipeline,
        "__init__",
        lambda self, pipeline_options: calls.append(pipeline_options),
    )

    with pytest.warns(
        DeprecationWarning,
        match="LegacyStandardPdfPipeline is deprecated; use StandardPdfPipeline",
    ):
        pipeline = LegacyStandardPdfPipeline(options)

    assert isinstance(pipeline, StandardPdfPipeline)
    assert calls == [options]
    assert "get_default_options" not in LegacyStandardPdfPipeline.__dict__


def test_threaded_layout_vlm_pipeline_bounds_live_streaming_backends() -> None:
    page_count = 80
    tracker = _BackendTracker()
    backend = _StreamingBackend(list(range(page_count, 0, -1)), tracker)
    pipeline = threaded_layout_module.ThreadedLayoutVlmPipeline.__new__(
        threaded_layout_module.ThreadedLayoutVlmPipeline
    )
    pipeline._run_seq = itertools.count(1)
    pipeline.pipeline_options = SimpleNamespace(
        images_scale=1.0,
        generate_page_images=False,
        generate_picture_images=False,
    )
    pipeline._create_run_ctx = MethodType(lambda self: _make_run_context(), pipeline)

    def _finalize_page_document(self, page):
        document = DoclingDocument(name=f"page-{page.page_no}")
        document.add_page(page_no=1, size=page.size)
        return document

    pipeline._finalize_page_document = MethodType(_finalize_page_document, pipeline)
    conv_res = _make_conversion_result(backend, page_count)

    pipeline._build_document(conv_res)

    assert [page.page_no for page in conv_res.pages] == list(range(1, page_count + 1))
    assert sorted(conv_res.document.pages) == list(range(1, page_count + 1))
    assert tracker.live == 0
    assert tracker.high_water <= 6


@pytest.mark.parametrize("generate_page_images", [False, True])
def test_threaded_layout_vlm_finalizes_pages_with_absolute_numbers(
    generate_page_images: bool,
) -> None:
    pipeline = threaded_layout_module.ThreadedLayoutVlmPipeline.__new__(
        threaded_layout_module.ThreadedLayoutVlmPipeline
    )
    pipeline.pipeline_options = SimpleNamespace(
        images_scale=1.0,
        generate_page_images=generate_page_images,
        generate_picture_images=False,
    )
    page_documents = []
    for page_no in [3, 5]:
        page = Page(
            page_no=page_no,
            size=Size(width=100, height=200),
            predictions=PagePredictions(
                vlm_response=VlmPrediction(
                    text=(
                        "<doctag><text><loc_10><loc_10><loc_90><loc_20>"
                        f"page {page_no}</text></doctag>"
                    )
                )
            ),
        )
        page._image_cache = {1.0: Image.new("RGB", (100, 200), "white")}
        page_document = pipeline._finalize_page_document(page)
        assert page_document is not None
        page_documents.append((page_no, page_document))

    document = pipeline._concatenate_page_documents(page_documents)

    assert sorted(document.pages) == [3, 5]
    assert {
        item.prov[0].page_no: item.text
        for item, _level in document.iterate_items()
        if isinstance(item, TextItem) and item.prov
    } == {3: "page 3", 5: "page 5"}
    assert all(
        (page.image is not None) is generate_page_images
        for page in document.pages.values()
    )
