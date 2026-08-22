import json
from types import SimpleNamespace

from docling_core.types.doc import Size

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    VlmStopReason,
)
from docling.datamodel.settings import DocumentLimits
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Tracker:
    def __init__(self) -> None:
        self.live_pages = 0
        self.live_images = 0
        self.page_high_water = 0
        self.image_high_water = 0


class _Image:
    def __init__(self, page_no: int, tracker: _Tracker) -> None:
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
        return []

    def get_bitmap_rects(self, scale: float = 1):
        return []

    def get_page_image(self, scale: float = 1, cropbox=None):
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

    def process_images(self, images, prompt):
        image = images[0]
        if image.page_no in self._failed_page_nos:
            raise RuntimeError(f"page {image.page_no} failed")
        yield SimpleNamespace(
            text=json.dumps({"page": image.page_no}),
            stop_reason=(
                VlmStopReason.LENGTH
                if image.page_no in self._truncated_page_nos
                else VlmStopReason.END_OF_SEQUENCE
            ),
        )


def _run_pipeline(
    *,
    page_nos: list[int],
    page_range: tuple[int, int],
    invalid_page_nos: set[int] | None = None,
    failed_page_nos: set[int] | None = None,
    truncated_page_nos: set[int] | None = None,
    document_timeout: float | None = None,
):
    tracker = _Tracker()
    backend = _StreamingBackend(page_nos, tracker, invalid_page_nos)
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=document_timeout,
        vlm_options=SimpleNamespace(scale=1.0),
    )
    pipeline.vlm_model = _Model(
        failed_page_nos=failed_page_nos,
        truncated_page_nos=truncated_page_nos,
    )
    ext_res = SimpleNamespace(
        input=SimpleNamespace(
            _backend=backend,
            limits=DocumentLimits(page_range=page_range),
        ),
        pages=[],
        errors=[],
        status=ConversionStatus.PENDING,
    )

    pipeline._extract_data(ext_res, template="{}")
    return pipeline, ext_res, tracker


def test_extraction_streams_out_of_order_pages_with_bounded_resources() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=list(range(80, 0, -1)),
        page_range=(5, 75),
        invalid_page_nos={41},
        truncated_page_nos={40},
    )

    assert [page.page_no for page in ext_res.pages] == [
        page_no for page_no in range(5, 76) if page_no != 41
    ]
    assert all(page.extracted_data == {"page": page.page_no} for page in ext_res.pages)
    assert pipeline._determine_status(ext_res) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0
    assert tracker.page_high_water == tracker.image_high_water == 1


def test_extraction_records_failed_page_by_absolute_number_and_continues() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=[2, 9, 5, 7, 6, 8],
        page_range=(5, 9),
        failed_page_nos={7},
    )

    assert [page.page_no for page in ext_res.pages] == [5, 6, 7, 8, 9]
    assert ext_res.pages[2].errors == ["page 7 failed"]
    assert pipeline._determine_status(ext_res) == ConversionStatus.FAILURE
    assert tracker.live_pages == tracker.live_images == 0


def test_extraction_timeout_keeps_partial_result_and_releases_page() -> None:
    pipeline, ext_res, tracker = _run_pipeline(
        page_nos=[9, 5, 7, 6, 8],
        page_range=(5, 9),
        document_timeout=0.0,
    )

    assert [page.page_no for page in ext_res.pages] == [9]
    assert [error.category for error in ext_res.errors] == [FailureCategory.TIMEOUT]
    assert pipeline._determine_status(ext_res) == ConversionStatus.PARTIAL_SUCCESS
    assert tracker.live_pages == tracker.live_images == 0
