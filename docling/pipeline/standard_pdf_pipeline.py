"""Thread-safe, production-ready PDF pipeline
================================================
A self-contained, thread-safe PDF conversion pipeline exploiting parallelism between pipeline stages and models.

* **Per-run isolation** - every :py:meth:`execute` call uses its own bounded queues and worker
  threads so that concurrent invocations never share mutable state.
* **Deterministic run identifiers** - pages are tracked with an internal *run-id* instead of
  relying on :pyfunc:`id`, which may clash after garbage collection.
* **Explicit back-pressure & shutdown** - producers block on full queues; queue *close()*
  propagates downstream so stages terminate deterministically without sentinels.
* **Minimal shared state** - heavyweight models are initialised once per pipeline instance
  and only read by worker threads; no runtime mutability is exposed.
* **Strict typing & clean API usage** - code is fully annotated and respects *coding_rules.md*.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

import numpy as np
from docling_core.types.doc import (
    DocItem,
    ImageRef,
    PageItem,
    PictureItem,
    Size,
    TableItem,
)

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    AssembledUnit,
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings
from docling.models.factories import (
    get_layout_factory,
    get_ocr_factory,
    get_table_structure_factory,
)
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)
from docling.models.stages.heading_hierarchy.heading_hierarchy_model import (
    HeadingHierarchyModel,
)
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.pipeline.base_pipeline import ConvertPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)

STAGE_FAILURE_CATEGORY = {
    "ocr": FailureCategory.INFERENCE_FAILURE,
    "layout": FailureCategory.INFERENCE_FAILURE,
    "table": FailureCategory.INFERENCE_FAILURE,
    "assemble": FailureCategory.INFERENCE_FAILURE,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper data structures
# ──────────────────────────────────────────────────────────────────────────────


def _make_error_item(
    *,
    component_type: DoclingComponentType,
    module_name: str,
    error: Exception,
    category: FailureCategory,
    page_no: int | None = None,
) -> ErrorItem:
    return ErrorItem(
        component_type=component_type,
        module_name=module_name,
        error_message=str(error) or error.__class__.__name__,
        category=category,
        page_no=page_no,
    )


@dataclass
class ThreadedItem:
    """Envelope that travels between pipeline stages."""

    payload: Page | None
    run_id: int  # Unique per *execute* call, monotonic across pipeline instance
    page_no: int
    conv_res: ConversionResult
    error: Exception | None = None
    failure: ErrorItem | None = None
    is_failed: bool = False


@dataclass
class ProcessingResult:
    """Aggregated outcome of a pipeline run."""

    pages: list[Page] = field(default_factory=list)
    failed_pages: list[tuple[int, Exception, ErrorItem | None]] = field(
        default_factory=list
    )
    total_expected: int = 0

    @property
    def success_count(self) -> int:
        return len(self.pages)

    @property
    def failure_count(self) -> int:
        return len(self.failed_pages)

    @property
    def is_partial_success(self) -> bool:
        return 0 < self.success_count < self.total_expected

    @property
    def is_complete_failure(self) -> bool:
        return self.success_count == 0 and self.failure_count > 0


class ThreadedQueue:
    """Bounded queue with blocking put/ get_batch and explicit *close()* semantics."""

    __slots__ = ("_closed", "_items", "_lock", "_max", "_not_empty", "_not_full")

    def __init__(self, max_size: int) -> None:
        self._max: int = max_size
        self._items: deque[ThreadedItem] = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    # ---------------------------------------------------------------- put()
    def put(self, item: ThreadedItem, timeout: float | None = None) -> bool:
        """Block until queue accepts *item* or is closed.  Returns *False* if closed."""
        with self._not_full:
            if self._closed:
                return False
            start = time.monotonic()
            while len(self._items) >= self._max and not self._closed:
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        return False
                    self._not_full.wait(remaining)
                else:
                    self._not_full.wait()
            if self._closed:
                return False
            self._items.append(item)
            self._not_empty.notify()
            return True

    # ------------------------------------------------------------ get_batch()
    def get_batch(self, size: int, timeout: float | None = None) -> list[ThreadedItem]:
        """Return up to *size* items.  Blocks until ≥1 item present or queue closed/timeout."""
        with self._not_empty:
            start = time.monotonic()
            while not self._items and not self._closed:
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        return []
                    self._not_empty.wait(remaining)
                else:
                    self._not_empty.wait()
            batch: list[ThreadedItem] = []
            while self._items and len(batch) < size:
                batch.append(self._items.popleft())
            if batch:
                self._not_full.notify_all()
            return batch

    # ---------------------------------------------------------------- close()
    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    # -------------------------------------------------------------- property
    @property
    def closed(self) -> bool:
        return self._closed


class ThreadedPipelineStage:
    """A single pipeline stage backed by one worker thread."""

    def __init__(
        self,
        *,
        name: str,
        model: Any,
        batch_size: int,
        batch_timeout: float,
        queue_max_size: int,
        postprocess: Callable[[ThreadedItem], None] | None = None,
        timed_out_run_ids: set[int] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.input_queue = ThreadedQueue(queue_max_size)
        self._outputs: list[ThreadedQueue] = []
        self._thread: threading.Thread | None = None
        self._running = False
        self._postprocess = postprocess
        self._timed_out_run_ids = (
            timed_out_run_ids if timed_out_run_ids is not None else set()
        )

    # ---------------------------------------------------------------- wiring
    def add_output_queue(self, q: ThreadedQueue) -> None:
        self._outputs.append(q)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"Stage-{self.name}", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.input_queue.close()
        if self._thread is not None:
            # Give thread 15s to finish naturally before abandoning
            self._thread.join(timeout=15.0)
            if self._thread.is_alive():
                _log.warning(
                    "Stage %s thread did not terminate within 15s. "
                    "Thread is likely stuck in a blocking call and will be abandoned (resources may leak).",
                    self.name,
                )

    # ------------------------------------------------------------------ _run
    def _run(self) -> None:
        try:
            while self._running:
                batch = self.input_queue.get_batch(self.batch_size, self.batch_timeout)
                if not batch and self.input_queue.closed:
                    break
                processed = self._process_batch(batch)
                self._emit(processed)
        except Exception:  # pragma: no cover - top-level guard
            _log.exception("Fatal error in stage %s", self.name)
        finally:
            for q in self._outputs:
                q.close()

    # ----------------------------------------------------- _process_batch()
    def _process_batch(self, batch: Sequence[ThreadedItem]) -> list[ThreadedItem]:
        """Run *model* on *batch* grouped by run_id to maximise batching."""
        groups: dict[int, list[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            groups[itm.run_id].append(itm)

        result: list[ThreadedItem] = []
        for rid, items in groups.items():
            # If run_id is timed out, skip processing but pass through items as-is
            # This allows already-completed work to flow through while aborting new work
            if rid in self._timed_out_run_ids:
                for it in items:
                    it.is_failed = True
                    if it.error is None:
                        it.error = RuntimeError("document timeout exceeded")
                    if it.failure is None:
                        error = it.error or RuntimeError("document timeout exceeded")
                        it.failure = _make_error_item(
                            component_type=DoclingComponentType.PIPELINE,
                            module_name=self.name,
                            error=error,
                            category=FailureCategory.TIMEOUT,
                            page_no=it.page_no,
                        )
                result.extend(items)
                continue

            good: list[ThreadedItem] = [i for i in items if not i.is_failed]
            if not good:
                result.extend(items)
                continue
            try:
                # Filter out None payloads and ensure type safety
                pages_with_payloads = [
                    (i, i.payload) for i in good if i.payload is not None
                ]
                if len(pages_with_payloads) != len(good):
                    # Some items have None payloads, mark all as failed
                    for it in good:
                        it.is_failed = True
                        error = RuntimeError("Page payload is None")
                        it.error = error
                        it.failure = _make_error_item(
                            component_type=DoclingComponentType.PIPELINE,
                            module_name=self.name,
                            error=error,
                            category=FailureCategory.UNKNOWN,
                            page_no=it.page_no,
                        )
                    result.extend(items)
                    continue

                pages: list[Page] = [payload for _, payload in pages_with_payloads]
                if _log.isEnabledFor(logging.DEBUG):
                    _t_start = time.time()
                    _t_mono = time.monotonic()
                processed_pages = list(self.model(good[0].conv_res, pages))  # type: ignore[arg-type]
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(
                        "PIPELINE_PROFILING Stage %s: run_id=%d pages=%s start=%.3f end=%.3f duration=%.3fs",
                        self.name,
                        rid,
                        [it.page_no for it in good],
                        _t_start,
                        time.time(),
                        time.monotonic() - _t_mono,
                    )
                if len(processed_pages) != len(pages):  # strict mismatch guard
                    raise RuntimeError(
                        f"Model {self.name} returned wrong number of pages"
                    )
                for idx, page in enumerate(processed_pages):
                    result.append(
                        ThreadedItem(
                            payload=page,
                            run_id=rid,
                            page_no=good[idx].page_no,
                            conv_res=good[idx].conv_res,
                        )
                    )
            except Exception as exc:
                _log.error(
                    "Stage %s failed for run %d: %s", self.name, rid, exc, exc_info=True
                )
                for it in good:
                    it.is_failed = True
                    it.error = exc
                    it.failure = _make_error_item(
                        component_type=DoclingComponentType.MODEL,
                        module_name=self.name,
                        error=exc,
                        category=STAGE_FAILURE_CATEGORY.get(
                            self.name, FailureCategory.UNKNOWN
                        ),
                        page_no=it.page_no,
                    )
                result.extend(items)
        return result

    # -------------------------------------------------------------- _emit()
    def _emit(self, items: Iterable[ThreadedItem]) -> None:
        for item in items:
            if self._postprocess is not None:
                self._postprocess(item)
            for q in self._outputs:
                if not q.put(item):
                    _log.error("Output queue closed while emitting from %s", self.name)


class PreprocessThreadedStage(ThreadedPipelineStage):
    """Pipeline stage that validates pre-attached backends and runs preprocessing."""

    def __init__(
        self,
        *,
        batch_timeout: float,
        queue_max_size: int,
        model: Any,
        timed_out_run_ids: set[int] | None = None,
    ) -> None:
        super().__init__(
            name="preprocess",
            model=model,
            batch_size=1,
            batch_timeout=batch_timeout,
            queue_max_size=queue_max_size,
            timed_out_run_ids=timed_out_run_ids,
        )

    def _process_batch(self, batch: Sequence[ThreadedItem]) -> list[ThreadedItem]:
        groups: dict[int, list[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            groups[itm.run_id].append(itm)

        result: list[ThreadedItem] = []
        for rid, items in groups.items():
            # If run_id is timed out, skip processing but pass through items as-is
            # This allows already-completed work to flow through while aborting new work
            if rid in self._timed_out_run_ids:
                for it in items:
                    it.is_failed = True
                    if it.error is None:
                        it.error = RuntimeError("document timeout exceeded")
                    if it.failure is None:
                        error = it.error or RuntimeError("document timeout exceeded")
                        it.failure = _make_error_item(
                            component_type=DoclingComponentType.PIPELINE,
                            module_name=self.name,
                            error=error,
                            category=FailureCategory.TIMEOUT,
                            page_no=it.page_no,
                        )
                result.extend(items)
                continue

            good = [i for i in items if not i.is_failed]
            if not good:
                result.extend(items)
                continue

            # Validate backends before the model call so that invalid-page
            # items are emitted exactly once, even if the model later raises.
            invalid: list[ThreadedItem] = []
            valid: list[tuple[ThreadedItem, Page]] = []
            for it in good:
                page = it.payload
                if page is None:
                    it.is_failed = True
                    error = RuntimeError("Page payload is None")
                    it.error = error
                    it.failure = _make_error_item(
                        component_type=DoclingComponentType.PIPELINE,
                        module_name=self.name,
                        error=error,
                        category=FailureCategory.UNKNOWN,
                        page_no=it.page_no,
                    )
                    invalid.append(it)
                elif page._backend is None:
                    it.is_failed = True
                    error = RuntimeError(
                        "Page backend must be attached before preprocess"
                    )
                    it.error = error
                    it.failure = _make_error_item(
                        component_type=DoclingComponentType.PIPELINE,
                        module_name=self.name,
                        error=error,
                        category=FailureCategory.UNKNOWN,
                        page_no=it.page_no,
                    )
                    invalid.append(it)
                elif not page._backend.is_valid():
                    it.is_failed = True
                    error = RuntimeError(f"Page {page.page_no} failed to parse.")
                    it.error = error
                    it.failure = _make_error_item(
                        component_type=DoclingComponentType.DOCUMENT_BACKEND,
                        module_name=self.name,
                        error=error,
                        category=FailureCategory.BACKEND_FAILURE,
                        page_no=it.page_no,
                    )
                    invalid.append(it)
                else:
                    valid.append((it, page))

            result.extend(invalid)

            if not valid:
                continue

            try:
                if _log.isEnabledFor(logging.DEBUG):
                    _t_start = time.time()
                    _t_mono = time.monotonic()
                pages = [page for _, page in valid]
                processed_pages = list(
                    self.model(valid[0][0].conv_res, pages)  # type: ignore[arg-type]
                )
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(
                        "PIPELINE_PROFILING Stage preprocess: run_id=%d pages=%s start=%.3f end=%.3f duration=%.3fs",
                        rid,
                        [it.page_no for it, _ in valid],
                        _t_start,
                        time.time(),
                        time.monotonic() - _t_mono,
                    )
                if len(processed_pages) != len(pages):
                    raise RuntimeError(
                        "PagePreprocessingModel returned unexpected number of pages"
                    )
                for idx, processed_page in enumerate(processed_pages):
                    result.append(
                        ThreadedItem(
                            payload=processed_page,
                            run_id=rid,
                            page_no=valid[idx][0].page_no,
                            conv_res=valid[idx][0].conv_res,
                        )
                    )
            except Exception as exc:
                _log.error(
                    "Stage preprocess failed for run %d, pages %s: %s",
                    rid,
                    [it.page_no for it, _ in valid],
                    exc,
                    exc_info=False,
                )
                for it, _ in valid:
                    it.is_failed = True
                    it.error = exc
                    it.failure = _make_error_item(
                        component_type=DoclingComponentType.MODEL,
                        module_name=self.name,
                        error=exc,
                        category=FailureCategory.UNKNOWN,
                        page_no=it.page_no,
                    )
                result.extend(it for it, _ in valid)
        return result


@dataclass
class RunContext:
    """Wiring for a single *execute* call."""

    stages: list[ThreadedPipelineStage]
    first_stage: ThreadedPipelineStage
    output_queue: ThreadedQueue
    timed_out_run_ids: set[int] = field(default_factory=set)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────


class StandardPdfPipeline(ConvertPipeline):
    """High-performance PDF pipeline with multi-threaded stages."""

    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: ThreadedPdfPipelineOptions = pipeline_options
        self._run_seq = itertools.count(1)  # deterministic, monotonic run ids
        self._page_sizes_by_no: dict[int, Size] = {}

        # initialise heavy models once
        self._init_models()

    # ────────────────────────────────────────────────────────────────────────
    # Heavy-model initialisation & helpers
    # ────────────────────────────────────────────────────────────────────────

    def _init_models(self) -> None:
        art_path = self.artifacts_path
        self.keep_images = (
            self.pipeline_options.generate_page_images
            or self.pipeline_options.generate_picture_images
            or self.pipeline_options.generate_table_images
        )
        self.preprocessing_model = PagePreprocessingModel(
            options=PagePreprocessingOptions(
                images_scale=self.pipeline_options.images_scale
            )
        )
        self.ocr_model = self._make_ocr_model(art_path)
        layout_factory = get_layout_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        self.layout_model = layout_factory.create_instance(
            options=self.pipeline_options.layout_options,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            enable_remote_services=self.pipeline_options.enable_remote_services,
        )
        table_factory = get_table_structure_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        self.table_model = table_factory.create_instance(
            options=self.pipeline_options.table_structure_options,
            enabled=self.pipeline_options.do_table_structure,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            enable_remote_services=self.pipeline_options.enable_remote_services,
        )
        self.assemble_model = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())
        self.heading_hierarchy_model = HeadingHierarchyModel(
            options=self.pipeline_options.heading_hierarchy_options
        )

        # --- optional enrichment ------------------------------------------------
        # Create a copy to avoid mutating pipeline_options in-place,
        # which would change its hash and break pipeline caching (#3109).
        code_formula_opts = self.pipeline_options.code_formula_options.model_copy(
            update={
                "extract_code": self.pipeline_options.do_code_enrichment,
                "extract_formulas": self.pipeline_options.do_formula_enrichment,
            }
        )

        self.enrichment_pipe = [
            # Code Formula Enrichment Model (using new VLM runtime system)
            CodeFormulaVlmModel(
                enabled=self.pipeline_options.do_code_enrichment
                or self.pipeline_options.do_formula_enrichment,
                artifacts_path=self.artifacts_path,
                options=code_formula_opts,
                accelerator_options=self.pipeline_options.accelerator_options,
                enable_remote_services=self.pipeline_options.enable_remote_services,
            ),
            *self.enrichment_pipe,
        ]

        self.keep_backend = any(
            (
                self.pipeline_options.do_formula_enrichment,
                self.pipeline_options.do_code_enrichment,
                self.pipeline_options.do_picture_classification,
                self.pipeline_options.do_picture_description,
                self.pipeline_options.do_chart_extraction,
            )
        )

    # ---------------------------------------------------------------- helpers
    def _make_ocr_model(self, art_path: Path | None) -> Any:
        factory = get_ocr_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    def _release_page_resources(self, item: ThreadedItem) -> None:
        page = item.payload
        if page is None:
            return
        if not self.keep_images:
            page._image_cache = {}
        if not self.keep_backend and page._backend is not None:
            page._backend.unload()
            page._backend = None
        if not self.pipeline_options.generate_parsed_pages:
            page.parsed_page = None

    # ────────────────────────────────────────────────────────────────────────
    # Build - thread pipeline
    # ────────────────────────────────────────────────────────────────────────

    def _create_run_ctx(self) -> RunContext:
        opts = self.pipeline_options
        timed_out_run_ids: set[int] = set()
        preprocess = PreprocessThreadedStage(
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            model=self.preprocessing_model,
            timed_out_run_ids=timed_out_run_ids,
        )
        ocr = ThreadedPipelineStage(
            name="ocr",
            model=self.ocr_model,
            batch_size=opts.ocr_batch_size,
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            timed_out_run_ids=timed_out_run_ids,
        )
        layout = ThreadedPipelineStage(
            name="layout",
            model=self.layout_model,
            batch_size=opts.layout_batch_size,
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            timed_out_run_ids=timed_out_run_ids,
        )
        table = ThreadedPipelineStage(
            name="table",
            model=self.table_model,
            batch_size=opts.table_batch_size,
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            timed_out_run_ids=timed_out_run_ids,
        )
        assemble = ThreadedPipelineStage(
            name="assemble",
            model=self.assemble_model,
            batch_size=1,
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            postprocess=self._release_page_resources,
            timed_out_run_ids=timed_out_run_ids,
        )

        # wire stages
        output_q = ThreadedQueue(opts.queue_max_size)
        preprocess.add_output_queue(ocr.input_queue)
        ocr.add_output_queue(layout.input_queue)
        layout.add_output_queue(table.input_queue)
        table.add_output_queue(assemble.input_queue)
        assemble.add_output_queue(output_q)

        stages = [preprocess, ocr, layout, table, assemble]
        return RunContext(
            stages=stages,
            first_stage=preprocess,
            output_queue=output_q,
            timed_out_run_ids=timed_out_run_ids,
        )

    # --------------------------------------------------------------------- build
    def _iter_requested_page_backends(
        self, backend: PdfDocumentBackend, expected_page_nos: list[int]
    ) -> Iterable[PdfPageBackend]:
        if backend.supports_random_page_access:
            for page_no in expected_page_nos:
                yield backend.load_page(page_no - 1)
            return

        expected_page_no_set = set(expected_page_nos)
        for page_backend in backend.iter_pages():
            if page_backend.page_no in expected_page_no_set:
                yield page_backend

    def _get_expected_page_nos(self, conv_res: ConversionResult) -> list[int]:
        start_page, end_page = conv_res.input.limits.page_range
        return list(
            range(
                max(1, start_page),
                min(conv_res.input.page_count, end_page) + 1,
            )
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Stream-build the document with a dedicated producer thread.

        Note: If a worker thread gets stuck in a blocking call (model inference or PDF backend
        iter_pages/get_size), that thread will be abandoned after a brief wait (15s) during cleanup.
        The thread continues running until the blocking call completes, potentially holding
        resources (e.g., pypdfium2_lock).
        """
        self._page_sizes_by_no = {}
        run_id = next(self._run_seq)
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        backend = conv_res.input._backend

        # Surface the PDF outline (bookmarks/ToC) for the heading-hierarchy stage, while the
        # backend is still open. Only extracted when bookmark inference is actually enabled.
        hh_opts = self.pipeline_options.heading_hierarchy_options
        if hh_opts.enabled and hh_opts.use_bookmarks:
            conv_res._pdf_outline = backend.get_document_outline()

        expected_page_nos = self._get_expected_page_nos(conv_res)
        if not expected_page_nos:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        page_by_no: dict[int, Page] = {}
        for page_no in expected_page_nos:
            page = Page(page_no=page_no)
            conv_res.pages.append(page)
            page_by_no[page_no] = page

        total_pages: int = len(expected_page_nos)
        ctx: RunContext = self._create_run_ctx()
        for st in ctx.stages:
            st.start()

        proc = ProcessingResult(total_expected=total_pages)
        batch_size: int = 32  # drain chunk
        start_time = time.monotonic()
        timeout_exceeded = False
        producer_error: list[Exception] = []

        def _completed_page_nos() -> set[int]:
            failed_page_nos = {
                page_no for page_no, _, _ in proc.failed_pages if page_no > 0
            }
            return {page.page_no for page in proc.pages} | failed_page_nos

        def _produce_pages() -> None:
            try:
                for page_backend in self._iter_requested_page_backends(
                    backend, expected_page_nos
                ):
                    page = page_by_no.get(page_backend.page_no)
                    if page is None:
                        continue
                    page._backend = page_backend
                    try:
                        page.size = page_backend.get_size()
                        self._page_sizes_by_no[page.page_no] = page.size
                    except Exception:
                        if page_backend.is_valid():
                            raise
                    if not ctx.first_stage.input_queue.put(
                        ThreadedItem(
                            payload=page,
                            run_id=run_id,
                            page_no=page.page_no,
                            conv_res=conv_res,
                        )
                    ):
                        break
            except Exception as exc:
                producer_error.append(exc)
                _log.error("Producer failed for run %d: %s", run_id, exc, exc_info=True)
            finally:
                ctx.first_stage.input_queue.close()

        producer_thread = threading.Thread(
            target=_produce_pages, name=f"PageProducer-{run_id}", daemon=False
        )
        producer_thread.start()

        try:
            while proc.success_count + proc.failure_count < total_pages:
                # Check timeout
                if (
                    self.pipeline_options.document_timeout is not None
                    and not timeout_exceeded
                ):
                    elapsed_time = time.monotonic() - start_time
                    if elapsed_time > self.pipeline_options.document_timeout:
                        _log.warning(
                            f"Document processing time ({elapsed_time:.3f}s) "
                            f"exceeded timeout of {self.pipeline_options.document_timeout:.3f}s"
                        )
                        timeout_exceeded = True
                        ctx.timed_out_run_ids.add(run_id)
                        ctx.first_stage.input_queue.close()
                        # Break immediately - don't wait for in-flight work
                        break

                # Drain - pull whatever is ready from the output side
                out_batch = ctx.output_queue.get_batch(batch_size, timeout=0.05)
                for itm in out_batch:
                    if itm.run_id != run_id:
                        continue
                    if itm.is_failed or itm.error:
                        error = itm.error or RuntimeError("unknown error")
                        proc.failed_pages.append((itm.page_no, error, itm.failure))
                    else:
                        assert itm.payload is not None
                        proc.pages.append(itm.payload)

                # Failure safety - downstream closed early
                if not out_batch and ctx.output_queue.closed:
                    missing_page_nos = sorted(
                        set(expected_page_nos) - _completed_page_nos()
                    )
                    if missing_page_nos:
                        error = (
                            producer_error[0]
                            if producer_error
                            else RuntimeError("pipeline terminated early")
                        )
                        proc.failed_pages.extend(
                            [
                                (
                                    page_no,
                                    error,
                                    _make_error_item(
                                        component_type=DoclingComponentType.PIPELINE,
                                        module_name=self.__class__.__name__,
                                        error=error,
                                        category=FailureCategory.UNKNOWN,
                                        page_no=page_no,
                                    ),
                                )
                                for page_no in missing_page_nos
                            ]
                        )
                    break

            # Mark remaining pages as failed if timeout occurred
            if timeout_exceeded:
                missing_page_nos = sorted(
                    set(expected_page_nos) - _completed_page_nos()
                )
                for page_no in missing_page_nos:
                    error = RuntimeError("document timeout exceeded")
                    proc.failed_pages.append(
                        (
                            page_no,
                            error,
                            _make_error_item(
                                component_type=DoclingComponentType.PIPELINE,
                                module_name=self.__class__.__name__,
                                error=error,
                                category=FailureCategory.TIMEOUT,
                                page_no=page_no,
                            ),
                        )
                    )
        finally:
            for st in ctx.stages:
                st.stop()
            ctx.output_queue.close()
            producer_thread.join(timeout=15.0)
            if producer_thread.is_alive():
                _log.warning(
                    "Producer thread for run %d did not terminate within 15s and will be abandoned.",
                    run_id,
                )

        self._integrate_results(conv_res, proc, timeout_exceeded=timeout_exceeded)
        return conv_res

    # ---------------------------------------------------- integrate_results()
    def _integrate_results(
        self,
        conv_res: ConversionResult,
        proc: ProcessingResult,
        timeout_exceeded: bool = False,
    ) -> None:
        page_map = {p.page_no: p for p in proc.pages}
        # Only keep pages that successfully completed processing
        conv_res.pages = [
            page_map[p.page_no] for p in conv_res.pages if p.page_no in page_map
        ]
        # Add error details from failed pages
        for page_no, error, failure in proc.failed_pages:
            if failure is not None:
                conv_res.errors.append(failure)
                continue
            conv_res.errors.append(
                _make_error_item(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error=error or RuntimeError("Page failed to process."),
                    category=FailureCategory.UNKNOWN,
                    page_no=page_no if page_no > 0 else None,
                )
            )
        if timeout_exceeded and proc.total_expected > 0:
            # Timeout exceeded: add structured error and set PARTIAL_SUCCESS
            timeout_msg = (
                f"Pipeline stage timeout: processed {len(proc.pages)}/{proc.total_expected} pages successfully, "
                f"{len(proc.failed_pages)} pages failed or incomplete."
            )
            timeout_error = ErrorItem(
                component_type=DoclingComponentType.PIPELINE,
                module_name=self.__class__.__name__,
                error_message=timeout_msg,
                category=FailureCategory.TIMEOUT,
            )
            conv_res.errors.append(timeout_error)
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        elif proc.is_complete_failure:
            conv_res.status = ConversionStatus.FAILURE
        elif proc.is_partial_success:
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        else:
            conv_res.status = ConversionStatus.SUCCESS
        if not self.keep_images:
            for p in conv_res.pages:
                p._image_cache = {}
        for p in conv_res.pages:
            if not self.keep_backend and p._backend is not None:
                p._backend.unload()
            if not self.pipeline_options.generate_parsed_pages:
                del p.parsed_page
                p.parsed_page = None

    # ---------------------------------------------------------------- assemble
    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        elements, headers, body = [], [], []
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            for p in conv_res.pages:
                if p.assembled:
                    elements.extend(p.assembled.elements)
                    headers.extend(p.assembled.headers)
                    body.extend(p.assembled.body)
            conv_res.assembled = AssembledUnit(
                elements=elements, headers=headers, body=body
            )
            conv_res.document = self.reading_order_model(conv_res)
            conv_res.document = self.heading_hierarchy_model(conv_res)

            # Generate page images in the output
            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    assert page.image is not None
                    page_no = page.page_no
                    conv_res.document.pages[page_no].image = ImageRef.from_pil(
                        page.image, dpi=int(72 * self.pipeline_options.images_scale)
                    )

            # Generate images of the requested element types
            with warnings.catch_warnings():  # deprecated generate_table_images
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                if (
                    self.pipeline_options.generate_picture_images
                    or self.pipeline_options.generate_table_images
                ):
                    scale = self.pipeline_options.images_scale
                    for element, _level in conv_res.document.iterate_items():
                        if not isinstance(element, DocItem) or len(element.prov) == 0:
                            continue
                        if (
                            isinstance(element, PictureItem)
                            and self.pipeline_options.generate_picture_images
                        ) or (
                            isinstance(element, TableItem)
                            and self.pipeline_options.generate_table_images
                        ):
                            page_ix = element.prov[0].page_no
                            page = next(
                                (p for p in conv_res.pages if p.page_no == page_ix),
                                cast("Page", None),
                            )
                            assert page is not None
                            assert page.size is not None
                            assert page.image is not None

                            crop_bbox = (
                                element.prov[0]
                                .bbox.scaled(scale=scale)
                                .to_top_left_origin(
                                    page_height=page.size.height * scale
                                )
                            )

                            cropped_im = page.image.crop(crop_bbox.as_tuple())
                            element.image = ImageRef.from_pil(
                                cropped_im, dpi=int(72 * scale)
                            )

            # Aggregate confidence values for document:
            if len(conv_res.pages) > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message="Mean of empty slice|All-NaN slice encountered",
                    )
                    conv_res.confidence.layout_score = float(
                        np.nanmean(
                            [c.layout_score for c in conv_res.confidence.pages.values()]
                        )
                    )
                    conv_res.confidence.parse_score = float(
                        np.nanquantile(
                            [c.parse_score for c in conv_res.confidence.pages.values()],
                            q=0.1,  # parse score should relate to worst 10% of pages.
                        )
                    )
                    conv_res.confidence.table_score = float(
                        np.nanmean(
                            [c.table_score for c in conv_res.confidence.pages.values()]
                        )
                    )
                    conv_res.confidence.ocr_score = float(
                        np.nanmean(
                            [c.ocr_score for c in conv_res.confidence.pages.values()]
                        )
                    )

            # Add failed pages to DoclingDocument.pages to preserve page numbering
            # This ensures page break markers are generated for skipped/failed pages
            self._add_failed_pages_to_document(
                conv_res, expected_page_nos=self._get_expected_page_nos(conv_res)
            )

        return conv_res

    def _add_failed_pages_to_document(
        self, conv_res: ConversionResult, expected_page_nos: list[int]
    ) -> None:
        """Add failed/skipped pages to DoclingDocument.pages.

        This ensures that page break markers are correctly generated for documents
        where some pages failed to parse. Without this, export functions would not
        know about the missing pages and would generate incorrect page break counts.

        The failed pages are added with their size information (if available from
        the backend) but without any content.
        """
        if conv_res.document is None:
            return

        # Find pages that are missing from the document
        existing_page_nos = set(conv_res.document.pages.keys())
        missing_page_nos = set(expected_page_nos) - existing_page_nos

        if not missing_page_nos:
            return

        for page_no in sorted(missing_page_nos):
            size = self._page_sizes_by_no.get(page_no, Size(width=0.0, height=0.0))

            # Add the failed page to the document's pages dict
            conv_res.document.pages[page_no] = PageItem(
                page_no=page_no,
                size=size,
                image=None,
            )

        _log.debug(
            "Added %d failed/skipped pages to document: %s",
            len(missing_page_nos),
            sorted(missing_page_nos),
        )

    # ---------------------------------------------------------------- misc
    @classmethod
    def get_default_options(cls) -> ThreadedPdfPipelineOptions:
        return ThreadedPdfPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        return isinstance(backend, PdfDocumentBackend)

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        return conv_res.status

    def _unload(self, conv_res: ConversionResult) -> None:
        self._page_sizes_by_no = {}
        for p in conv_res.pages:
            if p._backend is not None:
                p._backend.unload()
        if conv_res.input._backend:
            conv_res.input._backend.unload()
