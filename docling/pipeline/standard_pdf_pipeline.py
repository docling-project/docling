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
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from docling_core.types.doc import DocItem, ImageRef, PictureItem, TableItem

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import (
    AssembledUnit,
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
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
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
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

# ──────────────────────────────────────────────────────────────────────────────
# Helper data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ThreadedItem:
    """Envelope that travels between pipeline stages."""

    payload: Optional[Page]
    run_id: int  # Unique per *execute* call, monotonic across pipeline instance
    page_no: int
    conv_res: ConversionResult
    error: Optional[Exception] = None
    is_failed: bool = False


@dataclass
class ProcessingResult:
    """Aggregated outcome of a pipeline run."""

    pages: List[Page] = field(default_factory=list)
    failed_pages: List[Tuple[int, Exception]] = field(default_factory=list)
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
    def put(self, item: ThreadedItem, timeout: Optional[float] | None = None) -> bool:
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
    def get_batch(
        self, size: int, timeout: Optional[float] | None = None
    ) -> List[ThreadedItem]:
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
            batch: List[ThreadedItem] = []
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
        postprocess: Optional[Callable[[ThreadedItem], None]] = None,
        timed_out_run_ids: Optional[set[int]] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.input_queue = ThreadedQueue(queue_max_size)
        self._outputs: list[ThreadedQueue] = []
        self._thread: Optional[threading.Thread] = None
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
            # Give thread 2s to finish naturally before abandoning
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
                    for it in items:
                        it.is_failed = True
                        it.error = RuntimeError("Page payload is None")
                    result.extend(items)
                    continue

                pages: List[Page] = [payload for _, payload in pages_with_payloads]
                processed_pages = list(self.model(good[0].conv_res, pages))  # type: ignore[arg-type]
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
                for it in items:
                    it.is_failed = True
                    it.error = exc
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
    """Pipeline stage that lazily loads PDF backends just-in-time."""

    def __init__(
        self,
        *,
        batch_timeout: float,
        queue_max_size: int,
        model: Any,
        timed_out_run_ids: Optional[set[int]] = None,
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
                result.extend(items)
                continue

            good = [i for i in items if not i.is_failed]
            if not good:
                result.extend(items)
                continue
            try:
                pages_with_payloads: list[tuple[ThreadedItem, Page]] = []
                for it in good:
                    page = it.payload
                    if page is None:
                        raise RuntimeError("Page payload is None")
                    if page._backend is None:
                        backend = it.conv_res.input._backend
                        assert isinstance(backend, PdfDocumentBackend), (
                            "Threaded pipeline only supports PdfDocumentBackend."
                        )
                        page_backend = backend.load_page(page.page_no - 1)
                        page._backend = page_backend
                        if page_backend.is_valid():
                            page.size = page_backend.get_size()
                    pages_with_payloads.append((it, page))

                pages = [payload for _, payload in pages_with_payloads]
                processed_pages = list(
                    self.model(good[0].conv_res, pages)  # type: ignore[arg-type]
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
                            page_no=good[idx].page_no,
                            conv_res=good[idx].conv_res,
                        )
                    )
            except Exception as exc:
                page_numbers = [it.page_no for it in good]
                _log.error(
                    "Stage preprocess failed for run %d, pages %s: %s",
                    rid,
                    page_numbers,
                    exc,
                    exc_info=False,  # Put to True if you want detailed exception info
                )
                for it in good:
                    it.is_failed = True
                    it.error = exc
                result.extend(items)
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
        )
        table_factory = get_table_structure_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        self.table_model = table_factory.create_instance(
            options=self.pipeline_options.table_structure_options,
            enabled=self.pipeline_options.do_table_structure,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        self.assemble_model = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # --- optional enrichment ------------------------------------------------
        self.enrichment_pipe = [
            # Code Formula Enrichment Model
            CodeFormulaModel(
                enabled=self.pipeline_options.do_code_enrichment
                or self.pipeline_options.do_formula_enrichment,
                artifacts_path=self.artifacts_path,
                options=CodeFormulaModelOptions(
                    do_code_enrichment=self.pipeline_options.do_code_enrichment,
                    do_formula_enrichment=self.pipeline_options.do_formula_enrichment,
                ),
                accelerator_options=self.pipeline_options.accelerator_options,
            ),
            *self.enrichment_pipe,
        ]

        self.keep_backend = any(
            (
                self.pipeline_options.do_formula_enrichment,
                self.pipeline_options.do_code_enrichment,
                self.pipeline_options.do_picture_classification,
                self.pipeline_options.do_picture_description,
            )
        )

    # ---------------------------------------------------------------- helpers
    def _make_ocr_model(self, art_path: Optional[Path]) -> Any:
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
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Stream-build the document while interleaving producer and consumer work.

        Note: If a worker thread gets stuck in a blocking call (model inference or PDF backend
        load_page/get_size), that thread will be abandoned after a brief wait (15s) during cleanup.
        The thread continues running until the blocking call completes, potentially holding
        resources (e.g., pypdfium2_lock).
        """
        run_id = next(self._run_seq)
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)

        # Collect page placeholders; backends are loaded lazily in preprocess stage
        start_page, end_page = conv_res.input.limits.page_range
        pages: list[Page] = []
        for i in range(conv_res.input.page_count):
            if start_page - 1 <= i <= end_page - 1:
                page = Page(page_no=i + 1)
                conv_res.pages.append(page)
                pages.append(page)

        if not pages:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        total_pages: int = len(pages)
        ctx: RunContext = self._create_run_ctx()
        for st in ctx.stages:
            st.start()

        proc = ProcessingResult(total_expected=total_pages)
        fed_idx: int = 0  # number of pages successfully queued
        batch_size: int = 32  # drain chunk
        start_time = time.monotonic()
        timeout_exceeded = False
        input_queue_closed = False
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
                        if not input_queue_closed:
                            ctx.first_stage.input_queue.close()
                            input_queue_closed = True
                        # Break immediately - don't wait for in-flight work
                        break

                # 1) feed - try to enqueue until the first queue is full
                if not input_queue_closed:
                    while fed_idx < total_pages:
                        ok = ctx.first_stage.input_queue.put(
                            ThreadedItem(
                                payload=pages[fed_idx],
                                run_id=run_id,
                                page_no=pages[fed_idx].page_no,
                                conv_res=conv_res,
                            ),
                            timeout=0.0,  # non-blocking try-put
                        )
                        if ok:
                            fed_idx += 1
                            if fed_idx == total_pages:
                                ctx.first_stage.input_queue.close()
                                input_queue_closed = True
                        else:  # queue full - switch to draining
                            break

                # 2) drain - pull whatever is ready from the output side
                out_batch = ctx.output_queue.get_batch(batch_size, timeout=0.05)
                for itm in out_batch:
                    if itm.run_id != run_id:
                        continue
                    if itm.is_failed or itm.error:
                        proc.failed_pages.append(
                            (itm.page_no, itm.error or RuntimeError("unknown error"))
                        )
                    else:
                        assert itm.payload is not None
                        proc.pages.append(itm.payload)

                # 3) failure safety - downstream closed early
                if not out_batch and ctx.output_queue.closed:
                    missing = total_pages - (proc.success_count + proc.failure_count)
                    if missing > 0:
                        proc.failed_pages.extend(
                            [(-1, RuntimeError("pipeline terminated early"))] * missing
                        )
                    break

            # Mark remaining pages as failed if timeout occurred
            if timeout_exceeded:
                completed_page_nos = {p.page_no for p in proc.pages} | {
                    fp for fp, _ in proc.failed_pages
                }
                for page in pages[fed_idx:]:
                    if page.page_no not in completed_page_nos:
                        proc.failed_pages.append(
                            (page.page_no, RuntimeError("document timeout exceeded"))
                        )
        finally:
            for st in ctx.stages:
                st.stop()
            ctx.output_queue.close()

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
        for page_no, error in proc.failed_pages:
            page_label = f"Page {page_no}" if page_no > 0 else "Unknown page"
            error_msg = str(error) if error else ""
            error_item = ErrorItem(
                component_type=DoclingComponentType.PIPELINE,
                module_name=self.__class__.__name__,
                error_message=f"{page_label}: {error_msg}" if error_msg else page_label,
            )
            conv_res.errors.append(error_item)
        if timeout_exceeded and proc.total_expected > 0:
            # Timeout exceeded: set PARTIAL_SUCCESS if any pages were attempted
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

        return conv_res

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
        for p in conv_res.pages:
            if p._backend is not None:
                p._backend.unload()
        if conv_res.input._backend:
            conv_res.input._backend.unload()
