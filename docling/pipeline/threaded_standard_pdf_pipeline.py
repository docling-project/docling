# threaded_standard_pdf_pipeline.py
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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import AssembledUnit, ConversionStatus, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings
from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions
from docling.models.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.factories import get_ocr_factory, get_picture_description_factory
from docling.models.layout_model import LayoutModel
from docling.models.page_assemble_model import PageAssembleModel, PageAssembleOptions
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from docling.models.table_structure_model import TableStructureModel
from docling.pipeline.base_pipeline import BasePipeline
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
    ) -> None:
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.input_queue = ThreadedQueue(queue_max_size)
        self._outputs: list[ThreadedQueue] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ---------------------------------------------------------------- wiring
    def add_output_queue(self, q: ThreadedQueue) -> None:
        self._outputs.append(q)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"Stage-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.input_queue.close()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                _log.warning("Stage %s did not terminate cleanly within 30s", self.name)

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
                _log.error("Stage %s failed for run %d: %s", self.name, rid, exc)
                for it in items:
                    it.is_failed = True
                    it.error = exc
                result.extend(items)
        return result

    # -------------------------------------------------------------- _emit()
    def _emit(self, items: Iterable[ThreadedItem]) -> None:
        for item in items:
            for q in self._outputs:
                if not q.put(item):
                    _log.error("Output queue closed while emitting from %s", self.name)


@dataclass
class RunContext:
    """Wiring for a single *execute* call."""

    stages: list[ThreadedPipelineStage]
    first_stage: ThreadedPipelineStage
    output_queue: ThreadedQueue


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────


class ThreadedStandardPdfPipeline(BasePipeline):
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
        art_path = self._resolve_artifacts_path()
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
        self.layout_model = LayoutModel(
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            options=self.pipeline_options.layout_options,
        )
        self.table_model = TableStructureModel(
            enabled=self.pipeline_options.do_table_structure,
            artifacts_path=art_path,
            options=self.pipeline_options.table_structure_options,
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        self.assemble_model = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # --- optional enrichment ------------------------------------------------
        self.enrichment_pipe = []
        code_formula = CodeFormulaModel(
            enabled=self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_formula_enrichment,
            artifacts_path=art_path,
            options=CodeFormulaModelOptions(
                do_code_enrichment=self.pipeline_options.do_code_enrichment,
                do_formula_enrichment=self.pipeline_options.do_formula_enrichment,
            ),
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        if code_formula.enabled:
            self.enrichment_pipe.append(code_formula)

        picture_classifier = DocumentPictureClassifier(
            enabled=self.pipeline_options.do_picture_classification,
            artifacts_path=art_path,
            options=DocumentPictureClassifierOptions(),
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        if picture_classifier.enabled:
            self.enrichment_pipe.append(picture_classifier)

        picture_descr = self._make_picture_description_model(art_path)
        if picture_descr and picture_descr.enabled:
            self.enrichment_pipe.append(picture_descr)

        self.keep_backend = any(
            (
                self.pipeline_options.do_formula_enrichment,
                self.pipeline_options.do_code_enrichment,
                self.pipeline_options.do_picture_classification,
                self.pipeline_options.do_picture_description,
            )
        )

    # ---------------------------------------------------------------- helpers
    def _resolve_artifacts_path(self) -> Optional[Path]:
        if self.pipeline_options.artifacts_path:
            p = Path(self.pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path:
            p = Path(settings.artifacts_path).expanduser()
        else:
            return None
        if not p.is_dir():
            raise RuntimeError(
                f"{p} does not exist or is not a directory containing the required models"
            )
        return p

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

    def _make_picture_description_model(
        self, art_path: Optional[Path]
    ) -> Optional[PictureDescriptionBaseModel]:
        factory = get_picture_description_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.picture_description_options,
            enabled=self.pipeline_options.do_picture_description,
            enable_remote_services=self.pipeline_options.enable_remote_services,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Build - thread pipeline
    # ────────────────────────────────────────────────────────────────────────

    def _create_run_ctx(self) -> RunContext:
        opts = self.pipeline_options
        preprocess = ThreadedPipelineStage(
            name="preprocess",
            model=self.preprocessing_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        ocr = ThreadedPipelineStage(
            name="ocr",
            model=self.ocr_model,
            batch_size=opts.ocr_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        layout = ThreadedPipelineStage(
            name="layout",
            model=self.layout_model,
            batch_size=opts.layout_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        table = ThreadedPipelineStage(
            name="table",
            model=self.table_model,
            batch_size=opts.table_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        assemble = ThreadedPipelineStage(
            name="assemble",
            model=self.assemble_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # wire stages
        output_q = ThreadedQueue(opts.queue_max_size)
        preprocess.add_output_queue(ocr.input_queue)
        ocr.add_output_queue(layout.input_queue)
        layout.add_output_queue(table.input_queue)
        table.add_output_queue(assemble.input_queue)
        assemble.add_output_queue(output_q)

        stages = [preprocess, ocr, layout, table, assemble]
        return RunContext(stages=stages, first_stage=preprocess, output_queue=output_q)

    # --------------------------------------------------------------------- build
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Stream-build the document while interleaving producer and consumer work."""
        run_id = next(self._run_seq)
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        backend = conv_res.input._backend

        # preload & initialise pages -------------------------------------------------------------
        start_page, end_page = conv_res.input.limits.page_range
        pages: list[Page] = []
        for i in range(conv_res.input.page_count):
            if start_page - 1 <= i <= end_page - 1:
                page = Page(page_no=i)
                page._backend = backend.load_page(i)
                if page._backend and page._backend.is_valid():
                    page.size = page._backend.get_size()
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
        try:
            while proc.success_count + proc.failure_count < total_pages:
                # 1) feed - try to enqueue until the first queue is full
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

                # 3) failure safety - downstream closed early -> mark missing pages failed
                if not out_batch and ctx.output_queue.closed:
                    missing = total_pages - (proc.success_count + proc.failure_count)
                    if missing > 0:
                        proc.failed_pages.extend(
                            [(-1, RuntimeError("pipeline terminated early"))] * missing
                        )
                    break
        finally:
            for st in ctx.stages:
                st.stop()
            ctx.output_queue.close()

        self._integrate_results(conv_res, proc)
        return conv_res

    # ---------------------------------------------------- integrate_results()
    def _integrate_results(
        self, conv_res: ConversionResult, proc: ProcessingResult
    ) -> None:
        page_map = {p.page_no: p for p in proc.pages}
        conv_res.pages = [
            page_map.get(p.page_no, p)
            for p in conv_res.pages
            if p.page_no in page_map
            or not any(fp == p.page_no for fp, _ in proc.failed_pages)
        ]
        if proc.is_complete_failure:
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
