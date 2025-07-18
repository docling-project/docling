# threaded_standard_pdf_pipeline.py
# pylint: disable=too-many-lines
"""
Thread-safe, multi-threaded PDF pipeline.

Key points
----------
* Heavy models are initialised once per pipeline instance and safely reused across threads.
* Every `execute()` call creates its own `RunContext` with fresh queues and worker threads
  so concurrent executions never share mutable state.
* Back-pressure remains intact because every `ThreadedQueue` is bounded (`max_size` from
  pipeline options). When a downstream queue is full, upstream stages block on a condition
  variable instead of busy-polling.
* Pipeline termination relies on queue closing:
  - The producer thread closes its output queue after the last real page is queued.
  - Each stage propagates that closure downstream by closing its own output queues
    once it has drained its input queue and detected that it is closed.
  - The collector finishes when it has received either
      * all expected pages, or
      * the output queue is closed and empty.
* Per-page errors are captured, propagated, and turned into
  `ConversionStatus.PARTIAL_SUCCESS` when appropriate.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from docling_core.types.doc import ImageRef, PictureItem, TableItem

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
    """Item that moves between pipeline stages."""

    payload: Optional[Page]
    conv_res_id: int
    conv_res: ConversionResult
    page_no: int = -1
    error: Optional[Exception] = None
    is_failed: bool = False

    def __post_init__(self) -> None:
        if self.page_no == -1 and self.payload is not None:
            self.page_no = self.payload.page_no


@dataclass
class ProcessingResult:
    """Aggregate of processed and failed pages."""

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
        return self.success_count > 0 and self.failure_count > 0

    @property
    def is_complete_failure(self) -> bool:
        return self.success_count == 0 and self.failure_count > 0


@dataclass
class ThreadedQueue:
    """Bounded queue with explicit back-pressure and close semantics."""

    max_size: int = 512
    items: deque[ThreadedItem] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    not_full: threading.Condition = field(init=False)
    not_empty: threading.Condition = field(init=False)
    closed: bool = False

    def __post_init__(self) -> None:
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    # ------------------------------------------------------------------  put()
    def put(self, item: ThreadedItem, timeout: Optional[float] = None) -> bool:
        """Block until the queue has room or is closed. Returns False if closed."""
        with self.not_full:
            if self.closed:
                return False

            start = time.monotonic()
            while len(self.items) >= self.max_size and not self.closed:
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        return False
                    self.not_full.wait(remaining)
                else:
                    self.not_full.wait()

            if self.closed:
                return False

            self.items.append(item)
            self.not_empty.notify()
            return True

    # ------------------------------------------------------------  get_batch()
    def get_batch(
        self, batch_size: int, timeout: Optional[float] = None
    ) -> List[ThreadedItem]:
        """Return up to `batch_size` items; may return fewer on timeout/closure."""
        with self.not_empty:
            start = time.monotonic()
            while not self.items and not self.closed:
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        return []
                    self.not_empty.wait(remaining)
                else:
                    self.not_empty.wait()

            batch: List[ThreadedItem] = []
            while len(batch) < batch_size and self.items:
                batch.append(self.items.popleft())

            if batch:
                self.not_full.notify_all()
            return batch

    # ------------------------------------------------------------------  close
    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    # ---------------------------------------------------------------  cleanup
    def cleanup(self) -> None:
        with self.lock:
            self.items.clear()
            self.closed = True


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stage
# ──────────────────────────────────────────────────────────────────────────────


class ThreadedPipelineStage:
    """A processing stage with its own thread, batch size and timeouts."""

    def __init__(
        self,
        name: str,
        model: Any,
        batch_size: int,
        batch_timeout: float,
        queue_max_size: int,
    ) -> None:
        self.name: str = name
        self.model: Any = model
        self.batch_size: int = batch_size
        self.batch_timeout: float = batch_timeout
        self.input_queue: ThreadedQueue = ThreadedQueue(max_size=queue_max_size)
        self.output_queues: List[ThreadedQueue] = []
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ----------------------------------------------------------------  wiring
    def add_output_queue(self, queue: ThreadedQueue) -> None:
        self.output_queues.append(queue)

    # -----------------------------------------------------------  lifecycle
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
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                _log.warning("Stage %s thread did not shut down in time", self.name)

    def cleanup(self) -> None:
        self.input_queue.cleanup()
        for q in self.output_queues:
            q.cleanup()

    # ------------------------------------------------------------------  _run
    def _run(self) -> None:
        try:
            while self._running:
                batch = self.input_queue.get_batch(
                    self.batch_size, timeout=self.batch_timeout
                )

                # Exit when upstream is finished and queue drained
                if not batch and self.input_queue.closed:
                    break

                processed_items = self._process_batch(batch)
                self._send_to_outputs(processed_items)
        except Exception as exc:  # pragma: no cover - safety net
            _log.error("Fatal error in stage %s: %s", self.name, exc, exc_info=True)
        finally:
            # Propagate closure to downstream
            for q in self.output_queues:
                q.close()

    # -------------------------------------------------------  _process_batch
    def _process_batch(self, batch: Sequence[ThreadedItem]) -> List[ThreadedItem]:
        grouped: dict[int, List[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            grouped[itm.conv_res_id].append(itm)

        out: List[ThreadedItem] = []
        for conv_res_id, items in grouped.items():
            try:
                valid = [it for it in items if not it.is_failed]
                fail = [it for it in items if it.is_failed]

                if valid:
                    conv_res = valid[0].conv_res
                    pages = [it.payload for it in valid]  # type: ignore[arg-type]
                    assert all(p is not None for p in pages)
                    processed_pages = list(self.model(conv_res, pages))  # type: ignore[arg-type]

                    for idx, page in enumerate(processed_pages):
                        out.append(
                            ThreadedItem(
                                payload=page,
                                conv_res_id=conv_res_id,
                                conv_res=conv_res,
                                page_no=valid[idx].page_no,
                            )
                        )
                out.extend(fail)
            except Exception as exc:
                _log.error(
                    "Model %s failed for doc %s: %s", self.name, conv_res_id, exc
                )
                for it in items:
                    it.is_failed = True
                    it.error = exc
                    out.append(it)

        return out

    # ------------------------------------------------------  _send_to_outputs
    def _send_to_outputs(self, items: Sequence[ThreadedItem]) -> None:
        for item in items:
            for q in self.output_queues:
                if not q.put(item, timeout=None):
                    _log.error("Queue closed while writing from %s", self.name)


# ──────────────────────────────────────────────────────────────────────────────
# Run context (per-execute mutable state)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RunContext:
    preprocess_stage: ThreadedPipelineStage
    stages: List[ThreadedPipelineStage]
    output_queue: ThreadedQueue


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────


class ThreadedStandardPdfPipeline(BasePipeline):
    """Thread-safe PDF pipeline with per-run isolation and queue-closing protocol."""

    # ----------------------------------------------------------------  ctor
    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: ThreadedPdfPipelineOptions = pipeline_options

        # Flags set by enrichment logic
        self.keep_backend: bool = False
        self.keep_images: bool = False

        # Placeholders for heavy models
        self.preprocessing_model: PagePreprocessingModel
        self.ocr_model: Any
        self.layout_model: LayoutModel
        self.table_model: TableStructureModel
        self.assemble_model: PageAssembleModel
        self.reading_order_model: ReadingOrderModel

        self._initialize_models()

        # Weak tracking for stage-internal look-ups
        self._document_tracker: weakref.WeakValueDictionary[int, ConversionResult] = (
            weakref.WeakValueDictionary()
        )
        self._document_lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────────────
    # Model helpers
    # ────────────────────────────────────────────────────────────────────────

    def _get_artifacts_path(self) -> Optional[Path]:
        if self.pipeline_options.artifacts_path:
            path = Path(self.pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path:
            path = Path(settings.artifacts_path).expanduser()
        else:
            path = None

        if path is not None and not path.is_dir():
            raise RuntimeError(
                f"{path=} is not a directory containing the required models."
            )
        return path

    def _get_ocr_model(self, artifacts_path: Optional[Path]) -> Any:
        factory = get_ocr_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    def _get_picture_description_model(
        self,
        artifacts_path: Optional[Path],
    ) -> Optional[PictureDescriptionBaseModel]:
        factory = get_picture_description_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.picture_description_options,
            enabled=self.pipeline_options.do_picture_description,
            enable_remote_services=self.pipeline_options.enable_remote_services,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Heavy-model initialisation
    # ────────────────────────────────────────────────────────────────────────

    def _initialize_models(self) -> None:
        artifacts_path = self._get_artifacts_path()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        self.ocr_model = self._get_ocr_model(artifacts_path)
        self.layout_model = LayoutModel(
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            options=self.pipeline_options.layout_options,
        )
        self.table_model = TableStructureModel(
            enabled=self.pipeline_options.do_table_structure,
            artifacts_path=artifacts_path,
            options=self.pipeline_options.table_structure_options,
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        self.assemble_model = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # Optional enrichment
        self.enrichment_pipe = []
        code_formula = CodeFormulaModel(
            enabled=self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_formula_enrichment,
            artifacts_path=artifacts_path,
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
            artifacts_path=artifacts_path,
            options=DocumentPictureClassifierOptions(),
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        if picture_classifier.enabled:
            self.enrichment_pipe.append(picture_classifier)

        picture_descr = self._get_picture_description_model(artifacts_path)
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

    # ────────────────────────────────────────────────────────────────────────
    # Run context creation
    # ────────────────────────────────────────────────────────────────────────

    def _create_run(self) -> RunContext:
        opts = self.pipeline_options

        preprocess = ThreadedPipelineStage(
            "preprocess",
            self.preprocessing_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        ocr = ThreadedPipelineStage(
            "ocr",
            self.ocr_model,
            batch_size=opts.ocr_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        layout = ThreadedPipelineStage(
            "layout",
            self.layout_model,
            batch_size=opts.layout_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        table = ThreadedPipelineStage(
            "table",
            self.table_model,
            batch_size=opts.table_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )
        assemble = ThreadedPipelineStage(
            "assemble",
            self.assemble_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # Wiring
        output_queue = ThreadedQueue(max_size=opts.queue_max_size)
        preprocess.add_output_queue(ocr.input_queue)
        ocr.add_output_queue(layout.input_queue)
        layout.add_output_queue(table.input_queue)
        table.add_output_queue(assemble.input_queue)
        assemble.add_output_queue(output_queue)

        stages = [preprocess, ocr, layout, table, assemble]
        return RunContext(
            preprocess_stage=preprocess, stages=stages, output_queue=output_queue
        )

    # ────────────────────────────────────────────────────────────────────────
    # Document build
    # ────────────────────────────────────────────────────────────────────────

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        if not isinstance(conv_res.input._backend, PdfDocumentBackend):
            raise RuntimeError("Input backend must be PdfDocumentBackend")

        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            # ---------------------------------------------------------------- pages
            start_page, end_page = conv_res.input.limits.page_range
            pages_to_process: List[Page] = []
            for i in range(conv_res.input.page_count):
                if start_page - 1 <= i <= end_page - 1:
                    page = Page(page_no=i)
                    conv_res.pages.append(page)
                    page._backend = conv_res.input._backend.load_page(i)
                    if page._backend and page._backend.is_valid():
                        page.size = page._backend.get_size()
                        pages_to_process.append(page)

            if not pages_to_process:
                conv_res.status = ConversionStatus.FAILURE
                return conv_res

            # ---------------------------------------------------------------- run ctx
            ctx = self._create_run()

            # Weak-map registration (for potential cross-stage look-ups)
            with self._document_lock:
                self._document_tracker[id(conv_res)] = conv_res

            for stage in ctx.stages:
                stage.start()

            try:
                self._feed_pipeline(ctx.preprocess_stage, pages_to_process, conv_res)
                result = self._collect_results_with_recovery(
                    ctx, conv_res, len(pages_to_process)
                )
                self._update_document_with_results(conv_res, result)
            finally:
                for stage in ctx.stages:
                    stage.stop()
                    stage.cleanup()
                ctx.output_queue.cleanup()
                with self._document_lock:
                    self._document_tracker.pop(id(conv_res), None)

        return conv_res

    # ────────────────────────────────────────────────────────────────────────
    # Feed pages
    # ────────────────────────────────────────────────────────────────────────

    def _feed_pipeline(
        self,
        preprocess_stage: ThreadedPipelineStage,
        pages: Sequence[Page],
        conv_res: ConversionResult,
    ) -> None:
        for pg in pages:
            ok = preprocess_stage.input_queue.put(
                ThreadedItem(
                    payload=pg,
                    conv_res_id=id(conv_res),
                    conv_res=conv_res,
                    page_no=pg.page_no,
                ),
                timeout=None,
            )
            if not ok:
                raise RuntimeError(
                    "Preprocess queue closed unexpectedly while feeding pages"
                )

        # Upstream finished: close queue (no sentinel needed)
        preprocess_stage.input_queue.close()

    # ────────────────────────────────────────────────────────────────────────
    # Collect results
    # ────────────────────────────────────────────────────────────────────────

    def _collect_results_with_recovery(
        self,
        ctx: RunContext,
        conv_res: ConversionResult,
        expected_count: int,
    ) -> ProcessingResult:
        result = ProcessingResult(total_expected=expected_count)

        while True:
            batch = ctx.output_queue.get_batch(batch_size=expected_count, timeout=None)
            if not batch and ctx.output_queue.closed:
                break

            for item in batch:
                if item.conv_res_id != id(conv_res):
                    # In per-run isolation this cannot happen
                    continue
                if item.is_failed or item.error:
                    result.failed_pages.append(
                        (item.page_no, item.error or Exception("Unknown error"))
                    )
                else:
                    assert item.payload is not None
                    result.pages.append(item.payload)

            # Terminate when all pages accounted for
            if (result.success_count + result.failure_count) >= expected_count:
                break

        return result

    # ────────────────────────────────────────────────────────────────────────
    # Update ConversionResult
    # ────────────────────────────────────────────────────────────────────────

    def _update_document_with_results(
        self, conv_res: ConversionResult, proc: ProcessingResult
    ) -> None:
        page_map = {p.page_no: p for p in proc.pages}
        new_pages: List[Page] = []
        for page in conv_res.pages:
            if page.page_no in page_map:
                new_pages.append(page_map[page.page_no])
            elif not any(fp_no == page.page_no for fp_no, _ in proc.failed_pages):
                new_pages.append(page)
        conv_res.pages = new_pages

        if proc.is_partial_success:
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        elif proc.is_complete_failure:
            conv_res.status = ConversionStatus.FAILURE
        else:
            conv_res.status = ConversionStatus.SUCCESS

        if not self.keep_images:
            for p in conv_res.pages:
                p._image_cache = {}
        if not self.keep_backend:
            for p in conv_res.pages:
                if p._backend is not None:
                    p._backend.unload()

    # ────────────────────────────────────────────────────────────────────────
    # Assemble / enrich
    # ────────────────────────────────────────────────────────────────────────

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        all_elements, all_headers, all_body = [], [], []
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            for p in conv_res.pages:
                if p.assembled:
                    all_elements.extend(p.assembled.elements)
                    all_headers.extend(p.assembled.headers)
                    all_body.extend(p.assembled.body)

            conv_res.assembled = AssembledUnit(
                elements=all_elements, headers=all_headers, body=all_body
            )
            conv_res.document = self.reading_order_model(conv_res)
        return conv_res

    # ────────────────────────────────────────────────────────────────────────
    # Base overrides
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_default_options(cls) -> ThreadedPdfPipelineOptions:
        return ThreadedPdfPipelineOptions()  # type: ignore[call-arg]

    @classmethod
    def is_backend_supported(cls, backend) -> bool:  # type: ignore[override]
        return isinstance(backend, PdfDocumentBackend)

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        return conv_res.status

    def _unload(self, conv_res: ConversionResult) -> None:
        return
