import logging
import threading
import time
import warnings
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, cast

import numpy as np
from docling_core.types.doc import DocItem, ImageRef, PictureItem, TableItem

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


@dataclass
class ThreadedItem:
    """Item flowing through the threaded pipeline with document context"""

    payload: Page
    conv_res_id: int
    conv_res: ConversionResult
    page_no: int = -1
    error: Optional[Exception] = None
    is_failed: bool = False

    def __post_init__(self):
        """Ensure proper initialization of page number"""
        if self.page_no == -1 and isinstance(self.payload, Page):
            self.page_no = self.payload.page_no


@dataclass
class ProcessingResult:
    """Result of processing with error tracking for partial results"""

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
    """Thread-safe queue with backpressure control and memory management"""

    max_size: int = 100
    items: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    not_full: threading.Condition = field(init=False)
    not_empty: threading.Condition = field(init=False)
    closed: bool = False

    def __post_init__(self):
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item: ThreadedItem, timeout: Optional[float] = None) -> bool:
        """Put item with backpressure control"""
        with self.not_full:
            if self.closed:
                return False

            start_time = time.time()
            while len(self.items) >= self.max_size and not self.closed:
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
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

    def get_batch(
        self, batch_size: int, timeout: Optional[float] = None
    ) -> List[ThreadedItem]:
        """Get a batch of items"""
        with self.not_empty:
            start_time = time.time()

            # Wait for at least one item
            while len(self.items) == 0 and not self.closed:
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return []
                    self.not_empty.wait(remaining)
                else:
                    self.not_empty.wait()

            # Collect batch
            batch: List[ThreadedItem] = []
            while len(batch) < batch_size and len(self.items) > 0:
                batch.append(self.items.popleft())

            if batch:
                self.not_full.notify_all()

            return batch

    def close(self):
        """Close the queue and wake up waiting threads"""
        with self.lock:
            self.closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def is_empty(self) -> bool:
        with self.lock:
            return len(self.items) == 0

    def size(self) -> int:
        with self.lock:
            return len(self.items)

    def cleanup(self):
        """Clean up resources and clear items"""
        with self.lock:
            self.items.clear()
            self.closed = True


class ThreadedPipelineStage:
    """A pipeline stage that processes items using dedicated threads"""

    def __init__(
        self,
        name: str,
        model: Any,
        batch_size: int,
        batch_timeout: float,
        queue_max_size: int,
    ):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.input_queue = ThreadedQueue(max_size=queue_max_size)
        self.output_queues: List[ThreadedQueue] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def add_output_queue(self, queue: ThreadedQueue):
        """Connect this stage to an output queue"""
        self.output_queues.append(queue)

    def start(self):
        """Start the stage processing thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, name=f"Stage-{self.name}")
            self.thread.daemon = False  # Ensure proper shutdown
            self.thread.start()

    def stop(self):
        """Stop the stage processing"""
        self.running = False
        self.input_queue.close()
        if self.thread:
            self.thread.join(timeout=30.0)  # Reasonable timeout for shutdown
            if self.thread.is_alive():
                _log.warning(f"Stage {self.name} thread did not shutdown gracefully")

    def _run(self):
        """Main processing loop for the stage"""
        try:
            while self.running:
                batch = self.input_queue.get_batch(
                    self.batch_size, timeout=self.batch_timeout
                )

                if not batch and self.input_queue.closed:
                    break

                if batch:
                    try:
                        processed_items = self._process_batch(batch)
                        self._send_to_outputs(processed_items)
                    except Exception as e:
                        _log.error(f"Error in stage {self.name}: {e}", exc_info=True)
                        # Send failed items downstream for partial processing
                        failed_items = []
                        for item in batch:
                            item.is_failed = True
                            item.error = e
                            failed_items.append(item)
                        self._send_to_outputs(failed_items)

        except Exception as e:
            _log.error(f"Fatal error in stage {self.name}: {e}", exc_info=True)
        finally:
            # Close output queues when done
            for queue in self.output_queues:
                queue.close()

    def _process_batch(self, batch: List[ThreadedItem]) -> List[ThreadedItem]:
        """Process a batch through the model with error handling"""
        # Group by document to maintain document integrity
        grouped_by_doc = defaultdict(list)
        for item in batch:
            grouped_by_doc[item.conv_res_id].append(item)

        processed_items = []
        for conv_res_id, items in grouped_by_doc.items():
            try:
                # Filter out already failed items
                valid_items = [item for item in items if not item.is_failed]
                failed_items = [item for item in items if item.is_failed]

                if valid_items:
                    conv_res = valid_items[0].conv_res
                    pages = [item.payload for item in valid_items]

                    # Process through model
                    processed_pages = list(self.model(conv_res, pages))

                    # Re-wrap processed pages
                    for i, page in enumerate(processed_pages):
                        processed_items.append(
                            ThreadedItem(
                                payload=page,
                                conv_res_id=valid_items[i].conv_res_id,
                                conv_res=valid_items[i].conv_res,
                                page_no=valid_items[i].page_no,
                            )
                        )

                # Pass through failed items for downstream handling
                processed_items.extend(failed_items)

            except Exception as e:
                _log.error(f"Model {self.name} failed for document {conv_res_id}: {e}")
                # Mark all items as failed but continue processing
                for item in items:
                    item.is_failed = True
                    item.error = e
                    processed_items.append(item)

        return processed_items

    def _send_to_outputs(self, items: List[ThreadedItem]):
        """Send processed items to output queues"""
        for item in items:
            for queue in self.output_queues:
                # Use timeout to prevent blocking indefinitely
                if not queue.put(item, timeout=5.0):
                    _log.warning(
                        f"Failed to send item from {self.name} due to backpressure"
                    )

    def cleanup(self):
        """Clean up stage resources"""
        if self.input_queue:
            self.input_queue.cleanup()
        for queue in self.output_queues:
            queue.cleanup()


class ThreadedStandardPdfPipeline(BasePipeline):
    """
    A threaded pipeline implementation that processes pages through
    dedicated stage threads with batching and backpressure control.
    """

    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: ThreadedPdfPipelineOptions = pipeline_options

        # Initialize attributes with proper type annotations
        self.keep_backend: bool = False
        self.keep_images: bool = False

        # Model attributes - will be initialized in _initialize_models
        self.preprocessing_model: PagePreprocessingModel
        self.ocr_model: Any  # OCR models have different base types from factory
        self.layout_model: LayoutModel
        self.table_model: TableStructureModel
        self.assemble_model: PageAssembleModel
        self.reading_order_model: ReadingOrderModel

        self._initialize_models()
        self._setup_pipeline()

        # Use weak references for memory management
        self._document_tracker: weakref.WeakValueDictionary[int, ConversionResult] = (
            weakref.WeakValueDictionary()
        )
        self._document_lock = threading.Lock()

    def _get_artifacts_path(self) -> Optional[Path]:
        """Get artifacts path from options or settings"""
        artifacts_path = None
        if self.pipeline_options.artifacts_path is not None:
            artifacts_path = Path(self.pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )
        return artifacts_path

    def _get_ocr_model(self, artifacts_path: Optional[Path] = None):
        """Get OCR model instance"""
        factory = get_ocr_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    def _get_picture_description_model(self, artifacts_path: Optional[Path] = None):
        """Get picture description model instance"""
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

    def _initialize_models(self):
        """Initialize all pipeline models"""
        artifacts_path = self._get_artifacts_path()

        # Check if we need to keep images for processing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.keep_images = (
                self.pipeline_options.generate_page_images
                or self.pipeline_options.generate_picture_images
                or self.pipeline_options.generate_table_images
            )

        self.preprocessing_model = PagePreprocessingModel(
            options=PagePreprocessingOptions(
                images_scale=self.pipeline_options.images_scale,
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

        # Reading order and enrichment models
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # Initialize enrichment models and add only enabled ones to enrichment_pipe
        self.enrichment_pipe = []

        # Code Formula Enrichment Model
        code_formula_model = CodeFormulaModel(
            enabled=self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_formula_enrichment,
            artifacts_path=artifacts_path,
            options=CodeFormulaModelOptions(
                do_code_enrichment=self.pipeline_options.do_code_enrichment,
                do_formula_enrichment=self.pipeline_options.do_formula_enrichment,
            ),
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        if code_formula_model.enabled:
            self.enrichment_pipe.append(code_formula_model)

        # Document Picture Classifier
        picture_classifier = DocumentPictureClassifier(
            enabled=self.pipeline_options.do_picture_classification,
            artifacts_path=artifacts_path,
            options=DocumentPictureClassifierOptions(),
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        if picture_classifier.enabled:
            self.enrichment_pipe.append(picture_classifier)

        # Picture description model
        picture_description_model = self._get_picture_description_model(artifacts_path)
        if picture_description_model is not None and picture_description_model.enabled:
            self.enrichment_pipe.append(picture_description_model)

        # Determine if we need to keep backend for enrichment
        if (
            self.pipeline_options.do_formula_enrichment
            or self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_picture_classification
            or self.pipeline_options.do_picture_description
        ):
            self.keep_backend = True

    def _setup_pipeline(self):
        """Setup the pipeline stages and connections with proper typing"""
        # Use pipeline options directly - they have proper defaults
        opts = self.pipeline_options

        # Create pipeline stages
        self.preprocess_stage = ThreadedPipelineStage(
            "preprocess",
            self.preprocessing_model,
            1,
            opts.batch_timeout_seconds,
            opts.queue_max_size,
        )
        self.ocr_stage = ThreadedPipelineStage(
            "ocr",
            self.ocr_model,
            opts.ocr_batch_size,
            opts.batch_timeout_seconds,
            opts.queue_max_size,
        )
        self.layout_stage = ThreadedPipelineStage(
            "layout",
            self.layout_model,
            opts.layout_batch_size,
            opts.batch_timeout_seconds,
            opts.queue_max_size,
        )
        self.table_stage = ThreadedPipelineStage(
            "table",
            self.table_model,
            opts.table_batch_size,
            opts.batch_timeout_seconds,
            opts.queue_max_size,
        )
        self.assemble_stage = ThreadedPipelineStage(
            "assemble",
            self.assemble_model,
            1,
            opts.batch_timeout_seconds,
            opts.queue_max_size,
        )

        # Create output queue for final results
        self.output_queue = ThreadedQueue(max_size=opts.queue_max_size)

        # Connect stages in pipeline order
        self.preprocess_stage.add_output_queue(self.ocr_stage.input_queue)
        self.ocr_stage.add_output_queue(self.layout_stage.input_queue)
        self.layout_stage.add_output_queue(self.table_stage.input_queue)
        self.table_stage.add_output_queue(self.assemble_stage.input_queue)
        self.assemble_stage.add_output_queue(self.output_queue)

        self.stages = [
            self.preprocess_stage,
            self.ocr_stage,
            self.layout_stage,
            self.table_stage,
            self.assemble_stage,
        ]

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document by processing pages through threaded pipeline"""
        if not isinstance(conv_res.input._backend, PdfDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(conv_res.input._backend).__name__} for {conv_res.input.file} is not a PDF backend."
            )

        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            # Initialize pages
            start_page, end_page = conv_res.input.limits.page_range
            pages_to_process = []

            for i in range(conv_res.input.page_count):
                if (start_page - 1) <= i <= (end_page - 1):
                    page = Page(page_no=i)
                    conv_res.pages.append(page)

                    # Initialize page backend
                    page._backend = conv_res.input._backend.load_page(i)
                    if page._backend and page._backend.is_valid():
                        page.size = page._backend.get_size()
                        pages_to_process.append(page)

            if not pages_to_process:
                conv_res.status = ConversionStatus.FAILURE
                return conv_res

            # Register document for tracking with weak reference
            doc_id = id(conv_res)
            with self._document_lock:
                self._document_tracker[doc_id] = conv_res

            # Start pipeline stages
            for stage in self.stages:
                stage.start()

            try:
                # Feed pages into pipeline
                self._feed_pipeline(pages_to_process, conv_res)

                # Collect results from pipeline with partial processing support
                result = self._collect_results_with_recovery(
                    conv_res, len(pages_to_process)
                )

                # Update conv_res with processed pages and handle partial results
                self._update_document_with_results(conv_res, result)

            finally:
                # Stop pipeline stages
                for stage in self.stages:
                    stage.stop()

                # Cleanup stage resources
                for stage in self.stages:
                    stage.cleanup()

                # Cleanup output queue
                self.output_queue.cleanup()

                # Cleanup document tracking
                with self._document_lock:
                    self._document_tracker.pop(doc_id, None)

        return conv_res

    def _feed_pipeline(self, pages: List[Page], conv_res: ConversionResult):
        """Feed pages into the pipeline"""
        for page in pages:
            item = ThreadedItem(
                payload=page,
                conv_res_id=id(conv_res),
                conv_res=conv_res,
                page_no=page.page_no,
            )

            # Feed into first stage with timeout
            if not self.preprocess_stage.input_queue.put(
                item, timeout=self.pipeline_options.stage_timeout_seconds
            ):
                _log.warning(f"Failed to feed page {page.page_no} due to backpressure")

    def _collect_results_with_recovery(
        self, conv_res: ConversionResult, expected_count: int
    ) -> ProcessingResult:
        """Collect processed pages from the pipeline with partial result support"""
        result = ProcessingResult(total_expected=expected_count)
        doc_id = id(conv_res)

        # Collect from output queue
        while len(result.pages) + len(result.failed_pages) < expected_count:
            batch = self.output_queue.get_batch(
                batch_size=expected_count
                - len(result.pages)
                - len(result.failed_pages),
                timeout=self.pipeline_options.collection_timeout_seconds,
            )

            if not batch:
                # Timeout reached, log missing pages
                missing_count = (
                    expected_count - len(result.pages) - len(result.failed_pages)
                )
                if missing_count > 0:
                    _log.warning(f"Pipeline timeout: missing {missing_count} pages")
                break

            for item in batch:
                if item.conv_res_id == doc_id:
                    if item.is_failed or item.error is not None:
                        result.failed_pages.append(
                            (item.page_no, item.error or Exception("Unknown error"))
                        )
                        _log.warning(
                            f"Page {item.page_no} failed processing: {item.error}"
                        )
                    else:
                        result.pages.append(item.payload)

        return result

    def _update_document_with_results(
        self, conv_res: ConversionResult, result: ProcessingResult
    ):
        """Update document with processing results and handle partial success"""
        # Update conv_res with successfully processed pages
        page_map = {p.page_no: p for p in result.pages}
        valid_pages = []

        for page in conv_res.pages:
            if page.page_no in page_map:
                valid_pages.append(page_map[page.page_no])
            elif not any(
                failed_page_no == page.page_no
                for failed_page_no, _ in result.failed_pages
            ):
                # Page wasn't processed but also didn't explicitly fail - keep original
                valid_pages.append(page)

        conv_res.pages = valid_pages

        # Handle partial results
        if result.is_partial_success:
            _log.warning(
                f"Partial processing success: {result.success_count} pages succeeded, "
                f"{result.failure_count} pages failed"
            )
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        elif result.is_complete_failure:
            _log.error("Complete processing failure: all pages failed")
            conv_res.status = ConversionStatus.FAILURE
        elif result.success_count > 0:
            # All expected pages processed successfully
            conv_res.status = ConversionStatus.SUCCESS

        # Clean up page resources if not keeping images
        if not self.keep_images:
            for page in conv_res.pages:
                # _image_cache is always present on Page objects, no need for hasattr
                page._image_cache = {}

        # Clean up page backends if not keeping them
        if not self.keep_backend:
            for page in conv_res.pages:
                if page._backend is not None:
                    page._backend.unload()

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble the final document from processed pages"""
        all_elements = []
        all_headers = []
        all_body = []

        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            for p in conv_res.pages:
                if p.assembled is not None:
                    for el in p.assembled.body:
                        all_body.append(el)
                    for el in p.assembled.headers:
                        all_headers.append(el)
                    for el in p.assembled.elements:
                        all_elements.append(el)

            conv_res.assembled = AssembledUnit(
                elements=all_elements, headers=all_headers, body=all_body
            )

            conv_res.document = self.reading_order_model(conv_res)

            # Generate page images
            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    if page.image is not None:
                        page_no = page.page_no + 1
                        conv_res.document.pages[page_no].image = ImageRef.from_pil(
                            page.image, dpi=int(72 * self.pipeline_options.images_scale)
                        )

            # Generate element images
            self._generate_element_images(conv_res)

            # Aggregate confidence scores
            self._aggregate_confidence(conv_res)

        return conv_res

    def _generate_element_images(self, conv_res: ConversionResult):
        """Generate images for picture and table elements"""
        with warnings.catch_warnings():
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
                        page_ix = element.prov[0].page_no - 1
                        page = next(
                            (p for p in conv_res.pages if p.page_no == page_ix), None
                        )
                        if (
                            page is not None
                            and page.size is not None
                            and page.image is not None
                        ):
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

    def _aggregate_confidence(self, conv_res: ConversionResult):
        """Aggregate confidence scores across pages"""
        if len(conv_res.pages) > 0:
            import warnings

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
                        q=0.1,
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

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Run enrichment models on the document"""

        def _prepare_elements(conv_res: ConversionResult, model: Any) -> Iterable[Any]:
            for doc_element, _level in conv_res.document.iterate_items():
                prepared_element = model.prepare_element(
                    conv_res=conv_res, element=doc_element
                )
                if prepared_element is not None:
                    yield prepared_element

        with TimeRecorder(conv_res, "doc_enrich", scope=ProfilingScope.DOCUMENT):
            for model in self.enrichment_pipe:
                for element_batch in chunkify(
                    _prepare_elements(conv_res, model),
                    model.elements_batch_size,
                ):
                    for element in model(
                        doc=conv_res.document, element_batch=element_batch
                    ):  # Must exhaust!
                        pass

        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine the final conversion status"""
        if conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            return ConversionStatus.PARTIAL_SUCCESS
        elif conv_res.pages and conv_res.document:
            return ConversionStatus.SUCCESS
        else:
            return ConversionStatus.FAILURE

    @classmethod
    def get_default_options(cls) -> ThreadedPdfPipelineOptions:
        return ThreadedPdfPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend):
        return isinstance(backend, PdfDocumentBackend)
