import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import ConversionStatus, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import AsyncPdfPipelineOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions
from docling.models.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.factories import get_ocr_factory, get_picture_description_factory

# Import the same models used by StandardPdfPipeline
from docling.models.layout_model import LayoutModel
from docling.models.page_assemble_model import PageAssembleModel, PageAssembleOptions
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from docling.models.table_structure_model import TableStructureModel
from docling.pipeline.async_base_pipeline import AsyncPipeline
from docling.pipeline.resource_manager import (
    AsyncPageTracker,
    ConversionResultAccumulator,
)
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


@dataclass
class PageBatch:
    """Represents a batch of pages to process through models"""

    pages: List[Page] = field(default_factory=list)
    conv_results: List[ConversionResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


@dataclass
class QueueTerminator:
    """Sentinel value for proper queue termination tracking"""

    stage: str
    error: Optional[Exception] = None


class AsyncStandardPdfPipeline(AsyncPipeline):
    """Async pipeline implementation with cross-document batching using structured concurrency"""

    def __init__(self, pipeline_options: AsyncPdfPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: AsyncPdfPipelineOptions = pipeline_options

        # Resource management
        self.page_tracker = AsyncPageTracker(
            keep_images=self._should_keep_images(),
            keep_backend=self._should_keep_backend(),
        )

        # Initialize models (same as StandardPdfPipeline)
        self._initialize_models()

    def _should_keep_images(self) -> bool:
        """Determine if images should be kept (same logic as StandardPdfPipeline)"""
        return (
            self.pipeline_options.generate_page_images
            or self.pipeline_options.generate_picture_images
            or self.pipeline_options.generate_table_images
        )

    def _should_keep_backend(self) -> bool:
        """Determine if backend should be kept"""
        return (
            self.pipeline_options.do_formula_enrichment
            or self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_picture_classification
            or self.pipeline_options.do_picture_description
        )

    def _initialize_models(self):
        """Initialize all models (matching StandardPdfPipeline)"""
        artifacts_path = self._get_artifacts_path()

        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # Build pipeline stages
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

        # Enrichment models
        self.code_formula_model = CodeFormulaModel(
            enabled=self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_formula_enrichment,
            artifacts_path=artifacts_path,
            options=CodeFormulaModelOptions(
                do_code_enrichment=self.pipeline_options.do_code_enrichment,
                do_formula_enrichment=self.pipeline_options.do_formula_enrichment,
            ),
            accelerator_options=self.pipeline_options.accelerator_options,
        )

        self.picture_classifier = DocumentPictureClassifier(
            enabled=self.pipeline_options.do_picture_classification,
            artifacts_path=artifacts_path,
            options=DocumentPictureClassifierOptions(),
            accelerator_options=self.pipeline_options.accelerator_options,
        )

        self.picture_description_model = self._get_picture_description_model(
            artifacts_path
        )

    def _get_artifacts_path(self) -> Optional[str]:
        """Get artifacts path (same as StandardPdfPipeline)"""
        from pathlib import Path

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

    def _get_ocr_model(self, artifacts_path: Optional[str] = None) -> BaseOcrModel:
        """Get OCR model (same as StandardPdfPipeline)"""
        factory = get_ocr_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        return factory.create_instance(
            options=self.pipeline_options.ocr_options,
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=artifacts_path,
            accelerator_options=self.pipeline_options.accelerator_options,
        )

    def _get_picture_description_model(self, artifacts_path: Optional[str] = None):
        """Get picture description model (same as StandardPdfPipeline)"""
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

    async def execute_stream(
        self, input_docs: AsyncIterable[InputDocument]
    ) -> AsyncIterable[ConversionResult]:
        """Main async processing with structured concurrency and proper exception handling"""
        # Create queues for pipeline stages
        page_queue = asyncio.Queue(maxsize=self.pipeline_options.extraction_queue_size)
        completed_queue = asyncio.Queue()
        completed_docs = asyncio.Queue()

        # Track active documents for proper termination
        doc_tracker = {"active_docs": 0, "extraction_done": False}
        doc_lock = asyncio.Lock()

        # Create exception event for coordinated shutdown
        exception_event = asyncio.Event()

        async def track_document_start():
            async with doc_lock:
                doc_tracker["active_docs"] += 1

        async def track_document_complete():
            async with doc_lock:
                doc_tracker["active_docs"] -= 1
                if doc_tracker["extraction_done"] and doc_tracker["active_docs"] == 0:
                    # All documents completed
                    await completed_docs.put(None)

        try:
            async with asyncio.TaskGroup() as tg:
                # Start all tasks
                tg.create_task(
                    self._extract_documents_wrapper(
                        input_docs,
                        page_queue,
                        track_document_start,
                        exception_event,
                        doc_tracker,
                        doc_lock,
                    )
                )
                tg.create_task(
                    self._process_pages_wrapper(
                        page_queue, completed_queue, exception_event
                    )
                )
                tg.create_task(
                    self._aggregate_results_wrapper(
                        completed_queue,
                        completed_docs,
                        track_document_complete,
                        exception_event,
                    )
                )

                # Yield results as they complete
                async for result in self._yield_results(
                    completed_docs, exception_event
                ):
                    yield result

        except* Exception as eg:
            # Handle exception group from TaskGroup
            _log.error(f"Pipeline failed with exceptions: {eg.exceptions}")
            # Re-raise the first exception
            raise (eg.exceptions[0] if eg.exceptions else RuntimeError("Unknown error"))
        finally:
            # Ensure cleanup
            await self.page_tracker.cleanup_all()

    async def _extract_documents_wrapper(
        self,
        input_docs: AsyncIterable[InputDocument],
        page_queue: asyncio.Queue,
        track_document_start,
        exception_event: asyncio.Event,
        doc_tracker: Dict[str, Any],
        doc_lock: asyncio.Lock,
    ):
        """Wrapper for document extraction with exception handling"""
        try:
            await self._extract_documents_safe(
                input_docs,
                page_queue,
                track_document_start,
                exception_event,
            )
        except Exception:
            exception_event.set()
            raise
        finally:
            async with doc_lock:
                doc_tracker["extraction_done"] = True
            # Send termination signal
            await page_queue.put(QueueTerminator("extraction"))

    async def _process_pages_wrapper(
        self,
        page_queue: asyncio.Queue,
        completed_queue: asyncio.Queue,
        exception_event: asyncio.Event,
    ):
        """Wrapper for page processing with exception handling"""
        try:
            await self._process_pages_safe(page_queue, completed_queue, exception_event)
        except Exception:
            exception_event.set()
            raise
        finally:
            # Send termination signal
            await completed_queue.put(QueueTerminator("processing"))

    async def _aggregate_results_wrapper(
        self,
        completed_queue: asyncio.Queue,
        completed_docs: asyncio.Queue,
        track_document_complete,
        exception_event: asyncio.Event,
    ):
        """Wrapper for result aggregation with exception handling"""
        try:
            await self._aggregate_results_safe(
                completed_queue,
                completed_docs,
                track_document_complete,
                exception_event,
            )
        except Exception:
            exception_event.set()
            raise

    async def _yield_results(
        self, completed_docs: asyncio.Queue, exception_event: asyncio.Event
    ):
        """Yield results as they complete"""
        while True:
            if exception_event.is_set():
                break

            try:
                result = await asyncio.wait_for(completed_docs.get(), timeout=1.0)
                if result is None:
                    break
                yield result
            except asyncio.TimeoutError:
                continue
            except Exception:
                exception_event.set()
                raise

    async def _extract_documents_safe(
        self,
        input_docs: AsyncIterable[InputDocument],
        page_queue: asyncio.Queue,
        track_document_start,
        exception_event: asyncio.Event,
    ) -> None:
        """Extract pages from documents with exception handling"""
        async for in_doc in input_docs:
            if exception_event.is_set():
                break

            await track_document_start()
            conv_res = ConversionResult(input=in_doc)

            # Validate backend
            if not isinstance(conv_res.input._backend, PdfDocumentBackend):
                conv_res.status = ConversionStatus.FAILURE
                await page_queue.put((None, conv_res))  # Signal failed document
                continue

            try:
                # Initialize document
                total_pages = conv_res.input.page_count
                await self.page_tracker.register_document(conv_res, total_pages)

                # Extract pages with limited concurrency
                semaphore = asyncio.Semaphore(
                    self.pipeline_options.max_concurrent_extractions
                )

                async def extract_page(page_no: int):
                    if exception_event.is_set():
                        return

                    async with semaphore:
                        # Create page
                        page = Page(page_no=page_no)
                        conv_res.pages.append(page)

                        # Initialize page backend
                        page._backend = await asyncio.to_thread(
                            conv_res.input._backend.load_page, page_no
                        )

                        if page._backend is not None and page._backend.is_valid():
                            page.size = page._backend.get_size()
                            await self.page_tracker.track_page_loaded(page, conv_res)

                        # Send to processing queue
                        await page_queue.put((page, conv_res))

                # Extract all pages concurrently
                async with asyncio.TaskGroup() as tg:
                    for i in range(total_pages):
                        if exception_event.is_set():
                            break
                        start_page, end_page = conv_res.input.limits.page_range
                        if (start_page - 1) <= i <= (end_page - 1):
                            tg.create_task(extract_page(i))

            except Exception as e:
                _log.error(f"Failed to extract document {in_doc.file.name}: {e}")
                conv_res.status = ConversionStatus.FAILURE
                # Signal document failure
                await page_queue.put((None, conv_res))
                raise

    async def _process_pages_safe(
        self,
        page_queue: asyncio.Queue,
        completed_queue: asyncio.Queue,
        exception_event: asyncio.Event,
    ) -> None:
        """Process pages through model pipeline with proper termination"""
        # Process batches through each model stage
        preprocessing_queue = asyncio.Queue()
        ocr_queue = asyncio.Queue()
        layout_queue = asyncio.Queue()
        table_queue = asyncio.Queue()
        assemble_queue = asyncio.Queue()

        # Start processing stages using TaskGroup
        async with asyncio.TaskGroup() as tg:
            # Preprocessing stage
            tg.create_task(
                self._batch_process_stage_safe(
                    page_queue,
                    preprocessing_queue,
                    self._preprocess_batch,
                    1,
                    0,  # No batching for preprocessing
                    "preprocessing",
                    exception_event,
                )
            )

            # OCR stage
            tg.create_task(
                self._batch_process_stage_safe(
                    preprocessing_queue,
                    ocr_queue,
                    self._ocr_batch,
                    self.pipeline_options.ocr_batch_size,
                    self.pipeline_options.batch_timeout_seconds,
                    "ocr",
                    exception_event,
                )
            )

            # Layout stage
            tg.create_task(
                self._batch_process_stage_safe(
                    ocr_queue,
                    layout_queue,
                    self._layout_batch,
                    self.pipeline_options.layout_batch_size,
                    self.pipeline_options.batch_timeout_seconds,
                    "layout",
                    exception_event,
                )
            )

            # Table stage
            tg.create_task(
                self._batch_process_stage_safe(
                    layout_queue,
                    table_queue,
                    self._table_batch,
                    self.pipeline_options.table_batch_size,
                    self.pipeline_options.batch_timeout_seconds,
                    "table",
                    exception_event,
                )
            )

            # Assembly stage
            tg.create_task(
                self._batch_process_stage_safe(
                    table_queue,
                    assemble_queue,
                    self._assemble_batch,
                    1,
                    0,  # No batching for assembly
                    "assembly",
                    exception_event,
                )
            )

            # Finalization stage
            tg.create_task(
                self._finalize_pages_safe(
                    assemble_queue, completed_queue, exception_event
                )
            )

    async def _batch_process_stage_safe(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        process_func,
        batch_size: int,
        timeout: float,
        stage_name: str,
        exception_event: asyncio.Event,
    ) -> None:
        """Generic batch processing stage with proper termination handling"""
        batch = PageBatch()

        try:
            while not exception_event.is_set():
                # Collect batch
                try:
                    # Get first item or wait for timeout
                    if not batch.pages:
                        item = await input_queue.get()

                        # Check for termination
                        if isinstance(item, QueueTerminator):
                            # Propagate termination signal
                            await output_queue.put(item)
                            break

                        # Handle failed document signal
                        if item[0] is None:
                            # Pass through failure signal
                            await output_queue.put(item)
                            continue

                        batch.pages.append(item[0])
                        batch.conv_results.append(item[1])

                    # Try to fill batch up to batch_size
                    while len(batch.pages) < batch_size:
                        remaining_time = timeout - (time.time() - batch.start_time)
                        if remaining_time <= 0:
                            break

                        try:
                            item = await asyncio.wait_for(
                                input_queue.get(), timeout=remaining_time
                            )

                            # Check for termination
                            if isinstance(item, QueueTerminator):
                                # Put it back and process current batch
                                await input_queue.put(item)
                                break

                            # Handle failed document signal
                            if item[0] is None:
                                # Put it back and process current batch
                                await input_queue.put(item)
                                break

                            batch.pages.append(item[0])
                            batch.conv_results.append(item[1])
                        except asyncio.TimeoutError:
                            break

                    # Process batch
                    if batch.pages:
                        processed = await process_func(batch)

                        # Send results to output queue
                        for page, conv_res in processed:
                            await output_queue.put((page, conv_res))

                        # Clear batch
                        batch = PageBatch()

                except Exception as e:
                    _log.error(f"Error in {stage_name} batch processing: {e}")
                    # Send failed items downstream
                    for page, conv_res in zip(batch.pages, batch.conv_results):
                        await output_queue.put((page, conv_res))
                    batch = PageBatch()
                    raise

        except Exception as e:
            # Set exception event and propagate termination
            exception_event.set()
            await output_queue.put(QueueTerminator(stage_name, error=e))
            raise

    async def _preprocess_batch(
        self, batch: PageBatch
    ) -> List[Tuple[Page, ConversionResult]]:
        """Preprocess pages (no actual batching needed)"""
        results = []
        for page, conv_res in zip(batch.pages, batch.conv_results):
            processed_page = await asyncio.to_thread(
                lambda: next(iter(self.preprocessing_model(conv_res, [page])))
            )
            results.append((processed_page, conv_res))
        return results

    async def _ocr_batch(self, batch: PageBatch) -> List[Tuple[Page, ConversionResult]]:
        """Process OCR in batch"""
        # Group by conversion result for proper context
        grouped = defaultdict(list)
        for page, conv_res in zip(batch.pages, batch.conv_results):
            grouped[id(conv_res)].append(page)

        results = []
        for conv_res_id, pages in grouped.items():
            # Find the conv_res
            conv_res = next(
                cr
                for p, cr in zip(batch.pages, batch.conv_results)
                if id(cr) == conv_res_id
            )

            # Process batch through OCR model
            processed_pages = await asyncio.to_thread(
                lambda: list(self.ocr_model(conv_res, pages))
            )

            for page in processed_pages:
                results.append((page, conv_res))

        return results

    async def _layout_batch(
        self, batch: PageBatch
    ) -> List[Tuple[Page, ConversionResult]]:
        """Process layout in batch"""
        # Similar batching as OCR
        grouped = defaultdict(list)
        for page, conv_res in zip(batch.pages, batch.conv_results):
            grouped[id(conv_res)].append(page)

        results = []
        for conv_res_id, pages in grouped.items():
            conv_res = next(
                cr
                for p, cr in zip(batch.pages, batch.conv_results)
                if id(cr) == conv_res_id
            )

            processed_pages = await asyncio.to_thread(
                lambda: list(self.layout_model(conv_res, pages))
            )

            for page in processed_pages:
                results.append((page, conv_res))

        return results

    async def _table_batch(
        self, batch: PageBatch
    ) -> List[Tuple[Page, ConversionResult]]:
        """Process tables in batch"""
        grouped = defaultdict(list)
        for page, conv_res in zip(batch.pages, batch.conv_results):
            grouped[id(conv_res)].append(page)

        results = []
        for conv_res_id, pages in grouped.items():
            conv_res = next(
                cr
                for p, cr in zip(batch.pages, batch.conv_results)
                if id(cr) == conv_res_id
            )

            processed_pages = await asyncio.to_thread(
                lambda: list(self.table_model(conv_res, pages))
            )

            for page in processed_pages:
                results.append((page, conv_res))

        return results

    async def _assemble_batch(
        self, batch: PageBatch
    ) -> List[Tuple[Page, ConversionResult]]:
        """Assemble pages (no actual batching needed)"""
        results = []
        for page, conv_res in zip(batch.pages, batch.conv_results):
            assembled_page = await asyncio.to_thread(
                lambda: next(iter(self.assemble_model(conv_res, [page])))
            )
            results.append((assembled_page, conv_res))
        return results

    async def _finalize_pages_safe(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        exception_event: asyncio.Event,
    ) -> None:
        """Finalize pages and track completion with proper termination"""
        try:
            while not exception_event.is_set():
                item = await input_queue.get()

                # Check for termination
                if isinstance(item, QueueTerminator):
                    # Propagate termination signal
                    await output_queue.put(item)
                    break

                # Handle failed document signal
                if item[0] is None:
                    # Pass through failure signal
                    await output_queue.put(item)
                    continue

                page, conv_res = item

                # Track page completion for resource cleanup
                await self.page_tracker.track_page_completion(page, conv_res)

                # Send to output
                await output_queue.put((page, conv_res))

        except Exception as e:
            exception_event.set()
            await output_queue.put(QueueTerminator("finalization", error=e))
            raise

    async def _aggregate_results_safe(
        self,
        completed_queue: asyncio.Queue,
        completed_docs: asyncio.Queue,
        track_document_complete,
        exception_event: asyncio.Event,
    ) -> None:
        """Aggregate completed pages into documents with proper termination"""
        doc_pages = defaultdict(list)
        failed_docs = set()

        try:
            while not exception_event.is_set():
                item = await completed_queue.get()

                # Check for termination
                if isinstance(item, QueueTerminator):
                    # Finalize any remaining documents
                    for conv_res_id, pages in doc_pages.items():
                        if conv_res_id not in failed_docs:
                            # Find conv_res from first page
                            conv_res = pages[0][1]
                            await self._finalize_document(conv_res)
                            await completed_docs.put(conv_res)
                            await track_document_complete()
                    break

                # Handle failed document signal
                if item[0] is None:
                    conv_res = item[1]
                    doc_id = id(conv_res)
                    failed_docs.add(doc_id)
                    # Send failed document immediately
                    await completed_docs.put(conv_res)
                    await track_document_complete()
                    continue

                page, conv_res = item
                doc_id = id(conv_res)

                if doc_id not in failed_docs:
                    doc_pages[doc_id].append((page, conv_res))

                    # Check if document is complete
                    if len(doc_pages[doc_id]) == len(conv_res.pages):
                        await self._finalize_document(conv_res)
                        await completed_docs.put(conv_res)
                        await track_document_complete()
                        del doc_pages[doc_id]

        except Exception:
            exception_event.set()
            # Try to send any completed documents before failing
            for conv_res_id, pages in doc_pages.items():
                if conv_res_id not in failed_docs and pages:
                    conv_res = pages[0][1]
                    conv_res.status = ConversionStatus.PARTIAL_SUCCESS
                    await completed_docs.put(conv_res)
                    await track_document_complete()
            raise

    async def _finalize_document(self, conv_res: ConversionResult) -> None:
        """Finalize a complete document (same as StandardPdfPipeline._assemble_document)"""
        # This matches the logic from StandardPdfPipeline
        import warnings

        import numpy as np

        from docling.datamodel.base_models import AssembledUnit

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

            # Generate page images in the output
            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    if page.image is not None:
                        page_no = page.page_no + 1
                        from docling_core.types.doc import ImageRef

                        conv_res.document.pages[page_no].image = ImageRef.from_pil(
                            page.image, dpi=int(72 * self.pipeline_options.images_scale)
                        )

            # Handle picture/table images (same as StandardPdfPipeline)
            self._generate_element_images(conv_res)

            # Aggregate confidence values
            self._aggregate_confidence(conv_res)

            # Run enrichment pipeline
            await self._enrich_document(conv_res)

            # Set final status
            conv_res.status = self._determine_status(conv_res)

    def _generate_element_images(self, conv_res: ConversionResult) -> None:
        """Generate images for elements (same as StandardPdfPipeline)"""
        import warnings

        from docling_core.types.doc import DocItem, ImageRef, PictureItem, TableItem

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

    def _aggregate_confidence(self, conv_res: ConversionResult) -> None:
        """Aggregate confidence scores (same as StandardPdfPipeline)"""
        import warnings

        import numpy as np

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

    async def _enrich_document(self, conv_res: ConversionResult) -> None:
        """Run enrichment models on document"""
        # Run enrichment models (same as base pipeline but async)
        from docling.utils.utils import chunkify

        enrichment_models = [
            self.code_formula_model,
            self.picture_classifier,
            self.picture_description_model,
        ]

        for model in enrichment_models:
            if model is None or not getattr(model, "enabled", True):
                continue

            # Prepare elements
            elements_to_process = []
            for doc_element, _level in conv_res.document.iterate_items():
                prepared = model.prepare_element(conv_res=conv_res, element=doc_element)
                if prepared is not None:
                    elements_to_process.append(prepared)

            # Process in batches
            for element_batch in chunkify(
                elements_to_process, model.elements_batch_size
            ):
                # Run model in thread to avoid blocking
                await asyncio.to_thread(
                    lambda: list(model(conv_res.document, element_batch))
                )

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine conversion status"""
        # Simple implementation - could be enhanced
        if conv_res.pages and conv_res.document:
            return ConversionStatus.SUCCESS
        else:
            return ConversionStatus.FAILURE
