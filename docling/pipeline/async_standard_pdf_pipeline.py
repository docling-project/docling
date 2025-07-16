import asyncio
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

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
from docling.models.layout_model import LayoutModel
from docling.models.page_assemble_model import PageAssembleModel, PageAssembleOptions
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from docling.models.table_structure_model import TableStructureModel
from docling.pipeline.async_base_pipeline import AsyncPipeline
from docling.pipeline.graph import GraphRunner, get_pipeline_thread_pool
from docling.pipeline.resource_manager import AsyncPageTracker
from docling.pipeline.stages import (
    AggregationStage,
    BatchProcessorStage,
    ExtractionStage,
    PageProcessorStage,
    SinkStage,
    SourceStage,
)
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class AsyncStandardPdfPipeline(AsyncPipeline):
    """
    An async, graph-based pipeline for processing PDFs with cross-document batching.
    """

    def __init__(self, pipeline_options: AsyncPdfPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: AsyncPdfPipelineOptions = pipeline_options
        self.page_tracker = AsyncPageTracker(
            keep_images=self._should_keep_images(),
            keep_backend=self._should_keep_backend(),
        )
        # Get shared thread pool for enrichment operations
        self._thread_pool = get_pipeline_thread_pool()
        self._initialize_models()

    def _should_keep_images(self) -> bool:
        return (
            self.pipeline_options.generate_page_images
            or self.pipeline_options.generate_picture_images
            or self.pipeline_options.generate_table_images
        )

    def _should_keep_backend(self) -> bool:
        return (
            self.pipeline_options.do_formula_enrichment
            or self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_picture_classification
            or self.pipeline_options.do_picture_description
        )

    def _initialize_models(self):
        artifacts_path = self._get_artifacts_path()
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())
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
        """Main async processing driven by a pipeline graph."""
        stages = [
            SourceStage("source"),
            ExtractionStage(
                "extractor",
                self.page_tracker,
                self.pipeline_options.max_concurrent_extractions,
            ),
            PageProcessorStage("preprocessor", self.preprocessing_model),
            BatchProcessorStage(
                "ocr",
                self.ocr_model,
                self.pipeline_options.ocr_batch_size,
                self.pipeline_options.batch_timeout_seconds,
            ),
            BatchProcessorStage(
                "layout",
                self.layout_model,
                self.pipeline_options.layout_batch_size,
                self.pipeline_options.batch_timeout_seconds,
            ),
            BatchProcessorStage(
                "table",
                self.table_model,
                self.pipeline_options.table_batch_size,
                self.pipeline_options.batch_timeout_seconds,
            ),
            PageProcessorStage("assembler", self.assemble_model),
            AggregationStage("aggregator", self.page_tracker, self._finalize_document),
            SinkStage("sink"),
        ]

        edges = [
            # Main processing path
            {
                "from_stage": "source",
                "from_output": "out",
                "to_stage": "extractor",
                "to_input": "in",
            },
            {
                "from_stage": "extractor",
                "from_output": "out",
                "to_stage": "preprocessor",
                "to_input": "in",
            },
            {
                "from_stage": "preprocessor",
                "from_output": "out",
                "to_stage": "ocr",
                "to_input": "in",
            },
            {
                "from_stage": "ocr",
                "from_output": "out",
                "to_stage": "layout",
                "to_input": "in",
            },
            {
                "from_stage": "layout",
                "from_output": "out",
                "to_stage": "table",
                "to_input": "in",
            },
            {
                "from_stage": "table",
                "from_output": "out",
                "to_stage": "assembler",
                "to_input": "in",
            },
            {
                "from_stage": "assembler",
                "from_output": "out",
                "to_stage": "aggregator",
                "to_input": "in",
            },
            # Failure path
            {
                "from_stage": "extractor",
                "from_output": "fail",
                "to_stage": "aggregator",
                "to_input": "fail",
            },
            # Final output
            {
                "from_stage": "aggregator",
                "from_output": "out",
                "to_stage": "sink",
                "to_input": "in",
            },
        ]

        runner = GraphRunner(stages, edges)
        source_config = {"stage": "source", "channel": "out"}
        sink_config = {"stage": "sink", "channel": "in"}

        try:
            async for result in runner.run(
                input_docs,
                source_config,
                sink_config,
                self.pipeline_options.extraction_queue_size,
            ):
                yield result
        except* Exception as eg:
            _log.error(f"Pipeline failed with exceptions: {eg.exceptions}")
            raise (eg.exceptions[0] if eg.exceptions else RuntimeError("Unknown error"))
        finally:
            await self.page_tracker.cleanup_all()

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
                # Run model in shared thread pool to avoid blocking
                await asyncio.get_running_loop().run_in_executor(
                    self._thread_pool,
                    lambda: list(model(conv_res.document, element_batch)),
                )

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine conversion status"""
        # Simple implementation - could be enhanced
        if conv_res.pages and conv_res.document:
            return ConversionStatus.SUCCESS
        else:
            return ConversionStatus.FAILURE
