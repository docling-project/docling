"""Threaded Layout+VLM Pipeline
================================
A specialized two-stage threaded pipeline that combines layout model preprocessing
with VLM processing. The layout model detects document elements and coordinates,
which are then injected into the VLM prompt for enhanced structured output.
"""

from __future__ import annotations

import itertools
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from docling_core.types.doc import DocItem, DoclingDocument, ImageRef, PictureItem
from docling_core.types.doc.document import DocTagsDocument

if TYPE_CHECKING:
    from docling_core.types.doc.page import SegmentedPage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, iter_pdf_page_backends
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    LayoutPostprocessorOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
)
from docling.datamodel.settings import settings
from docling.experimental.datamodel.threaded_layout_vlm_pipeline_options import (
    ThreadedLayoutVlmPipelineOptions,
)
from docling.models.base_model import BaseVlmPageModel
from docling.models.factories import get_layout_factory
from docling.models.stages.layout.layout_postprocessing_model import (
    LayoutPostprocessingModel,
)
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.models.vlm_pipeline_models.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.standard_pdf_pipeline import (
    ProcessingResult,
    RunContext,
    ThreadedItem,
    ThreadedPipelineStage,
    ThreadedQueue,
)
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class ThreadedLayoutVlmPipeline(BasePipeline):
    """Two-stage threaded pipeline: Layout Model → VLM Model."""

    def __init__(self, pipeline_options: ThreadedLayoutVlmPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: ThreadedLayoutVlmPipelineOptions = pipeline_options
        self._run_seq = itertools.count(1)  # deterministic, monotonic run ids

        # VLM model type (initialized in _init_models)
        self.vlm_model: BaseVlmPageModel

        # Initialize models
        self._init_models()

    def _init_models(self) -> None:
        """Initialize layout and VLM models."""
        art_path = self._resolve_artifacts_path()

        # Layout model
        layout_factory = get_layout_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        self.layout_model = layout_factory.create_instance(
            options=self.pipeline_options.layout_options,
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            enable_remote_services=self.pipeline_options.enable_remote_services,
        )

        # Standalone layout post-processing stage; the VLM prompt augmentation
        # reads processed clusters from page.predictions.layout.
        lo = self.pipeline_options.layout_options
        self.layout_postprocessing_model = LayoutPostprocessingModel(
            options=LayoutPostprocessorOptions(
                skip_cell_assignment=lo.skip_cell_assignment,
                keep_empty_clusters=lo.keep_empty_clusters,
                create_orphan_clusters=lo.create_orphan_clusters,
                run_postprocessor=self.layout_model.requires_layout_postprocessing,
            )
        )

        # VLM model based on options type
        # Create layout-aware VLM options internally
        base_vlm_options = self.pipeline_options.vlm_options

        class LayoutAwareVlmOptions(type(base_vlm_options)):  # type: ignore[misc]
            def build_prompt(
                self,
                page: Optional[SegmentedPage],
                *,
                _internal_page: Optional[Page] = None,
            ) -> str:
                base_prompt = self.prompt
                augmented_prompt = base_prompt

                # In this layout-aware pipeline, _internal_page is always provided
                if _internal_page is None:
                    return base_prompt

                if not _internal_page.size:
                    _log.warning(
                        f"Page size not available for page {_internal_page.page_no}. Cannot enhance prompt with layout info."
                    )
                    return base_prompt

                if _internal_page.predictions.layout:
                    from docling_core.types.doc.tokens import DocumentToken

                    layout_elements = []
                    for cluster in _internal_page.predictions.layout.clusters:
                        # Get proper tag name from DocItemLabel
                        tag_name = DocumentToken.create_token_name_from_doc_item_label(
                            label=cluster.label
                        )

                        # Convert bbox to tuple and get location tokens
                        bbox_tuple = cluster.bbox.as_tuple()
                        location_tokens = DocumentToken.get_location(
                            bbox=bbox_tuple,
                            page_w=_internal_page.size.width,
                            page_h=_internal_page.size.height,
                        )

                        # Create XML element with DocTags format
                        xml_element = f"<{tag_name}>{location_tokens}</{tag_name}>"
                        layout_elements.append(xml_element)

                    if layout_elements:
                        # Join elements with newlines and wrap in layout tags
                        layout_xml = (
                            "<layout>" + "\n".join(layout_elements) + "</layout>"
                        )
                        layout_injection = f"{layout_xml}"

                        augmented_prompt = base_prompt + layout_injection

                    _log.debug(
                        "Enhanced Prompt with Layout Info: %s\n", augmented_prompt
                    )

                return augmented_prompt

        vlm_options = LayoutAwareVlmOptions(**base_vlm_options.model_dump())

        if isinstance(base_vlm_options, ApiVlmOptions):
            self.vlm_model = ApiVlmModel(
                enabled=True,
                enable_remote_services=self.pipeline_options.enable_remote_services,
                vlm_options=vlm_options,
            )
        elif isinstance(base_vlm_options, InlineVlmOptions):
            if vlm_options.inference_framework == InferenceFramework.TRANSFORMERS:
                self.vlm_model = HuggingFaceTransformersVlmModel(
                    enabled=True,
                    artifacts_path=art_path,
                    accelerator_options=self.pipeline_options.accelerator_options,
                    vlm_options=vlm_options,
                )
            elif vlm_options.inference_framework == InferenceFramework.MLX:
                self.vlm_model = HuggingFaceMlxModel(
                    enabled=True,
                    artifacts_path=art_path,
                    accelerator_options=self.pipeline_options.accelerator_options,
                    vlm_options=vlm_options,
                )
            elif vlm_options.inference_framework == InferenceFramework.VLLM:
                from docling.models.vlm_pipeline_models.vllm_model import VllmVlmModel

                self.vlm_model = VllmVlmModel(
                    enabled=True,
                    artifacts_path=art_path,
                    accelerator_options=self.pipeline_options.accelerator_options,
                    vlm_options=vlm_options,
                )
            else:
                raise ValueError(
                    f"Unsupported VLM inference framework: {vlm_options.inference_framework}"
                )
        else:
            raise ValueError(f"Unsupported VLM options type: {type(base_vlm_options)}")

    def _resolve_artifacts_path(self) -> Optional[Path]:
        """Resolve artifacts path from options or settings."""
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

    def _create_run_ctx(self) -> RunContext:
        """Create pipeline stages and wire them together."""
        opts = self.pipeline_options

        # Layout stage
        layout_stage = ThreadedPipelineStage(
            name="layout",
            model=self.layout_model,
            batch_size=opts.layout_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # Layout post-processing stage
        layout_postprocess_stage = ThreadedPipelineStage(
            name="layout_postprocess",
            model=self.layout_postprocessing_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # VLM stage - now layout-aware through enhanced build_prompt
        vlm_stage = ThreadedPipelineStage(
            name="vlm",
            model=self.vlm_model,
            batch_size=opts.vlm_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # Wire stages
        output_q = ThreadedQueue(opts.queue_max_size)
        layout_stage.add_output_queue(layout_postprocess_stage.input_queue)
        layout_postprocess_stage.add_output_queue(vlm_stage.input_queue)
        vlm_stage.add_output_queue(output_q)

        stages = [layout_stage, layout_postprocess_stage, vlm_stage]
        return RunContext(
            stages=stages, first_stage=layout_stage, output_queue=output_q
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document using threaded layout+VLM pipeline."""
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        backend = conv_res.input._backend
        run_id = next(self._run_seq)

        start_page, end_page = conv_res.input.limits.page_range
        expected_page_nos = list(
            range(max(1, start_page), min(conv_res.input.page_count, end_page) + 1)
        )
        if not expected_page_nos:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        page_by_no = {page_no: Page(page_no=page_no) for page_no in expected_page_nos}
        conv_res.pages = list(page_by_no.values())
        for page in conv_res.pages:
            page._default_image_scale = self.pipeline_options.images_scale

        total_pages = len(expected_page_nos)
        ctx = self._create_run_ctx()
        for st in ctx.stages:
            st.start()

        proc = ProcessingResult(total_expected=total_pages)
        batch_size = 32
        producer_error: list[Exception] = []
        page_documents: dict[int, DoclingDocument] = {}

        def _completed_page_nos() -> set[int]:
            return {page.page_no for page in proc.pages} | {
                page_no for page_no, _, _ in proc.failed_pages if page_no > 0
            }

        def _produce_pages() -> None:
            try:
                for page_backend in iter_pdf_page_backends(backend, expected_page_nos):
                    page = page_by_no.get(page_backend.page_no)
                    if page is None:
                        page_backend.unload()
                        continue
                    page._backend = page_backend
                    try:
                        page.size = page_backend.get_size()
                    except Exception:
                        if page_backend.is_valid():
                            page_backend.unload()
                            page._backend = None
                            raise
                    if not ctx.first_stage.input_queue.put(
                        ThreadedItem(
                            payload=page,
                            run_id=run_id,
                            page_no=page.page_no,
                            conv_res=conv_res,
                        )
                    ):
                        self._release_page_resources(page)
                        break
            except Exception as exc:
                producer_error.append(exc)
                _log.error("Producer failed for run %d: %s", run_id, exc, exc_info=True)
            finally:
                ctx.first_stage.input_queue.close()

        producer_thread = threading.Thread(
            target=_produce_pages, name=f"LayoutVlmPageProducer-{run_id}", daemon=False
        )
        producer_thread.start()

        try:
            while proc.success_count + proc.failure_count < total_pages:
                out_batch = ctx.output_queue.get_batch(batch_size, timeout=0.05)
                for itm in out_batch:
                    if itm.run_id != run_id:
                        continue
                    if itm.is_failed or itm.error:
                        proc.failed_pages.append(
                            (
                                itm.page_no,
                                itm.error or RuntimeError("unknown error"),
                                itm.failure,
                            )
                        )
                        if itm.payload is not None:
                            self._release_page_resources(itm.payload)
                    else:
                        assert itm.payload is not None
                        page = itm.payload
                        try:
                            page_document = self._finalize_page_document(page)
                            if page_document is not None:
                                page_documents[page.page_no] = page_document
                            proc.pages.append(page)
                        except Exception as exc:
                            proc.failed_pages.append(
                                (
                                    page.page_no,
                                    exc,
                                    ErrorItem(
                                        component_type=DoclingComponentType.PIPELINE,
                                        module_name=self.__class__.__name__,
                                        error_message=str(exc)
                                        or exc.__class__.__name__,
                                        category=FailureCategory.UNKNOWN,
                                        page_no=page.page_no,
                                    ),
                                )
                            )
                        finally:
                            self._release_page_resources(page)

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
                            (page_no, error, None) for page_no in missing_page_nos
                        )
                    break
        finally:
            for st in ctx.stages:
                st.stop()
            ctx.output_queue.close()
            producer_thread.join(timeout=15.0)
            if producer_thread.is_alive():
                _log.warning(
                    "Producer thread for run %d did not terminate within 15.0s and will be abandoned.",
                    run_id,
                )
            for page in conv_res.pages:
                self._release_page_resources(page)

        self._integrate_results(conv_res, proc)
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            ordered_page_documents = [
                (page.page_no, page_documents[page.page_no])
                for page in conv_res.pages
                if page.page_no in page_documents
            ]
            conv_res.document = self._concatenate_page_documents(ordered_page_documents)
        return conv_res

    def _finalize_page_document(self, page: Page) -> DoclingDocument | None:
        image = page.image
        response = page.predictions.vlm_response
        if image is None or response is None:
            return None

        doctags_document = DocTagsDocument.from_doctags_and_image_pairs(
            [response.text], [image]
        )
        document = DoclingDocument.load_from_doctags(doctag_document=doctags_document)

        if self.pipeline_options.generate_picture_images:
            assert page.size is not None
            scale = self.pipeline_options.images_scale
            for element, _level in document.iterate_items():
                if not isinstance(element, PictureItem) or not element.prov:
                    continue
                crop_bbox = (
                    element.prov[0]
                    .bbox.scaled(scale=scale)
                    .to_top_left_origin(page_height=page.size.height * scale)
                )
                element.image = ImageRef.from_pil(
                    image.crop(crop_bbox.as_tuple()), dpi=int(72 * scale)
                )

        if not self.pipeline_options.generate_page_images:
            for page_item in document.pages.values():
                page_item.image = None
        return document

    @staticmethod
    def _concatenate_page_documents(
        page_documents: list[tuple[int, DoclingDocument]],
    ) -> DoclingDocument:
        if not page_documents:
            return DoclingDocument(name="")

        document = DoclingDocument.concatenate(
            docs=[page_document for _, page_document in page_documents]
        )
        page_no_map = {
            current_page_no: requested_page_no
            for current_page_no, (requested_page_no, _) in zip(
                sorted(document.pages), page_documents
            )
        }
        document.pages = {
            page_no_map[page_no]: page_item
            for page_no, page_item in document.pages.items()
        }
        for page_no, page_item in document.pages.items():
            page_item.page_no = page_no
        for item, _level in document.iterate_items():
            if isinstance(item, DocItem):
                for provenance in item.prov:
                    provenance.page_no = page_no_map[provenance.page_no]
        return document

    @staticmethod
    def _release_page_resources(page: Page) -> None:
        if page._backend is not None:
            page._backend.unload()
            page._backend = None
        page._image_cache = {}
        page.parsed_page = None

    def _integrate_results(
        self, conv_res: ConversionResult, proc: ProcessingResult
    ) -> None:
        """Integrate processing results into conversion result."""
        page_map = {p.page_no: p for p in proc.pages}

        conv_res.pages = [
            page_map[p.page_no] for p in conv_res.pages if p.page_no in page_map
        ]

        for page_no, error, failure in proc.failed_pages:
            conv_res.errors.append(
                failure
                or ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message=str(error) or error.__class__.__name__,
                    category=FailureCategory.UNKNOWN,
                    page_no=page_no if page_no > 0 else None,
                )
            )

        if proc.is_complete_failure:
            conv_res.status = ConversionStatus.FAILURE
        elif proc.is_partial_success:
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        else:
            conv_res.status = ConversionStatus.SUCCESS

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        return conv_res

    @classmethod
    def get_default_options(cls) -> ThreadedLayoutVlmPipelineOptions:
        return ThreadedLayoutVlmPipelineOptions()

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
