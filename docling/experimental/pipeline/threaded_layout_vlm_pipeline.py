"""Threaded Layout+VLM Pipeline
================================
A specialized two-stage threaded pipeline that combines layout model preprocessing
with VLM processing. The layout model detects document elements and coordinates,
which are then injected into the VLM prompt for enhanced structured output.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union, cast

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image as PILImage

if TYPE_CHECKING:
    from docling_core.types.doc.page import SegmentedPage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import ConversionStatus, Page
from docling.datamodel.document import ConversionResult
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
from docling.models.stages.layout.layout_model import LayoutModel
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
        self.layout_model = LayoutModel(
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            options=self.pipeline_options.layout_options,
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
        layout_stage.add_output_queue(vlm_stage.input_queue)
        vlm_stage.add_output_queue(output_q)

        stages = [layout_stage, vlm_stage]
        return RunContext(
            stages=stages, first_stage=layout_stage, output_queue=output_q
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document using threaded layout+VLM pipeline."""
        run_id = next(self._run_seq)
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        backend = conv_res.input._backend

        # Initialize pages
        start_page, end_page = conv_res.input.limits.page_range
        pages: List[Page] = []
        images_scale = self.pipeline_options.images_scale
        for i in range(conv_res.input.page_count):
            if start_page - 1 <= i <= end_page - 1:
                page = Page(page_no=i + 1)
                if images_scale is not None:
                    page._default_image_scale = images_scale
                page._backend = backend.load_page(i)
                if page._backend and page._backend.is_valid():
                    page.size = page._backend.get_size()
                    conv_res.pages.append(page)
                    pages.append(page)

        if not pages:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        total_pages = len(pages)
        ctx = self._create_run_ctx()
        for st in ctx.stages:
            st.start()

        proc = ProcessingResult(total_expected=total_pages)
        fed_idx = 0
        batch_size = 32

        try:
            while proc.success_count + proc.failure_count < total_pages:
                # Feed pages to first stage
                while fed_idx < total_pages:
                    ok = ctx.first_stage.input_queue.put(
                        ThreadedItem(
                            payload=pages[fed_idx],
                            run_id=run_id,
                            page_no=pages[fed_idx].page_no,
                            conv_res=conv_res,
                        ),
                        timeout=0.0,
                    )
                    if ok:
                        fed_idx += 1
                        if fed_idx == total_pages:
                            ctx.first_stage.input_queue.close()
                    else:
                        break

                # Drain results from output
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

                # Handle early termination
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

    def _integrate_results(
        self, conv_res: ConversionResult, proc: ProcessingResult
    ) -> None:
        """Integrate processing results into conversion result."""
        page_map = {p.page_no: p for p in proc.pages}

        # Track failed pages for cleanup
        failed_page_nos = {fp for fp, _ in proc.failed_pages}

        # Collect pages that will be removed (failed pages) for resource cleanup
        pages_to_remove = [p for p in conv_res.pages if p.page_no in failed_page_nos]

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

        # Clean up resources for failed pages that were removed
        for p in pages_to_remove:
            if p._backend is not None:
                p._backend.unload()
            p._image_cache = {}
            # Clean up parsed_page if it exists (it's Optional[SegmentedPdfPage])
            if p.parsed_page is not None:
                del p.parsed_page
                p.parsed_page = None

        # Clean up images if not needed for remaining pages
        if not self.pipeline_options.generate_page_images:
            for p in conv_res.pages:
                p._image_cache = {}

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble final document from VLM predictions."""
        from docling_core.types.doc import DocItem, ImageRef, PictureItem

        from docling.datamodel.pipeline_options_vlm_model import ResponseFormat

        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            # Response format validation is done in ThreadedLayoutVlmPipelineOptions
            # This check is kept as a safety net, but should never trigger if validation works
            if (
                self.pipeline_options.vlm_options.response_format
                != ResponseFormat.DOCTAGS
            ):
                raise RuntimeError(
                    f"Unsupported VLM response format {self.pipeline_options.vlm_options.response_format}. Only DOCTAGS format is supported."
                )
            conv_res.document = self._turn_dt_into_doc(conv_res)

            # Generate images of the requested element types
            if self.pipeline_options.generate_picture_images:
                # Create mapping from page_no to Page object since pages may be non-continuous
                page_map = {p.page_no: p for p in conv_res.pages}
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        isinstance(element, PictureItem)
                        and self.pipeline_options.generate_picture_images
                    ):
                        page_no = element.prov[0].page_no
                        page = page_map.get(page_no)
                        if page is None:
                            _log.warning(
                                f"Page {page_no} not found in conversion result for picture element. Skipping image generation."
                            )
                            continue
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )

        return conv_res

    def _turn_dt_into_doc(self, conv_res: ConversionResult) -> DoclingDocument:
        """Convert DOCTAGS response format to DoclingDocument."""
        doctags_list = []
        image_list = []
        for page in conv_res.pages:
            # Only include pages that have both an image and VLM predictions
            if page.image and page.predictions.vlm_response:
                predicted_doctags = page.predictions.vlm_response.text
                image_list.append(page.image)
                doctags_list.append(predicted_doctags)

        doctags_list_c = cast(List[Union[Path, str]], doctags_list)
        image_list_c = cast(List[Union[Path, PILImage.Image]], image_list)
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags_list_c, image_list_c
        )
        document = DoclingDocument.load_from_doctags(doctag_document=doctags_doc)

        return document

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
