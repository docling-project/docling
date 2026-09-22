# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Crop-based Layout+VLM Pipeline
=================================
The layout model finds the regions of each page, a VLM recognizes every region
from its crop, and the standard assembly and reading-order stages pack the
results into a `DoclingDocument`. OCR and the table-structure model are replaced
by the VLM; everything downstream of the page stages is the standard PDF
pipeline.
"""

from __future__ import annotations

import logging

from docling_core.types.doc import DocItemLabel

from docling.datamodel.document import ConversionResult
from docling.experimental.datamodel.layout_crop_vlm_pipeline_options import (
    LayoutCropVlmPipelineOptions,
)
from docling.experimental.models.layout_crop_vlm_model import LayoutCropVlmModel
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.pipeline.standard_pdf_pipeline import (
    PreprocessThreadedStage,
    RunContext,
    StandardPdfPipeline,
    ThreadedPipelineStage,
    ThreadedQueue,
)

_log = logging.getLogger(__name__)


class LayoutCropVlmPipeline(StandardPdfPipeline):
    """Layout model → per-region VLM recognition → standard assembly."""

    def __init__(self, pipeline_options: LayoutCropVlmPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: LayoutCropVlmPipelineOptions = pipeline_options

    def _init_models(self) -> None:
        super()._init_models()
        opts: LayoutCropVlmPipelineOptions = self.pipeline_options  # type: ignore[assignment]

        # All content comes from the VLM, so the native text cells are not decoded.
        self.preprocessing_model = PagePreprocessingModel(
            options=PagePreprocessingOptions(
                images_scale=opts.images_scale,
                skip_cell_extraction=True,
            )
        )
        self.crop_vlm_model = LayoutCropVlmModel(
            enabled=True,
            enable_remote_services=opts.enable_remote_services,
            artifacts_path=self.artifacts_path,
            options=opts.crop_vlm_options,
            accelerator_options=opts.accelerator_options,
        )

    def _create_run_ctx(self) -> RunContext:
        opts: LayoutCropVlmPipelineOptions = self.pipeline_options  # type: ignore[assignment]
        timed_out_run_ids: set[int] = set()
        preprocess = PreprocessThreadedStage(
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            model=self.preprocessing_model,
            shutdown_timeout=opts.stage_shutdown_timeout_seconds,
            timed_out_run_ids=timed_out_run_ids,
        )

        def _stage(name: str, model: object, batch_size: int) -> ThreadedPipelineStage:
            return ThreadedPipelineStage(
                name=name,
                model=model,
                batch_size=batch_size,
                batch_timeout=opts.batch_polling_interval_seconds,
                queue_max_size=opts.queue_max_size,
                shutdown_timeout=opts.stage_shutdown_timeout_seconds,
                timed_out_run_ids=timed_out_run_ids,
            )

        layout = _stage("layout", self.layout_model, opts.layout_batch_size)
        layout_postprocess = _stage(
            "layout_postprocess", self.layout_postprocessing_model, 1
        )
        crop_vlm = _stage("crop_vlm", self.crop_vlm_model, opts.crop_vlm_batch_size)
        assemble = ThreadedPipelineStage(
            name="assemble",
            model=self.assemble_model,
            batch_size=1,
            batch_timeout=opts.batch_polling_interval_seconds,
            queue_max_size=opts.queue_max_size,
            shutdown_timeout=opts.stage_shutdown_timeout_seconds,
            postprocess=self._release_page_resources,
            timed_out_run_ids=timed_out_run_ids,
        )

        output_q = ThreadedQueue(opts.queue_max_size)
        preprocess.add_output_queue(layout.input_queue)
        layout.add_output_queue(layout_postprocess.input_queue)
        layout_postprocess.add_output_queue(crop_vlm.input_queue)
        crop_vlm.add_output_queue(assemble.input_queue)
        assemble.add_output_queue(output_q)

        return RunContext(
            stages=[preprocess, layout, layout_postprocess, crop_vlm, assemble],
            first_stage=preprocess,
            output_queue=output_q,
            timed_out_run_ids=timed_out_run_ids,
        )

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        conv_res = super()._assemble_document(conv_res)
        # The reading-order stage parks a formula's recognized content in `orig`
        # and leaves `text` for a formula enrichment model. Here the VLM already
        # produced the LaTeX, so it is the text.
        for item in conv_res.document.texts:
            if item.label == DocItemLabel.FORMULA and not item.text:
                item.text = item.orig
        return conv_res

    @classmethod
    def get_default_options(cls) -> LayoutCropVlmPipelineOptions:
        return LayoutCropVlmPipelineOptions()
