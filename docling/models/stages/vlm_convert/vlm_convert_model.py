# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""VLM-based document conversion stage using the new runtime system.

This stage converts document pages to structured formats (DocTags, Markdown, etc.)
using vision-language models through a pluggable runtime system.
"""

import logging
import time
from collections.abc import Iterable
from pathlib import Path

from PIL import Image as PILImage

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Page,
    VlmPrediction,
    VlmPredictionToken,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.models.base_model import BasePageModel
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
    create_vlm_engine,
)
from docling.utils.mineru_utils import (
    MINERU2_LAYOUT_PROMPT,
    parse_mineru2_layout,
    prepare_mineru2_crops,
    prepare_mineru2_layout_image,
    serialize_mineru2_regions,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)
_VLM_STOP_REASON_VALUES = {reason.value for reason in VlmStopReason}


def _prediction_from_engine_output(output: VlmEngineOutput) -> VlmPrediction:
    stop_reason = VlmStopReason.UNSPECIFIED
    if output.stop_reason in _VLM_STOP_REASON_VALUES:
        stop_reason = VlmStopReason(output.stop_reason)

    metadata = output.metadata or {}

    generated_tokens = []

    logprobs = metadata.get("logprobs")
    if logprobs and logprobs.content:
        generated_tokens = [
            VlmPredictionToken(
                text=token.token,
                logprob=token.logprob,
            )
            for token in logprobs.content
        ]

    return VlmPrediction(
        text=output.text,
        stop_reason=stop_reason,
        generation_time=metadata.get("generation_time", -1),
        num_tokens=metadata.get("num_tokens"),
        usage=metadata.get("usage"),
        generated_tokens=generated_tokens,
    )


class VlmConvertModel(BasePageModel):
    """Stage for VLM-based document conversion using the new runtime system.

    This stage:
    1. Takes document pages with images
    2. Processes them through a VLM runtime (transformers, mlx, api, etc.)
    3. Returns pages with VLM predictions attached

    The actual model inference is delegated to the runtime layer, making this
    stage runtime-agnostic.
    """

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Path | str | None,
        options: VlmConvertOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initialize the VLM convert stage.

        Args:
            enabled: Whether this stage is enabled
            options: Configuration options including model spec and runtime options
        """
        self.enabled = enabled
        self.options = options

        if not self.enabled:
            return

        # Get engine type from options
        engine_type = options.engine_options.engine_type

        # Get model configuration for this engine (for logging)
        self.repo_id = options.model_spec.get_repo_id(engine_type)
        self.revision = options.model_spec.get_revision(engine_type)

        _log.info(
            f"Initializing VlmConvertModel with engine={engine_type.value}, "
            f"model={self.repo_id}, revision={self.revision}"
        )

        # Create the engine - pass model_spec, let factory handle config generation
        self.engine: BaseVlmEngine = create_vlm_engine(
            options=self.options.engine_options,
            model_spec=self.options.model_spec,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
            enable_remote_services=enable_remote_services,
        )

        _log.info("VlmConvertModel initialized successfully")

    def _resolve_runtime_engine_type(self) -> VlmEngineType:
        selected_engine_type = getattr(self.engine, "selected_engine_type", None)
        if selected_engine_type is not None:
            return selected_engine_type
        return self.options.engine_options.engine_type

    def _build_engine_inputs(
        self,
        images: list[PILImage.Image],
        prompts: list[str],
    ) -> list[VlmEngineInput]:
        """Build a batch of ``VlmEngineInput`` sharing one generation-config template.

        Stop strings and the runtime generation config are identical for every
        page in a batch, so building them once here avoids reallocating them
        per item.
        """
        model_spec = self.options.model_spec
        runtime_engine_type = self._resolve_runtime_engine_type()
        stop_strings = list(model_spec.stop_strings)
        extra_generation_config = model_spec.get_runtime_input_extra_config(
            runtime_engine_type
        )
        return [
            VlmEngineInput(
                image=image,
                prompt=prompt,
                temperature=model_spec.temperature,
                max_new_tokens=model_spec.max_new_tokens,
                stop_strings=stop_strings,
                extra_generation_config=extra_generation_config,
            )
            for image, prompt in zip(images, prompts)
        ]

    @staticmethod
    def _mineru2_stop_reason(
        layout_output: VlmEngineOutput,
        recognition_outputs: list[VlmEngineOutput],
    ) -> str | None:
        incomplete_reasons = {
            VlmStopReason.LENGTH.value,
            VlmStopReason.CONTENT_FILTERED.value,
        }
        outputs = [layout_output, *recognition_outputs]
        for output in outputs:
            if output.stop_reason in incomplete_reasons:
                return output.stop_reason
        return layout_output.stop_reason

    def _predict_mineru2(self, images: list[PILImage.Image]) -> list[VlmEngineOutput]:
        """Run MinerU2 layout detection followed by region recognition."""
        layout_images = [prepare_mineru2_layout_image(image) for image in images]
        layout_inputs = self._build_engine_inputs(
            layout_images, [MINERU2_LAYOUT_PROMPT] * len(layout_images)
        )
        layout_outputs = self.engine.predict_batch(layout_inputs)
        if len(layout_outputs) != len(images):
            raise RuntimeError(
                "MinerU2 layout output count does not match the input page count"
            )

        regions_by_page = [
            parse_mineru2_layout(output.text) for output in layout_outputs
        ]
        crop_images: list[PILImage.Image] = []
        crop_prompts: list[str] = []
        crop_targets: list[tuple[int, int]] = []
        for page_index, (image, regions) in enumerate(zip(images, regions_by_page)):
            for crop in prepare_mineru2_crops(image, regions):
                crop_images.append(crop.image)
                crop_prompts.append(crop.prompt)
                crop_targets.append((page_index, crop.region_index))

        recognition_outputs = (
            self.engine.predict_batch(
                self._build_engine_inputs(crop_images, crop_prompts)
            )
            if crop_images
            else []
        )
        if len(recognition_outputs) != len(crop_targets):
            raise RuntimeError(
                "MinerU2 recognition output count does not match the region count"
            )

        outputs_by_page: list[list[VlmEngineOutput]] = [[] for _image in images]
        for (page_index, region_index), output in zip(
            crop_targets, recognition_outputs
        ):
            regions_by_page[page_index][region_index].content = output.text
            outputs_by_page[page_index].append(output)

        combined_outputs = []
        for regions, layout_output, page_outputs in zip(
            regions_by_page, layout_outputs, outputs_by_page
        ):
            all_outputs = [layout_output, *page_outputs]
            combined_outputs.append(
                VlmEngineOutput(
                    text=serialize_mineru2_regions(regions),
                    stop_reason=self._mineru2_stop_reason(layout_output, page_outputs),
                    metadata={
                        "generation_time": sum(
                            output.metadata.get("generation_time") or 0
                            for output in all_outputs
                        ),
                        "num_tokens": sum(
                            output.metadata.get("num_tokens") or 0
                            for output in all_outputs
                        ),
                    },
                )
            )
        return combined_outputs

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Process a batch of pages through the VLM engine.

        Args:
            conv_res: Conversion result context
            page_batch: Batch of pages to process

        Yields:
            Pages with VLM predictions attached
        """
        if not self.enabled:
            yield from page_batch
            return

        page_list = list(page_batch)
        if not page_list:
            return

        with TimeRecorder(conv_res, "vlm_convert"):
            # Prepare images and prompts
            images = []
            prompts = []
            valid_pages = []
            image_prep_time = 0.0

            for page in page_list:
                image_prep_start = time.perf_counter()
                image = page.get_image(
                    scale=self.options.scale,
                    max_size=self.options.max_size,
                )
                image_prep_time += time.perf_counter() - image_prep_start
                if image is None:
                    _log.warning(
                        f"Page {page.page_no} has no image, skipping VLM conversion"
                    )
                    continue

                images.append(image)
                prompts.append(self.options.model_spec.prompt)
                valid_pages.append(page)

            if not images:
                _log.warning("No valid images to process")
                return

            # Process through runtime using batch prediction
            _log.debug(
                "Prepared %s pages for VLM engine in %.3fs",
                len(images),
                image_prep_time,
            )

            try:
                # Run batch inference
                batch_start = time.perf_counter()
                if self.options.model_spec.response_format == ResponseFormat.MINERU2:
                    outputs = self._predict_mineru2(images)
                else:
                    engine_inputs = self._build_engine_inputs(images, prompts)
                    outputs = self.engine.predict_batch(engine_inputs)
                batch_time = time.perf_counter() - batch_start

                # Engines report generated (not prompt) token counts, so this is
                # decode throughput for the batch.
                batch_tokens = sum(
                    output.metadata.get("num_tokens") or 0 for output in outputs
                )
                _log.info(
                    "Processed %s page(s): %s tokens in %.2f sec. (%.2f tok/s)",
                    len(images),
                    batch_tokens,
                    batch_time,
                    batch_tokens / batch_time if batch_time > 0 else 0.0,
                )

                # Attach predictions to pages
                for page, output in zip(valid_pages, outputs):
                    page.predictions.vlm_response = _prediction_from_engine_output(
                        output
                    )
                    _log.debug(
                        f"Page {page.page_no}: Generated {len(output.text)} chars, "
                        f"stop_reason={output.stop_reason}"
                    )

            except Exception as e:
                _log.error(f"Error processing pages through VLM engine: {e}")
                raise

        # Yield all pages (including those that were skipped)
        yield from page_list

    def process_images(
        self,
        image_batch: Iterable[PILImage.Image],
        prompt: str | list[str],
    ) -> Iterable[VlmPrediction]:
        """Process raw images without page metadata.

        This method provides a simpler interface for processing images directly,
        useful for testing or when page metadata is not available.

        Args:
            image_batch: Iterable of PIL Images
            prompt: Either a single prompt string or list of prompts (one per image)

        Yields:
            VLM predictions for each image

        Raises:
            ValueError: If prompt list length doesn't match image count
        """
        if not self.enabled:
            return

        images = list(image_batch)
        if not images:
            return

        # Handle prompt
        if isinstance(prompt, str):
            prompts = [prompt] * len(images)
        else:
            if len(prompt) != len(images):
                raise ValueError(
                    f"Prompt list length ({len(prompt)}) must match "
                    f"image count ({len(images)})"
                )
            prompts = prompt

        # Process batch of images (shared generation template)
        engine_inputs = self._build_engine_inputs(images, prompts)

        # Run batch inference
        outputs = self.engine.predict_batch(engine_inputs)

        # Convert outputs to VlmPredictions
        for output in outputs:
            yield _prediction_from_engine_output(output)

    def __del__(self):
        """Cleanup engine resources."""
        if hasattr(self, "engine"):
            try:
                self.engine.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up engine: {e}")
