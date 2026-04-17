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
from docling.datamodel.base_models import Page, VlmPrediction, VlmStopReason
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.models.base_model import BasePageModel
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineType,
    create_vlm_engine,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


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
            rasterize_time = 0.0
            scale_resize_time = 0.0
            max_size_resize_time = 0.0

            for page in page_list:
                rasterize_start = time.perf_counter()
                image = page.image
                rasterize_time += time.perf_counter() - rasterize_start

                if image is None:
                    _log.warning(
                        f"Page {page.page_no} has no image, skipping VLM conversion"
                    )
                    continue

                # Scale image if needed
                if self.options.scale != 1.0:
                    resize_start = time.perf_counter()
                    new_size = (
                        int(image.width * self.options.scale),
                        int(image.height * self.options.scale),
                    )
                    image = image.resize(new_size, PILImage.Resampling.LANCZOS)
                    scale_resize_time += time.perf_counter() - resize_start

                # Apply max_size constraint if specified
                if self.options.max_size is not None:
                    max_dim = max(image.width, image.height)
                    if max_dim > self.options.max_size:
                        resize_start = time.perf_counter()
                        scale_factor = self.options.max_size / max_dim
                        new_size = (
                            int(image.width * scale_factor),
                            int(image.height * scale_factor),
                        )
                        image = image.resize(new_size, PILImage.Resampling.LANCZOS)
                        max_size_resize_time += time.perf_counter() - resize_start

                images.append(image)
                prompts.append(self.options.model_spec.prompt)
                valid_pages.append(page)

            if not images:
                _log.warning("No valid images to process")
                return

            # Process through runtime using batch prediction
            _log.debug(
                "Prepared %s pages for VLM engine: rasterize=%.3fs, scale_resize=%.3fs, max_size_resize=%.3fs",
                len(images),
                rasterize_time,
                scale_resize_time,
                max_size_resize_time,
            )

            try:
                # Create batch of runtime inputs (shared generation template)
                engine_inputs = self._build_engine_inputs(images, prompts)

                # Run batch inference
                batch_start = time.perf_counter()
                outputs = self.engine.predict_batch(engine_inputs)
                _log.debug(
                    "Processed %s pages through VLM engine in %.3fs",
                    len(engine_inputs),
                    time.perf_counter() - batch_start,
                )

                # Attach predictions to pages
                for page, output in zip(valid_pages, outputs):
                    # Convert string stop_reason to VlmStopReason enum
                    stop_reason = VlmStopReason.UNSPECIFIED
                    if output.stop_reason:
                        try:
                            stop_reason = VlmStopReason(output.stop_reason)
                        except ValueError:
                            stop_reason = VlmStopReason.UNSPECIFIED

                    page.predictions.vlm_response = VlmPrediction(
                        text=output.text,
                        stop_reason=stop_reason,
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
            # Convert string stop_reason to VlmStopReason enum
            stop_reason = VlmStopReason.UNSPECIFIED
            if output.stop_reason:
                try:
                    stop_reason = VlmStopReason(output.stop_reason)
                except ValueError:
                    stop_reason = VlmStopReason.UNSPECIFIED

            # Convert to VlmPrediction
            yield VlmPrediction(
                text=output.text,
                stop_reason=stop_reason,
            )

    def __del__(self):
        """Cleanup engine resources."""
        if hasattr(self, "engine"):
            try:
                self.engine.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up engine: {e}")
