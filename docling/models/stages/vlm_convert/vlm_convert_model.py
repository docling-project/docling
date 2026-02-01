"""VLM-based document conversion stage using the new runtime system.

This stage converts document pages to structured formats (DocTags, Markdown, etc.)
using vision-language models through a pluggable runtime system.
"""

import logging
from collections.abc import Iterable

from PIL import Image as PILImage

from docling.datamodel.base_models import Page, VlmPrediction, VlmStopReason
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.models.base_model import BasePageModel
from docling.models.runtimes.base import (
    BaseVlmRuntime,
    VlmRuntimeInput,
)
from docling.models.runtimes.factory import create_vlm_runtime
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
        options: VlmConvertOptions,
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

        # Get runtime type from options
        runtime_type = options.runtime_options.runtime_type

        # Get model configuration for this runtime (for logging)
        self.repo_id = options.model_spec.get_repo_id(runtime_type)
        self.revision = options.model_spec.get_revision(runtime_type)

        _log.info(
            f"Initializing VlmConvertModel with runtime={runtime_type.value}, "
            f"model={self.repo_id}, revision={self.revision}"
        )

        # Create the runtime - pass model_spec, let factory handle config generation
        self.runtime: BaseVlmRuntime = create_vlm_runtime(
            options.runtime_options,
            model_spec=options.model_spec,
        )

        _log.info("VlmConvertModel initialized successfully")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Process a batch of pages through the VLM runtime.

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

            for page in page_list:
                if page.image is None:
                    _log.warning(
                        f"Page {page.page_no} has no image, skipping VLM conversion"
                    )
                    continue

                # Scale image if needed
                image = page.image
                if self.options.scale != 1.0:
                    new_size = (
                        int(image.width * self.options.scale),
                        int(image.height * self.options.scale),
                    )
                    image = image.resize(new_size, PILImage.Resampling.LANCZOS)

                # Apply max_size constraint if specified
                if self.options.max_size is not None:
                    max_dim = max(image.width, image.height)
                    if max_dim > self.options.max_size:
                        scale_factor = self.options.max_size / max_dim
                        new_size = (
                            int(image.width * scale_factor),
                            int(image.height * scale_factor),
                        )
                        image = image.resize(new_size, PILImage.Resampling.LANCZOS)

                images.append(image)
                prompts.append(self.options.model_spec.prompt)
                valid_pages.append(page)

            if not images:
                _log.warning("No valid images to process")
                return

            # Process through runtime using batch prediction
            _log.debug(f"Processing {len(images)} pages through VLM runtime (batched)")

            try:
                # Create batch of runtime inputs
                runtime_inputs = [
                    VlmRuntimeInput(
                        image=img,
                        prompt=prompt,
                        temperature=0.0,  # Use from options if needed
                        max_new_tokens=4096,  # Use from options if needed
                    )
                    for img, prompt in zip(images, prompts)
                ]

                # Run batch inference
                outputs = self.runtime.predict_batch(runtime_inputs)

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
                _log.error(f"Error processing pages through VLM runtime: {e}")
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

        # Process batch of images
        runtime_inputs = [
            VlmRuntimeInput(
                image=img,
                prompt=p,
                temperature=0.0,
                max_new_tokens=4096,
            )
            for img, p in zip(images, prompts)
        ]

        # Run batch inference
        outputs = self.runtime.predict_batch(runtime_inputs)

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
        """Cleanup runtime resources."""
        if hasattr(self, "runtime"):
            try:
                self.runtime.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up runtime: {e}")
