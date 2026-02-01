"""Picture description stage using the VLM runtime system.

This module provides a runtime-agnostic picture description stage that can use
any VLM runtime (Transformers, MLX, API, etc.) through the unified runtime interface.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type, Union

from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmRuntimeOptions,
)
from docling.datamodel.stage_model_specs import RuntimeModelConfig
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.runtimes.base import BaseVlmRuntime, VlmRuntimeInput
from docling.models.runtimes.factory import create_vlm_runtime

_log = logging.getLogger(__name__)


class PictureDescriptionVlmRuntimeModel(PictureDescriptionBaseModel):
    """Picture description stage using the VLM runtime system.

    This stage uses the unified VLM runtime interface to generate descriptions
    for pictures in documents. It supports all runtime types (Transformers, MLX,
    API, etc.) through the runtime factory.

    The stage:
    1. Filters pictures based on size and classification thresholds
    2. Uses the runtime to generate descriptions
    3. Stores descriptions in PictureItem metadata

    Example:
        ```python
        from docling.datamodel.pipeline_options import PictureDescriptionVlmRuntimeOptions

        # Use preset with default runtime
        options = PictureDescriptionVlmRuntimeOptions.from_preset("smolvlm")

        # Create stage
        stage = PictureDescriptionVlmRuntimeModel(
            enabled=True,
            enable_remote_services=False,
            artifacts_path=None,
            options=options,
            accelerator_options=AcceleratorOptions(),
        )
        ```
    """

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionVlmRuntimeOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionVlmRuntimeOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionVlmRuntimeOptions
        self.runtime: Optional[BaseVlmRuntime] = None

        if self.enabled:
            # Get runtime type from options
            runtime_type = self.options.runtime_options.runtime_type

            # Get model configuration for this runtime (for logging)
            self.repo_id = self.options.model_spec.get_repo_id(runtime_type)
            self.revision = self.options.model_spec.get_revision(runtime_type)

            _log.info(
                f"Initializing PictureDescriptionVlmRuntimeModel with runtime system: "
                f"model={self.repo_id}, "
                f"runtime={runtime_type.value}"
            )

            # Create runtime - pass model_spec, let factory handle config generation
            self.runtime = create_vlm_runtime(
                self.options.runtime_options,
                model_spec=self.options.model_spec,
            )

            # Set provenance from model spec
            self.provenance = f"{self.repo_id} ({runtime_type.value})"

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        """Generate descriptions for a batch of images.

        Args:
            images: Iterable of PIL images to describe

        Yields:
            Description text for each image
        """
        if self.runtime is None:
            raise RuntimeError("Runtime not initialized")

        # Get prompt from options
        prompt = self.options.prompt

        # Convert to list for batch processing
        image_list = list(images)

        if not image_list:
            return

        try:
            # Prepare batch of runtime inputs
            runtime_inputs = [
                VlmRuntimeInput(
                    image=image,
                    prompt=prompt,
                    temperature=0.0,
                    max_new_tokens=200,  # Use from options if available
                )
                for image in image_list
            ]

            # Generate descriptions using batch prediction
            outputs = self.runtime.predict_batch(runtime_inputs)

            # Extract and yield descriptions
            for output in outputs:
                description = output.text.strip()
                _log.debug(f"Generated description: {description[:100]}...")
                yield description

        except Exception as e:
            _log.error(f"Error generating picture descriptions: {e}")
            # Yield empty strings on error to maintain batch alignment
            for _ in image_list:
                yield ""

    def __del__(self):
        """Cleanup runtime resources."""
        if self.runtime is not None:
            try:
                self.runtime.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up runtime: {e}")
