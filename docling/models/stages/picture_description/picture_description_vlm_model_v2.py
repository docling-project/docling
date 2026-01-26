"""Picture description stage using the new VLM runtime system.

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
    PictureDescriptionVlmOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.runtimes.base import BaseVlmRuntime, VlmRuntimeInput
from docling.models.runtimes.factory import create_vlm_runtime

_log = logging.getLogger(__name__)


class PictureDescriptionVlmModelV2(PictureDescriptionBaseModel):
    """Picture description stage using the new runtime system.

    This stage uses the unified VLM runtime interface to generate descriptions
    for pictures in documents. It supports all runtime types (Transformers, MLX,
    API, etc.) through the runtime factory.

    The stage:
    1. Filters pictures based on size and classification thresholds
    2. Uses the runtime to generate descriptions
    3. Stores descriptions in PictureItem metadata

    Example:
        ```python
        from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions

        # Use preset with default runtime
        options = PictureDescriptionVlmOptions.from_preset("smolvlm")

        # Create stage
        stage = PictureDescriptionVlmModelV2(
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
        return PictureDescriptionVlmOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionVlmOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionVlmOptions
        self.runtime: Optional[BaseVlmRuntime] = None

        if self.enabled:
            # Check if using new runtime system
            if (
                self.options.model_spec is not None
                and self.options.runtime_options is not None
            ):
                # New runtime system path
                # Get runtime type from options
                runtime_type = self.options.runtime_options.runtime_type

                # Get model configuration for this runtime
                self.repo_id = self.options.model_spec.get_repo_id(runtime_type)
                self.revision = self.options.model_spec.get_revision(runtime_type)

                _log.info(
                    f"Initializing PictureDescriptionVlmModelV2 with runtime system: "
                    f"model={self.repo_id}, "
                    f"runtime={runtime_type.value}"
                )

                # Create runtime using factory
                self.runtime = create_vlm_runtime(self.options.runtime_options)

                # Set provenance from model spec
                self.provenance = f"{self.repo_id} ({runtime_type.value})"

            else:
                # Legacy path - fall back to old implementation
                raise ValueError(
                    "PictureDescriptionVlmModelV2 requires model_spec and runtime_options. "
                    "Use PictureDescriptionVlmOptions.from_preset() to create options, "
                    "or use the legacy PictureDescriptionVlmModel class."
                )

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

        # Process images one by one (TODO: implement batching)
        for image in images:
            try:
                # Prepare runtime input
                runtime_input = VlmRuntimeInput(
                    image=image,
                    prompt=prompt,
                    repo_id=self.repo_id,
                    temperature=0.0,
                    max_new_tokens=200,  # Use from options if available
                )

                # Generate description using runtime (call runtime as callable)
                output = self.runtime(runtime_input)

                # Extract text from output
                description = output.text.strip()

                _log.debug(f"Generated description: {description[:100]}...")

                yield description

            except Exception as e:
                _log.error(f"Error generating picture description: {e}")
                # Yield empty string on error to maintain batch alignment
                yield ""

    def __del__(self):
        """Cleanup runtime resources."""
        if self.runtime is not None:
            try:
                self.runtime.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up runtime: {e}")
