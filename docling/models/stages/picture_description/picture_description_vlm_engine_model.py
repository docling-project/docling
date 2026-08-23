"""Picture description stage using the VLM engine system.

This module provides an engine-agnostic picture description stage that can use
any VLM engine (Transformers, MLX, API, etc.) through the unified engine interface.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type, Union

from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmEngineOptions,
)
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineType,
    create_vlm_engine,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel

_log = logging.getLogger(__name__)


class PictureDescriptionVlmEngineModel(PictureDescriptionBaseModel):
    """Picture description stage using the VLM engine system.

    This stage uses the unified VLM engine interface to generate descriptions
    for pictures in documents. It supports all engine types (Transformers, MLX,
    API, etc.) through the engine factory.

    The stage:
    1. Filters pictures based on size and classification thresholds
    2. Uses the engine to generate descriptions
    3. Stores descriptions in PictureItem metadata

    Example:
        ```python
        from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions

        # Use preset with default engine
        options = PictureDescriptionVlmEngineOptions.from_preset("smolvlm")

        # Create stage
        stage = PictureDescriptionVlmEngineModel(
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
        return PictureDescriptionVlmEngineOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionVlmEngineOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionVlmEngineOptions
        self.engine: Optional[BaseVlmEngine] = None

        if self.enabled:
            # Get engine type from options
            engine_type = self.options.engine_options.engine_type

            # Get model configuration for this engine (for logging)
            self.repo_id = self.options.model_spec.get_repo_id(engine_type)
            self.revision = self.options.model_spec.get_revision(engine_type)

            _log.info(
                f"Initializing PictureDescriptionVlmEngineModel with engine system: "
                f"model={self.repo_id}, "
                f"engine={engine_type.value}"
            )

            # Create engine - pass model_spec, let factory handle config generation
            self.engine = create_vlm_engine(
                options=self.options.engine_options,
                model_spec=self.options.model_spec,
                accelerator_options=accelerator_options,
                artifacts_path=artifacts_path,
                enable_remote_services=enable_remote_services,
            )

            # Set provenance from model spec
            self.provenance = f"{self.repo_id} ({engine_type.value})"

    def _resolve_runtime_engine_type(self) -> VlmEngineType:
        selected_engine_type = getattr(self.engine, "selected_engine_type", None)
        if selected_engine_type is not None:
            return selected_engine_type
        return self.options.engine_options.engine_type

    def _build_engine_inputs(
        self, image_list: list[Image.Image]
    ) -> list[VlmEngineInput]:
        """Build a batch of ``VlmEngineInput`` sharing one generation-config template.

        Generation config is derived from the model spec, but can be overridden by the stage options.

        Args:
            image_list: list of Images to build Engine Inputs for.

        Returns:
            List of VlmEngineInput objects for batch prediction.
        """
        prompt = self.options.prompt
        model_spec = self.options.model_spec
        runtime_engine_type = self._resolve_runtime_engine_type()

        stop_strings = list(model_spec.stop_strings)
        extra_generation_config = model_spec.get_runtime_input_extra_config(
            runtime_engine_type
        )

        # if present generation_config overrides model_spec defaults
        gen_cfg = self.options.generation_config or {}
        temperature = gen_cfg.get("temperature", model_spec.temperature)
        max_new_tokens = gen_cfg.get("max_new_tokens", model_spec.max_new_tokens)

        return [
            VlmEngineInput(
                image=image,
                prompt=prompt,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                stop_strings=stop_strings,
                extra_generation_config=extra_generation_config,
            )
            for image in image_list
        ]

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        """Generate descriptions for a batch of images.

        Args:
            images: Iterable of PIL images to describe

        Yields:
            Description text for each image
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        # Convert to list for batch processing
        # TODO: Consider using chunking here
        image_list = list(images)

        if not image_list:
            return

        try:
            # Prepare batch of engine inputs
            engine_inputs = self._build_engine_inputs(image_list)

            # Generate descriptions using batch prediction
            outputs = self.engine.predict_batch(engine_inputs)

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
        """Cleanup engine resources."""
        if self.engine is not None:
            try:
                self.engine.cleanup()
            except Exception as e:
                _log.warning(f"Error cleaning up engine: {e}")
