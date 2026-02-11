"""Factory for creating image-classification engines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngine,
    BaseImageClassificationEngineOptions,
    ImageClassificationEngineType,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import (
        EngineModelConfig,
        ImageClassificationModelSpec,
    )


def create_image_classification_engine(
    *,
    options: BaseImageClassificationEngineOptions,
    model_spec: Optional[ImageClassificationModelSpec] = None,
    accelerator_options: AcceleratorOptions,
    artifacts_path: Optional[Union[Path, str]] = None,
) -> BaseImageClassificationEngine:
    """Factory to create image-classification engines."""
    model_config: Optional[EngineModelConfig] = None
    if model_spec is not None:
        model_config = model_spec.get_engine_config(options.engine_type)

    if options.engine_type == ImageClassificationEngineType.ONNXRUNTIME:
        from docling.datamodel.image_classification_engine_options import (
            OnnxRuntimeImageClassificationEngineOptions,
        )
        from docling.models.inference_engines.image_classification.onnxruntime_engine import (
            OnnxRuntimeImageClassificationEngine,
        )

        if not isinstance(options, OnnxRuntimeImageClassificationEngineOptions):
            raise ValueError(
                "Expected OnnxRuntimeImageClassificationEngineOptions, "
                f"got {type(options)}"
            )

        return OnnxRuntimeImageClassificationEngine(
            options=options,
            model_config=model_config,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
        )

    if options.engine_type == ImageClassificationEngineType.TRANSFORMERS:
        from docling.datamodel.image_classification_engine_options import (
            TransformersImageClassificationEngineOptions,
        )
        from docling.models.inference_engines.image_classification.transformers_engine import (
            TransformersImageClassificationEngine,
        )

        if not isinstance(options, TransformersImageClassificationEngineOptions):
            raise ValueError(
                "Expected TransformersImageClassificationEngineOptions, "
                f"got {type(options)}"
            )

        return TransformersImageClassificationEngine(
            options=options,
            model_config=model_config,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
        )

    raise ValueError(f"Unknown engine type: {options.engine_type}")
