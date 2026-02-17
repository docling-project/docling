"""Factory for creating object detection engines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineType,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import (
        EngineModelConfig,
        ObjectDetectionModelSpec,
    )

_log = logging.getLogger(__name__)


def create_object_detection_engine(
    *,
    options: BaseObjectDetectionEngineOptions,
    model_spec: Optional[ObjectDetectionModelSpec] = None,
    accelerator_options: AcceleratorOptions,
    artifacts_path: Optional[Union[Path, str]] = None,
) -> BaseObjectDetectionEngine:
    """Factory to create object detection engines.

    Args:
        options: Engine-specific options
        model_spec: Model specification used to derive engine configuration
        accelerator_options: Hardware accelerator configuration
        artifacts_path: Optional path to local model artifacts root

    Returns:
        Initialized engine instance (call .initialize() before use)
    """
    model_config: Optional[EngineModelConfig] = None
    if model_spec is not None:
        model_config = model_spec.get_engine_config(options.engine_type)

    if options.engine_type == ObjectDetectionEngineType.ONNXRUNTIME:
        from docling.datamodel.object_detection_engine_options import (
            OnnxRuntimeObjectDetectionEngineOptions,
        )
        from docling.models.inference_engines.object_detection.onnxruntime_engine import (
            OnnxRuntimeObjectDetectionEngine,
        )

        if not isinstance(options, OnnxRuntimeObjectDetectionEngineOptions):
            raise ValueError(
                f"Expected OnnxRuntimeObjectDetectionEngineOptions, got {type(options)}"
            )

        return OnnxRuntimeObjectDetectionEngine(
            options=options,
            model_config=model_config,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
        )

    elif options.engine_type == ObjectDetectionEngineType.TRANSFORMERS:
        from docling.datamodel.object_detection_engine_options import (
            TransformersObjectDetectionEngineOptions,
        )
        from docling.models.inference_engines.object_detection.transformers_engine import (
            TransformersObjectDetectionEngine,
        )

        if not isinstance(options, TransformersObjectDetectionEngineOptions):
            raise ValueError(
                f"Expected TransformersObjectDetectionEngineOptions, got {type(options)}"
            )

        return TransformersObjectDetectionEngine(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )

    else:
        raise ValueError(f"Unknown engine type: {options.engine_type}")
