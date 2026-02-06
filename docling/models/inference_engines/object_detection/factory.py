"""Factory for creating object detection engines."""

import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineType,
)

_log = logging.getLogger(__name__)


def create_object_detection_engine(
    options: BaseObjectDetectionEngineOptions,
    accelerator_options: Optional[AcceleratorOptions] = None,
    artifacts_path: Optional[Path] = None,
    model_config: Optional[object] = None,
) -> BaseObjectDetectionEngine:
    """Factory to create object detection engines.

    Args:
        options: Engine-specific options
        accelerator_options: Hardware accelerator configuration
        artifacts_path: Path to cached model artifacts
        model_config: Model configuration (repo_id, revision, extra_config)

    Returns:
        Initialized engine instance (call .initialize() before use)
    """
    if options.engine_type == ObjectDetectionEngineType.ONNXRUNTIME:
        from docling.models.inference_engines.object_detection.onnxruntime_engine import (
            OnnxRuntimeObjectDetectionEngine,
        )

        return OnnxRuntimeObjectDetectionEngine(
            options,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
            model_config=model_config,
        )

    elif options.engine_type == ObjectDetectionEngineType.TRANSFORMERS:
        raise NotImplementedError(
            "Transformers engine for object detection not yet implemented"
        )

    else:
        raise ValueError(f"Unknown engine type: {options.engine_type}")
