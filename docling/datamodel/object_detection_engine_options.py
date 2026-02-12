"""Engine option helpers for object-detection runtimes."""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineType,
)


class OnnxRuntimeObjectDetectionEngineOptions(BaseObjectDetectionEngineOptions):
    """Runtime configuration for ONNX Runtime based object-detection models.

    Preprocessing parameters come from HuggingFace preprocessor configs,
    not from these options.
    """

    engine_type: Literal[ObjectDetectionEngineType.ONNXRUNTIME] = (
        ObjectDetectionEngineType.ONNXRUNTIME
    )

    model_filename: str = Field(
        default="model.onnx",
        description="Filename of the ONNX export inside the model repository",
    )

    providers: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"],
        description="Ordered list of ONNX Runtime execution providers to try",
    )


class TransformersObjectDetectionEngineOptions(BaseObjectDetectionEngineOptions):
    """Runtime configuration for Transformers-based object-detection models."""

    engine_type: Literal[ObjectDetectionEngineType.TRANSFORMERS] = (
        ObjectDetectionEngineType.TRANSFORMERS
    )

    torch_dtype: str | None = Field(
        default=None,
        description="PyTorch dtype for model inference (e.g., 'float32', 'float16', 'bfloat16')",
    )
