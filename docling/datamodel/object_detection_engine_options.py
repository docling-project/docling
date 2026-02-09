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

    image_input_name: str = Field(
        default="images",
        description="Name of the tensor input that receives the image batch",
    )

    sizes_input_name: str = Field(
        default="orig_target_sizes",
        description="Name of the tensor input that receives the input sizes",
    )

    score_threshold: float = Field(
        default=0.3,
        description="Minimum confidence score to keep a detection (0.0 to 1.0)",
    )


class TransformersObjectDetectionEngineOptions(BaseObjectDetectionEngineOptions):
    """Placeholder for future Transformers engine support."""

    engine_type: Literal[ObjectDetectionEngineType.TRANSFORMERS] = (
        ObjectDetectionEngineType.TRANSFORMERS
    )

    # TBD: Add transformers-specific options when implemented
