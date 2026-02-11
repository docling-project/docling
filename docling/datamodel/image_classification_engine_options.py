"""Engine option helpers for image-classification runtimes."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngineOptions,
    ImageClassificationEngineType,
)


class OnnxRuntimeImageClassificationEngineOptions(BaseImageClassificationEngineOptions):
    """Runtime configuration for ONNX Runtime based image-classification models."""

    engine_type: Literal[ImageClassificationEngineType.ONNXRUNTIME] = (
        ImageClassificationEngineType.ONNXRUNTIME
    )

    model_filename: str = Field(
        default="model.onnx",
        description="Filename of the ONNX export inside the model repository",
    )

    input_name: Optional[str] = Field(
        default=None,
        description="Optional ONNX input tensor name. Uses first model input when omitted.",
    )

    output_name: Optional[str] = Field(
        default=None,
        description="Optional ONNX output tensor name for logits. Uses first output when omitted.",
    )

    providers: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"],
        description="Ordered list of ONNX Runtime execution providers to try",
    )


class TransformersImageClassificationEngineOptions(
    BaseImageClassificationEngineOptions
):
    """Runtime configuration for Transformers-based image-classification models."""

    engine_type: Literal[ImageClassificationEngineType.TRANSFORMERS] = (
        ImageClassificationEngineType.TRANSFORMERS
    )

    torch_dtype: str | None = Field(
        default=None,
        description="PyTorch dtype for model inference (e.g., 'float32', 'float16', 'bfloat16')",
    )

    compile_model: bool = Field(
        default=False,
        description="Whether to compile the model with torch.compile() for better performance.",
    )
