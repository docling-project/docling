"""Engine option helpers for image-classification runtimes."""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from docling.datamodel.kserve_v2_options import KserveV2OptionsMixin
from docling.datamodel.settings import default_compile_model
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

    providers: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"],
        description="Ordered list of ONNX Runtime execution providers to try",
    )

    graph_optimization_level: int = Field(
        default=99,
        description=(
            "ONNX Runtime graph optimization level. "
            "Accepts onnxruntime.GraphOptimizationLevel int values: "
            "0 (ORT_DISABLE_ALL), 1 (ORT_ENABLE_BASIC), "
            "2 (ORT_ENABLE_EXTENDED), 99 (ORT_ENABLE_ALL). "
            "Default enables all optimizations including layout optimizations."
        ),
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
        default_factory=default_compile_model,
        description="Whether to compile the model with torch.compile() for better performance.",
    )


class ApiKserveV2ImageClassificationEngineOptions(
    BaseImageClassificationEngineOptions, KserveV2OptionsMixin
):
    """Runtime configuration for remote KServe v2 inference."""

    engine_type: Literal[ImageClassificationEngineType.API_KSERVE_V2] = (
        ImageClassificationEngineType.API_KSERVE_V2
    )
