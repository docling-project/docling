"""Engine option helpers for image-classification runtimes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AnyUrl, Field

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


class ApiKserveV2ImageClassificationEngineOptions(BaseImageClassificationEngineOptions):
    """Runtime configuration for remote KServe v2 inference."""

    engine_type: Literal[ImageClassificationEngineType.API_KSERVE_V2] = (
        ImageClassificationEngineType.API_KSERVE_V2
    )

    url: AnyUrl = Field(
        description=(
            "Base URL of the KServe v2 server (e.g., 'http://localhost:8000'). "
            "The full endpoint path is constructed automatically as "
            "/v2/models/{model_name}[/versions/{version}]/infer."
        ),
    )

    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Remote model name registered in the KServe v2 endpoint. "
            "If omitted, a repo_id-derived default is used."
        ),
    )

    model_version: Optional[str] = Field(
        default=None,
        description="Optional model version. If omitted, the server default is used.",
    )

    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for authentication/routing.",
    )

    timeout: float = Field(
        default=60.0,
        description="HTTP request timeout in seconds.",
    )

    request_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional top-level KServe v2 infer request parameters.",
    )
