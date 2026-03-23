"""Engine option helpers for object-detection runtimes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from docling.datamodel.settings import default_compile_model
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


class TransformersObjectDetectionEngineOptions(BaseObjectDetectionEngineOptions):
    """Runtime configuration for Transformers-based object-detection models."""

    engine_type: Literal[ObjectDetectionEngineType.TRANSFORMERS] = (
        ObjectDetectionEngineType.TRANSFORMERS
    )

    torch_dtype: str | None = Field(
        default=None,
        description="PyTorch dtype for model inference (e.g., 'float32', 'float16', 'bfloat16')",
    )

    compile_model: bool = Field(
        default_factory=default_compile_model,
        description="Whether to compile the model with torch.compile() for better performance.",
    )


class ApiKserveV2ObjectDetectionEngineOptions(BaseObjectDetectionEngineOptions):
    """Runtime configuration for remote KServe v2 inference."""

    engine_type: Literal[ObjectDetectionEngineType.API_KSERVE_V2] = (
        ObjectDetectionEngineType.API_KSERVE_V2
    )

    url: str = Field(
        description=(
            "Endpoint URL for KServe v2 transport. "
            "For transport='http', use http(s)://host[:port] or plain host:port. "
            "For transport='grpc', use plain host:port."
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

    transport: Literal["grpc", "http"] = Field(
        default="grpc",
        description=(
            "Transport protocol for KServe v2 calls. "
            "Use 'grpc' for binary tensor payloads (default), or 'http' for JSON REST."
        ),
    )

    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for authentication/routing when transport='http'.",
    )

    grpc_metadata: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional gRPC metadata for authentication/routing when transport='grpc'. "
            "No HTTP headers are reused in gRPC mode."
        ),
    )

    grpc_use_tls: bool = Field(
        default=False,
        description=(
            "Whether to use TLS for the gRPC channel. "
            "When omitted, plain-text h2c is used."
        ),
    )

    grpc_max_message_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="Max send/receive gRPC message size in bytes.",
    )

    grpc_use_binary_data: bool = Field(
        default=True,
        description=(
            "Whether to request/expect binary tensor payloads on gRPC output tensors. "
            "Set to False for servers that do not support binary_data output parameters."
        ),
    )

    timeout: float = Field(
        default=60.0,
        description="Per-request timeout in seconds for both HTTP and gRPC calls.",
    )

    request_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional top-level KServe v2 infer request parameters.",
    )
