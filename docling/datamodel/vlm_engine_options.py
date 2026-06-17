"""Engine options for VLM inference.

This module defines engine-specific configuration options that are independent
of model specifications and prompts.
"""

import logging
from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import AnyUrl, Field

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.settings import default_compile_model
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngineOptions,
    VlmEngineType,
)

_log = logging.getLogger(__name__)


# =============================================================================
# AUTO_INLINE ENGINE OPTIONS
# =============================================================================


class AutoInlineVlmEngineOptions(BaseVlmEngineOptions):
    """Options for auto-selecting the best local inference engine.

    Automatically selects the best available local engine based on:
    - Platform (macOS -> MLX, Linux/Windows -> Transformers/VLLM)
    - Available hardware (CUDA, MPS, CPU)
    - Model support
    """

    engine_type: Literal[VlmEngineType.AUTO_INLINE] = VlmEngineType.AUTO_INLINE

    prefer_vllm: bool = Field(
        default=False,
        description="Prefer VLLM over Transformers when both are available on CUDA",
    )


# =============================================================================
# TRANSFORMERS ENGINE OPTIONS
# =============================================================================


class TransformersVlmEngineOptions(BaseVlmEngineOptions):
    """Options for HuggingFace Transformers inference engine."""

    engine_type: Literal[VlmEngineType.TRANSFORMERS] = VlmEngineType.TRANSFORMERS

    device: Optional[AcceleratorDevice] = Field(
        default=None, description="Device to use (auto-detected if None)"
    )

    load_in_8bit: bool = Field(
        default=True, description="Load model in 8-bit precision using bitsandbytes"
    )

    llm_int8_threshold: float = Field(
        default=6.0, description="Threshold for LLM.int8() quantization"
    )

    quantized: bool = Field(
        default=False, description="Whether the model is pre-quantized"
    )

    torch_dtype: Optional[str] = Field(
        default=None, description="PyTorch dtype (e.g., 'float16', 'bfloat16')"
    )

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )

    use_kv_cache: bool = Field(
        default=True, description="Enable key-value caching for attention"
    )

    compile_model: bool = Field(
        default_factory=default_compile_model,
        description="Whether to compile the model with torch.compile() for better performance.",
    )


# =============================================================================
# MLX ENGINE OPTIONS
# =============================================================================


class MlxVlmEngineOptions(BaseVlmEngineOptions):
    """Options for Apple MLX inference engine (Apple Silicon only)."""

    engine_type: Literal[VlmEngineType.MLX] = VlmEngineType.MLX

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )


# =============================================================================
# VLLM ENGINE OPTIONS
# =============================================================================


class VllmCudaGraphMode(str, Enum):
    """CUDA graph capture mode for the vLLM v1 engine.

    Controls whether and how vLLM captures CUDA graphs to speed up inference.
    CUDA graphs reduce kernel-launch overhead by replaying a recorded sequence
    of CUDA operations instead of launching each kernel individually.

    NONE:
        Disable CUDA graphs entirely; everything runs in eager mode.
        Fastest startup, lowest steady-state throughput.
        Best for short-lived processes, notebooks, and debugging.

    FULL:
        Capture the entire forward pass as one monolithic CUDA graph.
        Maximum graph coverage but requires very static execution shapes;
        may fail with some models or dynamic workloads.

    PIECEWISE:
        Capture segments of the model (e.g. transformer blocks) as multiple
        smaller graphs between selected ops.  Handles dynamic shapes better
        than FULL while still accelerating most of the forward pass.

    FULL_AND_PIECEWISE:
        Hybrid mode (default in many vLLM versions): FULL graphs for
        decode-only batches; PIECEWISE graphs for prefill and mixed
        prefill+decode batches.  Usually the best throughput option for
        typical LLM serving workloads.

    FULL_DECODE_ONLY:
        FULL CUDA graphs only for decode batches; prefill and mixed batches
        run in eager mode.  Dramatically reduces graph-capture time and
        memory footprint compared to FULL_AND_PIECEWISE while still
        accelerating token generation.
    """

    NONE = "NONE"
    FULL = "FULL"
    PIECEWISE = "PIECEWISE"
    FULL_AND_PIECEWISE = "FULL_AND_PIECEWISE"
    FULL_DECODE_ONLY = "FULL_DECODE_ONLY"


class VllmVlmEngineOptions(BaseVlmEngineOptions):
    """Options for vLLM inference engine (high-throughput serving)."""

    engine_type: Literal[VlmEngineType.VLLM] = VlmEngineType.VLLM

    device: Optional[AcceleratorDevice] = Field(
        default=None, description="Device to use (auto-detected if None)"
    )

    tensor_parallel_size: int = Field(
        default=1, description="Number of GPUs for tensor parallelism"
    )

    gpu_memory_utilization: float = Field(
        default=0.9, description="Fraction of GPU memory to use"
    )

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )

    cudagraph_mode: VllmCudaGraphMode = Field(
        default=VllmCudaGraphMode.PIECEWISE,
        description=(
            "CUDA graph capture mode (vLLM v1 engine only). "
            "See VllmCudaGraphMode for the available options and their trade-offs."
        ),
    )

    model_impl: str = Field(
        default="auto",
        description=(
            "vLLM model implementation backend. "
            "Accepted values depend on the installed vLLM version; common values are "
            "'auto', 'vllm', and 'transformers'. "
            "'auto' uses vLLM's native implementation when available and otherwise falls back "
            "to the Transformers modeling backend; 'vllm' forces the native implementation; "
            "'transformers' forces the Transformers modeling backend."
        ),
    )


# =============================================================================
# API ENGINE OPTIONS
# =============================================================================


class ApiVlmEngineOptions(BaseVlmEngineOptions):
    """Options for API-based VLM services.

    Supports multiple API variants:
    - Generic OpenAI-compatible API
    - Ollama
    - LM Studio
    - OpenAI
    """

    engine_type: VlmEngineType = Field(
        default=VlmEngineType.API, description="API variant to use"
    )

    url: AnyUrl = Field(
        default=AnyUrl("http://localhost:11434/v1/chat/completions"),
        description="API endpoint URL",
    )

    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers for authentication"
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional API parameters (model, max_tokens, etc.)",
    )

    timeout: float = Field(default=60.0, description="Request timeout in seconds")

    concurrency: int = Field(default=1, description="Number of concurrent requests")

    def __init__(self, **data):
        """Initialize with default URLs based on engine type."""
        if "engine_type" in data and "url" not in data:
            engine_type = data["engine_type"]
            if engine_type == VlmEngineType.API_OLLAMA:
                data["url"] = "http://localhost:11434/v1/chat/completions"
            elif engine_type == VlmEngineType.API_LMSTUDIO:
                data["url"] = "http://localhost:1234/v1/chat/completions"
            elif engine_type == VlmEngineType.API_OPENAI:
                data["url"] = "https://api.openai.com/v1/chat/completions"

        super().__init__(**data)
