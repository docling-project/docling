from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Union

from docling_core.types.doc.page import SegmentedPage
from pydantic import AnyUrl, BaseModel, ConfigDict, Field
from transformers import StoppingCriteria
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.models.utils.generation_utils import GenerationStopper

if TYPE_CHECKING:
    from docling_core.types.doc.page import SegmentedPage

    from docling.datamodel.base_models import Page


class BaseVlmOptions(BaseModel):
    """Base configuration for vision-language models."""

    kind: Annotated[
        str,
        Field(
            description=(
                "Type identifier for the VLM options. Used for discriminating between different VLM "
                "configurations."
            ),
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            description=(
                "Prompt template for the vision-language model. Guides the model's output format and content focus."
            ),
        ),
    ]
    scale: float = Field(
        default=2.0,
        description=(
            "Scaling factor for image resolution before processing. Higher values provide more detail but increase "
            "processing time and memory usage. Range: 0.5-4.0 typical."
        ),
    )
    max_size: Optional[int] = Field(
        default=None,
        description=(
            "Maximum image dimension (width or height) in pixels. Images larger than this are resized while "
            "maintaining aspect ratio. If None, no size limit is enforced."
        ),
    )
    temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for text generation. 0.0 uses greedy decoding (deterministic), higher values "
            "(e.g., 0.7-1.0) increase randomness. Recommended: 0.0 for consistent outputs."
        ),
    )

    def build_prompt(
        self,
        page: Optional["SegmentedPage"],
        *,
        _internal_page: Optional["Page"] = None,
    ) -> str:
        """Build the prompt for VLM inference.

        Args:
            page: The parsed/segmented page to process.
            _internal_page: Internal parameter for experimental layout-aware pipelines.
                Do not rely on this in user code - subject to change.

        Returns:
            The formatted prompt string.
        """
        return self.prompt

    def decode_response(self, text: str) -> str:
        return text


class ResponseFormat(str, Enum):
    DOCTAGS = "doctags"
    MARKDOWN = "markdown"
    DEEPSEEKOCR_MARKDOWN = "deepseekocr_markdown"
    HTML = "html"
    OTSL = "otsl"
    PLAINTEXT = "plaintext"


class InferenceFramework(str, Enum):
    MLX = "mlx"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


class TransformersModelType(str, Enum):
    AUTOMODEL = "automodel"
    AUTOMODEL_VISION2SEQ = "automodel-vision2seq"
    AUTOMODEL_CAUSALLM = "automodel-causallm"
    AUTOMODEL_IMAGETEXTTOTEXT = "automodel-imagetexttotext"


class TransformersPromptStyle(str, Enum):
    CHAT = "chat"
    RAW = "raw"
    NONE = "none"


class InlineVlmOptions(BaseVlmOptions):
    """Configuration for inline vision-language models running locally."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["inline_model_options"] = "inline_model_options"
    repo_id: Annotated[
        str,
        Field(
            description=(
                "HuggingFace model repository ID for the vision-language model. Must be a model capable of "
                "processing images and generating text."
            ),
            examples=["Qwen/Qwen2-VL-2B-Instruct", "ibm-granite/granite-vision-3.3-2b"],
        ),
    ]
    revision: str = Field(
        default="main",
        description=(
            "Git revision (branch, tag, or commit hash) of the model repository. Allows pinning to specific "
            "model versions for reproducibility."
        ),
        examples=["main", "v1.0.0"],
    )
    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Allow execution of custom code from the model repository. Required for some models with custom "
            "architectures. Enable only for trusted sources due to security implications."
        ),
    )
    load_in_8bit: bool = Field(
        default=True,
        description=(
            "Load model weights in 8-bit precision using bitsandbytes quantization. Reduces memory usage by ~50% "
            "with minimal accuracy loss. Requires bitsandbytes library and CUDA."
        ),
    )
    llm_int8_threshold: float = Field(
        default=6.0,
        description=(
            "Threshold for LLM.int8() quantization outlier detection. Values with magnitude above this threshold "
            "are kept in float16 for accuracy. Lower values increase quantization but may reduce quality."
        ),
    )
    quantized: bool = Field(
        default=False,
        description=(
            "Indicates if the model is pre-quantized (e.g., GGUF, AWQ). When True, skips runtime quantization. "
            "Use for models already quantized during training or conversion."
        ),
    )
    inference_framework: Annotated[
        InferenceFramework,
        Field(
            description=(
                "Inference framework for running the VLM. Options: `transformers` (HuggingFace), `mlx` (Apple "
                "Silicon), `vllm` (high-throughput serving)."
            ),
        ),
    ]
    transformers_model_type: TransformersModelType = Field(
        default=TransformersModelType.AUTOMODEL,
        description=(
            "HuggingFace Transformers model class to use. Options: `automodel` (auto-detect), `automodel-vision2seq` "
            "(vision-to-sequence), `automodel-causallm` (causal LM), `automodel-imagetexttotext` (image+text to text)."
        ),
    )
    transformers_prompt_style: TransformersPromptStyle = Field(
        default=TransformersPromptStyle.CHAT,
        description=(
            "Prompt formatting style for Transformers models. Options: `chat` (chat template), `raw` (raw text), "
            "`none` (no formatting). Use `chat` for instruction-tuned models."
        ),
    )
    response_format: Annotated[
        ResponseFormat,
        Field(
            description=(
                "Expected output format from the VLM. Options: `doctags` (structured tags), `markdown`, `html`, "
                "`otsl` (table structure), `plaintext`. Guides model output parsing."
            ),
        ),
    ]
    torch_dtype: Optional[str] = Field(
        default=None,
        description=(
            "PyTorch data type for model weights. Options: `float32`, `float16`, `bfloat16`. Lower precision "
            "reduces memory and increases speed. If None, uses model default."
        ),
    )
    supported_devices: List[AcceleratorDevice] = Field(
        default=[
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
            AcceleratorDevice.XPU,
        ],
        description="List of hardware accelerators supported by this VLM configuration.",
    )
    stop_strings: List[str] = Field(
        default=[],
        description=(
            "List of strings that trigger generation stopping when encountered. Used to prevent the model from "
            "generating beyond desired output boundaries."
        ),
    )
    custom_stopping_criteria: List[Union[StoppingCriteria, GenerationStopper]] = Field(
        default=[],
        description=(
            "Custom stopping criteria objects for fine-grained control over generation termination. Allows "
            "implementing complex stopping logic beyond simple string matching."
        ),
    )
    extra_generation_config: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional generation configuration parameters passed to the model. Overrides or extends default "
            "generation settings (e.g., top_p, top_k, repetition_penalty)."
        ),
    )
    extra_processor_kwargs: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional keyword arguments passed to the image processor. Used for model-specific preprocessing "
            "options not covered by standard parameters."
        ),
    )
    use_kv_cache: bool = Field(
        default=True,
        description=(
            "Enable key-value caching for transformer attention. Significantly speeds up generation by caching "
            "attention computations. Disable only for debugging or memory-constrained scenarios."
        ),
    )
    max_new_tokens: int = Field(
        default=4096,
        description=(
            "Maximum number of tokens to generate. Limits output length to prevent runaway generation. "
            "Adjust based on expected output size and memory constraints."
        ),
    )
    track_generated_tokens: bool = Field(
        default=False,
        description=(
            "Track and store generated tokens during inference. Useful for debugging, analysis, or implementing "
            "custom post-processing. Increases memory usage."
        ),
    )
    track_input_prompt: bool = Field(
        default=False,
        description=(
            "Track and store the input prompt sent to the model. Useful for debugging, logging, or auditing. "
            "May contain sensitive information."
        ),
    )

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


@deprecated("Use InlineVlmOptions instead.")
class HuggingFaceVlmOptions(InlineVlmOptions):
    pass


class ApiVlmOptions(BaseVlmOptions):
    """Configuration for API-based vision-language model services."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["api_model_options"] = "api_model_options"
    url: AnyUrl = Field(
        default=AnyUrl("http://localhost:11434/v1/chat/completions"),
        description=(
            "API endpoint URL for VLM service. Must be OpenAI-compatible chat completions endpoint. "
            "Default points to local Ollama server; update for cloud services or custom deployments."
        ),
    )
    headers: Dict[str, str] = Field(
        default={},
        description=(
            "HTTP headers to include in API requests. Use for authentication or custom headers required by your "
            "API service."
        ),
        examples=[{"Authorization": "Bearer TOKEN"}],
    )
    params: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional query parameters to include in API requests. Service-specific parameters for customizing "
            "API behavior beyond standard options."
        ),
    )
    timeout: float = Field(
        default=60.0,
        description=(
            "Maximum time in seconds to wait for API response before timing out. Increase for slow networks or "
            "complex vision tasks. Recommended: 30-120 seconds."
        ),
    )
    concurrency: int = Field(
        default=1,
        description=(
            "Number of concurrent API requests allowed. Higher values improve throughput but may hit API rate limits. "
            "Adjust based on API service quotas and network capacity."
        ),
    )
    response_format: Annotated[
        ResponseFormat,
        Field(
            description=(
                "Expected output format from the VLM API. Options: `doctags` (structured tags), `markdown`, `html`, "
                "`otsl` (table structure), `plaintext`. Guides response parsing."
            ),
        ),
    ]
    stop_strings: List[str] = Field(
        default=[],
        description=(
            "List of strings that trigger generation stopping when encountered. Sent to API to prevent the model "
            "from generating beyond desired output boundaries."
        ),
    )
    custom_stopping_criteria: List[GenerationStopper] = Field(
        default=[],
        description=(
            "Custom stopping criteria objects for client-side generation control. Applied after receiving API "
            "responses for additional filtering or termination logic."
        ),
    )
    track_input_prompt: bool = Field(
        default=False,
        description=(
            "Track and store the input prompt sent to the API. Useful for debugging, logging, or auditing. "
            "May contain sensitive information."
        ),
    )
