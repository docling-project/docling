"""Base classes for VLM runtimes."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

_log = logging.getLogger(__name__)


class VlmRuntimeType(str, Enum):
    """Types of VLM runtimes available."""

    # Local/inline runtimes
    TRANSFORMERS = "transformers"
    MLX = "mlx"
    VLLM = "vllm"

    # API-based runtimes
    API = "api"
    API_OLLAMA = "api_ollama"
    API_LMSTUDIO = "api_lmstudio"
    API_OPENAI = "api_openai"

    # Auto-selection
    AUTO_INLINE = "auto_inline"

    @classmethod
    def is_api_variant(cls, runtime_type: "VlmRuntimeType") -> bool:
        """Check if a runtime type is an API variant."""
        return runtime_type in {
            cls.API,
            cls.API_OLLAMA,
            cls.API_LMSTUDIO,
            cls.API_OPENAI,
        }

    @classmethod
    def is_inline_variant(cls, runtime_type: "VlmRuntimeType") -> bool:
        """Check if a runtime type is an inline/local variant."""
        return runtime_type in {
            cls.TRANSFORMERS,
            cls.MLX,
            cls.VLLM,
        }


class BaseVlmRuntimeOptions(BaseModel):
    """Base configuration for VLM runtimes.

    Runtime options are independent of model specifications and prompts.
    They only control how the inference is executed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime_type: VlmRuntimeType = Field(
        description="Type of runtime to use for inference"
    )


class VlmRuntimeInput(BaseModel):
    """Input to a VLM runtime.

    This is the generic interface that all runtimes accept.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image = Field(description="PIL Image to process")
    prompt: str = Field(description="Text prompt for the model")
    repo_id: str = Field(description="Model repository ID (e.g., HuggingFace repo)")
    temperature: float = Field(
        default=0.0, description="Sampling temperature for generation"
    )
    max_new_tokens: int = Field(
        default=4096, description="Maximum number of tokens to generate"
    )
    stop_strings: List[str] = Field(
        default_factory=list, description="Strings that trigger generation stopping"
    )
    extra_generation_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional generation configuration"
    )


class VlmRuntimeOutput(BaseModel):
    """Output from a VLM runtime.

    This is the generic interface that all runtimes return.
    """

    text: str = Field(description="Generated text from the model")
    stop_reason: Optional[str] = Field(
        default=None, description="Reason why generation stopped"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata from the runtime"
    )


class BaseVlmRuntime(ABC):
    """Abstract base class for VLM runtimes.

    A runtime handles the low-level model inference with generic inputs
    (PIL images + text prompts) and returns text predictions.

    Runtimes are independent of:
    - Model specifications (repo_id, prompts)
    - Pipeline stages (DoclingDocument, Page objects)
    - Response formats (doctags, markdown, etc.)

    These concerns are handled by the stages that use the runtime.
    """

    def __init__(self, options: BaseVlmRuntimeOptions):
        """Initialize the runtime.

        Args:
            options: Runtime-specific configuration options
        """
        self.options = options
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the runtime (load models, setup connections, etc.).

        This is called once before the first inference.
        Implementations should set self._initialized = True when done.
        """

    @abstractmethod
    def predict(self, input_data: VlmRuntimeInput) -> VlmRuntimeOutput:
        """Run inference on a single input.

        Args:
            input_data: Generic input containing image, prompt, and config

        Returns:
            Generic output containing generated text and metadata
        """

    def predict_batch(
        self, input_batch: List[VlmRuntimeInput]
    ) -> List[VlmRuntimeOutput]:
        """Run inference on a batch of inputs.

        Default implementation processes inputs sequentially. Subclasses should
        override this method to implement efficient batched inference.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        # Default: process sequentially
        return [self.predict(input_data) for input_data in input_batch]

    def __call__(
        self, input_data: VlmRuntimeInput | List[VlmRuntimeInput]
    ) -> VlmRuntimeOutput | List[VlmRuntimeOutput]:
        """Convenience method to run inference.

        Args:
            input_data: Single input or list of inputs

        Returns:
            Single output or list of outputs
        """
        if not self._initialized:
            self.initialize()

        if isinstance(input_data, list):
            return self.predict_batch(input_data)
        else:
            return self.predict(input_data)

    def cleanup(self) -> None:
        """Clean up resources (optional).

        Called when the runtime is no longer needed.
        Implementations can override to release resources.
        """
