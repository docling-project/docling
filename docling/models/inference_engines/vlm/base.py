"""Base classes for VLM inference engines."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class VlmEngineType(str, Enum):
    """Types of VLM inference engines available."""

    # Local/inline engines
    TRANSFORMERS = "transformers"
    MLX = "mlx"
    VLLM = "vllm"

    # API-based engines
    API = "api"
    API_OLLAMA = "api_ollama"
    API_LMSTUDIO = "api_lmstudio"
    API_OPENAI = "api_openai"

    # Auto-selection
    AUTO_INLINE = "auto_inline"

    @classmethod
    def is_api_variant(cls, engine_type: "VlmEngineType") -> bool:
        """Check if an engine type is an API variant."""
        return engine_type in {
            cls.API,
            cls.API_OLLAMA,
            cls.API_LMSTUDIO,
            cls.API_OPENAI,
        }

    @classmethod
    def is_inline_variant(cls, engine_type: "VlmEngineType") -> bool:
        """Check if an engine type is an inline/local variant."""
        return engine_type in {
            cls.TRANSFORMERS,
            cls.MLX,
            cls.VLLM,
        }


class BaseVlmEngineOptions(BaseModel):
    """Base configuration for VLM inference engines.

    Engine options are independent of model specifications and prompts.
    They only control how the inference is executed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine_type: VlmEngineType = Field(description="Type of inference engine to use")


class VlmEngineInput(BaseModel):
    """Input to a VLM inference engine.

    This is the generic interface that all engines accept.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image = Field(description="PIL Image to process")
    prompt: str = Field(description="Text prompt for the model")
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


class VlmEngineOutput(BaseModel):
    """Output from a VLM inference engine.

    This is the generic interface that all engines return.
    """

    text: str = Field(description="Generated text from the model")
    stop_reason: Optional[str] = Field(
        default=None, description="Reason why generation stopped"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata from the engine"
    )


class BaseVlmEngine(ABC):
    """Abstract base class for VLM inference engines.

    An engine handles the low-level model inference with generic inputs
    (PIL images + text prompts) and returns text predictions.

    Engines are independent of:
    - Pipeline stages (DoclingDocument, Page objects)
    - Response formats (doctags, markdown, etc.)

    But they ARE aware of:
    - Model specifications (repo_id, revision, model_type via EngineModelConfig)

    These model specs are provided at construction time for eager initialization.
    """

    def __init__(
        self,
        options: BaseVlmEngineOptions,
        model_config: Optional["EngineModelConfig"] = None,
    ):
        """Initialize the engine.

        Args:
            options: Engine-specific configuration options
            model_config: Model configuration (repo_id, revision, extra_config)
                         If None, model must be specified in predict() calls
        """
        self.options = options
        self.model_config = model_config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the engine (load models, setup connections, etc.).

        This is called once before the first inference.
        Implementations should set self._initialized = True when done.
        """

    @abstractmethod
    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs.

        This is the primary method that all engines must implement.
        Single predictions are routed through this method.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """

    def predict(self, input_data: VlmEngineInput) -> VlmEngineOutput:
        """Run inference on a single input.

        This is a convenience method that wraps the input in a list and calls
        predict_batch(). Engines should NOT override this method - all
        inference logic should be in predict_batch().

        Args:
            input_data: Generic input containing image, prompt, and config

        Returns:
            Generic output containing generated text and metadata
        """
        if not self._initialized:
            self.initialize()

        results = self.predict_batch([input_data])
        return results[0]

    def __call__(
        self, input_data: VlmEngineInput | List[VlmEngineInput]
    ) -> VlmEngineOutput | List[VlmEngineOutput]:
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

        Called when the engine is no longer needed.
        Implementations can override to release resources.
        """
