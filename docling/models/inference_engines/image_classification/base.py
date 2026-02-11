"""Base classes for image-classification inference engines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class ImageClassificationEngineType(str, Enum):
    """Supported inference engine types for image-classification models."""

    ONNXRUNTIME = "onnxruntime"
    TRANSFORMERS = "transformers"


class BaseImageClassificationEngineOptions(BaseModel):
    """Base configuration shared across image-classification engines."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine_type: ImageClassificationEngineType = Field(
        description="Type of inference engine to use",
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of classes to return. If None, all classes are returned.",
    )


class ImageClassificationEngineInput(BaseModel):
    """Generic input accepted by every image-classification engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image = Field(description="PIL image to run inference on")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata that is echoed back in the output",
    )


class ImageClassificationEngineOutput(BaseModel):
    """Output returned by image-classification engines."""

    label_ids: List[int] = Field(
        default_factory=list,
        description="Predicted class indices sorted by confidence descending",
    )
    scores: List[float] = Field(
        default_factory=list,
        description="Confidence scores sorted descending",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata echoed back from the input or engine",
    )


class BaseImageClassificationEngine(ABC):
    """Abstract base-class for image-classification engines."""

    def __init__(
        self,
        options: BaseImageClassificationEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            options: Engine-specific configuration options
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        self.options = options
        self.model_config = model_config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize engine resources (load models, allocate buffers, etc.)."""

    @abstractmethod
    def predict_batch(
        self, input_batch: List[ImageClassificationEngineInput]
    ) -> List[ImageClassificationEngineOutput]:
        """Run inference on a batch of inputs."""

    @abstractmethod
    def get_label_mapping(self) -> Dict[int, str]:
        """Get the label mapping for this model.

        Returns:
            Dictionary mapping label IDs to label names
        """

    def predict(
        self, input_data: ImageClassificationEngineInput
    ) -> ImageClassificationEngineOutput:
        """Helper to run inference on a single input."""
        if not self._initialized:
            _log.debug("Initializing %s for single prediction", type(self).__name__)
            self.initialize()

        results = self.predict_batch([input_data])
        return results[0]

    def __call__(
        self,
        input_data: ImageClassificationEngineInput
        | List[ImageClassificationEngineInput],
    ) -> ImageClassificationEngineOutput | List[ImageClassificationEngineOutput]:
        if not self._initialized:
            _log.debug("Initializing %s for call", type(self).__name__)
            self.initialize()

        if isinstance(input_data, list):
            return self.predict_batch(input_data)
        return self.predict(input_data)
