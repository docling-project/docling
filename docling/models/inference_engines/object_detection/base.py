"""Base classes for object-detection inference engines."""

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


class ObjectDetectionEngineType(str, Enum):
    """Supported inference engine types for object-detection models."""

    ONNXRUNTIME = "onnxruntime"
    TRANSFORMERS = "transformers"


class BaseObjectDetectionEngineOptions(BaseModel):
    """Base configuration shared across object-detection engines."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine_type: ObjectDetectionEngineType = Field(
        description="Type of inference engine to use",
    )


class ObjectDetectionEngineInput(BaseModel):
    """Generic input accepted by every object-detection engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image = Field(description="PIL image to run inference on")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata that is echoed back in the output",
    )


class ObjectDetectionEnginePrediction(BaseModel):
    """Single detection prediction."""

    label_id: int = Field(description="Predicted class index")
    score: float = Field(description="Confidence score for the prediction")
    bbox: List[float] = Field(
        description="Bounding box as [x_min, y_min, x_max, y_max] in input image pixels",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata provided by the engine",
    )


class ObjectDetectionEngineOutput(BaseModel):
    """Output returned by object-detection engines."""

    predictions: List[ObjectDetectionEnginePrediction] = Field(
        default_factory=list,
        description="List of predictions for the corresponding input image",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata echoed back from the input or engine",
    )


class BaseObjectDetectionEngine(ABC):
    """Abstract base-class for object-detection engines."""

    def __init__(
        self,
        options: BaseObjectDetectionEngineOptions,
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
        self, input_batch: List[ObjectDetectionEngineInput]
    ) -> List[ObjectDetectionEngineOutput]:
        """Run inference on a batch of inputs."""

    def predict(
        self, input_data: ObjectDetectionEngineInput
    ) -> ObjectDetectionEngineOutput:
        """Helper to run inference on a single input."""
        if not self._initialized:
            _log.debug("Initializing %s for single prediction", type(self).__name__)
            self.initialize()

        results = self.predict_batch([input_data])
        return results[0]

    def __call__(
        self,
        input_data: ObjectDetectionEngineInput | List[ObjectDetectionEngineInput],
    ) -> ObjectDetectionEngineOutput | List[ObjectDetectionEngineOutput]:
        if not self._initialized:
            _log.debug("Initializing %s for call", type(self).__name__)
            self.initialize()

        if isinstance(input_data, list):
            return self.predict_batch(input_data)
        return self.predict(input_data)

    def cleanup(self) -> None:
        """Release resources when the engine is no longer needed."""
        return None
