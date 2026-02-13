"""Shared HuggingFace-based helpers for image-classification engines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.common import HfVisionModelMixin
from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngine,
    BaseImageClassificationEngineOptions,
    ImageClassificationEngineInput,
    ImageClassificationEngineOutput,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig


class HfImageClassificationEngineBase(
    HfVisionModelMixin, BaseImageClassificationEngine
):
    """Base class for image-classification engines that load HF artifacts/configs."""

    def __init__(
        self,
        *,
        options: BaseImageClassificationEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ) -> None:
        super().__init__(options=options, model_config=model_config)
        self.options: BaseImageClassificationEngineOptions = options
        self._init_hf_vision_model(
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
            model_family_name="image-classification",
        )

    def _build_output(
        self,
        *,
        input_item: ImageClassificationEngineInput,
        labels: Iterable[Any],
        scores: Iterable[Any],
    ) -> ImageClassificationEngineOutput:
        """Build standard engine output from class-score iterables.

        Note: Assumes labels and scores are already sorted by descending score.
        """
        predictions: list[tuple[int, float]] = []
        for label, score in zip(labels, scores):
            predictions.append((self._as_int(label), self._as_float(score)))

        if self.options.top_k is not None:
            predictions = predictions[: self.options.top_k]

        return ImageClassificationEngineOutput(
            label_ids=[label for label, _ in predictions],
            scores=[score for _, score in predictions],
            metadata=input_item.metadata.copy(),
        )
