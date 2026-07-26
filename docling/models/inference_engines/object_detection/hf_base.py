"""Shared HuggingFace-based helpers for object-detection engines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.common import HfVisionModelMixin
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig


class HfObjectDetectionEngineBase(HfVisionModelMixin, BaseObjectDetectionEngine):
    """Base class for object-detection engines that load HF artifacts and configs."""

    def __init__(
        self,
        *,
        options: BaseObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ) -> None:
        super().__init__(options=options, model_config=model_config)
        self.options: BaseObjectDetectionEngineOptions = options
        self._init_hf_vision_model(
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
            model_family_name="object-detection",
        )

    def _build_output(
        self,
        *,
        input_item: ObjectDetectionEngineInput,
        labels: Iterable[Any],
        scores: Iterable[Any],
        boxes: Iterable[Sequence[Any]],
        apply_score_threshold: bool = False,
    ) -> ObjectDetectionEngineOutput:
        """Build standard engine output from raw detection iterables."""
        label_ids: list[int] = []
        output_scores: list[float] = []
        bboxes: list[list[float]] = []

        for label, score, box in zip(labels, scores, boxes):
            score_float = self._as_float(score)
            if apply_score_threshold and score_float < self.options.score_threshold:
                continue

            label_ids.append(self._as_int(label))
            output_scores.append(score_float)
            bboxes.append([self._as_float(value) for value in box])

        return ObjectDetectionEngineOutput(
            label_ids=label_ids,
            scores=output_scores,
            bboxes=bboxes,
            metadata=input_item.metadata.copy(),
        )
