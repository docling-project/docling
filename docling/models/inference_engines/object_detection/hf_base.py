"""Shared HuggingFace-based helpers for object-detection engines."""

from __future__ import annotations

import logging
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.inference_engines.vlm._utils import resolve_model_artifacts_path
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class HfObjectDetectionEngineBase(
    BaseObjectDetectionEngine, HuggingFaceModelDownloadMixin
):
    """Base class for object-detection engines that load HF artifacts and configs."""

    def __init__(
        self,
        *,
        options: BaseObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ) -> None:
        if model_config is None or model_config.repo_id is None:
            raise ValueError(
                f"{type(self).__name__} requires model_config with repo_id"
            )

        super().__init__(options=options, model_config=model_config)
        self.options: BaseObjectDetectionEngineOptions = options
        self._model_config: EngineModelConfig = model_config
        self._repo_id: str = model_config.repo_id
        self._accelerator_options = accelerator_options
        self._artifacts_path = (
            artifacts_path if artifacts_path is None else Path(artifacts_path)
        )
        self._processor: Optional[BaseImageProcessor] = None
        self._id_to_label: Dict[int, str] = {}

    def _resolve_model_folder(self, repo_id: str, revision: str) -> Path:
        """Resolve model folder from artifacts_path or HF download."""

        def download_wrapper(download_repo_id: str, download_revision: str) -> Path:
            _log.info(
                "Downloading object-detection model from HuggingFace: %s@%s",
                download_repo_id,
                download_revision,
            )
            return self.download_models(
                repo_id=download_repo_id,
                revision=download_revision,
                local_dir=None,
                force=False,
                progress=False,
            )

        return resolve_model_artifacts_path(
            repo_id=repo_id,
            revision=revision,
            artifacts_path=self._artifacts_path,
            download_fn=download_wrapper,
        )

    def _load_preprocessor(self, model_folder: Path) -> BaseImageProcessor:
        """Load HuggingFace image processor from model folder."""
        preprocessor_config = model_folder / "preprocessor_config.json"
        if not preprocessor_config.exists():
            raise FileNotFoundError(
                f"Image processor config not found: {preprocessor_config}"
            )

        try:
            from transformers import AutoImageProcessor

            _log.debug("Loading image processor from %s", model_folder)
            return AutoImageProcessor.from_pretrained(str(model_folder))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image processor from {model_folder}: {exc}"
            )

    def _load_label_mapping(self, model_folder: Path) -> Dict[int, str]:
        """Load label mapping from HuggingFace model config."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(str(model_folder))
            return {
                int(label_id): label_name
                for label_id, label_name in config.id2label.items()
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load label mapping from model config at {model_folder}: {exc}"
            )

    def get_label_mapping(self) -> Dict[int, str]:
        """Get the label mapping for this model."""
        return self._id_to_label

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

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, Real):
            return float(value)

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError(
                    f"Expected scalar-like ndarray with size 1, got shape={value.shape}"
                )
            return float(value.reshape(-1)[0])

        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"Expected scalar-like tensor with one element, got shape={tuple(value.shape)}"
                )
            return float(value.item())

        raise TypeError(f"Unsupported score/box value type: {type(value)!r}")

    @staticmethod
    def _as_int(value: Any) -> int:
        if isinstance(value, Integral):
            return int(value)

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError(
                    f"Expected scalar-like ndarray with size 1, got shape={value.shape}"
                )
            return int(value.reshape(-1)[0])

        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"Expected scalar-like tensor with one element, got shape={tuple(value.shape)}"
                )
            return int(value.item())

        raise TypeError(f"Unsupported label value type: {type(value)!r}")
