"""MLX-based object-detection engine for Apple Silicon."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from docling.datamodel.object_detection_engine_options import (
    MlxObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)

# Global lock for MLX model calls - MLX models are not thread-safe.
_MLX_GLOBAL_LOCK = threading.Lock()


class MlxObjectDetectionEngine(
    BaseObjectDetectionEngine, HuggingFaceModelDownloadMixin
):
    """MLX engine for object-detection inference on Apple Silicon.

    Uses ``mlx_vlm.models.rt_detr_v2`` to load and run RT-DETR v2 models
    converted to the MLX safetensors format.
    """

    def __init__(
        self,
        *,
        options: MlxObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: Any = None,
        artifacts_path: Optional[Union[Path, str]] = None,
    ) -> None:
        super().__init__(options=options, model_config=model_config)
        self.options: MlxObjectDetectionEngineOptions = options
        self._artifacts_path = (
            artifacts_path if artifacts_path is None else Path(artifacts_path)
        )

        self._predictor: Any = None
        self._id_to_label: Dict[int, str] = {}

    def _resolve_model_folder(self, repo_id: str, revision: str) -> Path:
        """Resolve model folder from artifacts_path or HF download."""
        repo_cache_folder = repo_id.replace("/", "--")

        if self._artifacts_path is None:
            _log.info(
                "Downloading MLX object-detection model from HuggingFace: %s@%s",
                repo_id,
                revision,
            )
            return self.download_models(repo_id=repo_id, revision=revision)

        cached = self._artifacts_path / repo_cache_folder
        if cached.exists():
            return cached

        available = (
            [p.name for p in self._artifacts_path.iterdir() if p.is_dir()]
            if self._artifacts_path.exists()
            else []
        )
        raise FileNotFoundError(
            f"Model '{repo_id}' not found in artifacts_path.\n"
            f"Expected: {cached}\n"
            f"Available: {', '.join(available) if available else 'none'}"
        )

    def initialize(self) -> None:
        """Load the MLX model, processor, and predictor."""
        if self._initialized:
            return

        _log.info("Initializing MLX object-detection engine")

        try:
            import mlx_vlm.models.rt_detr_v2  # registers the processor
            from mlx_vlm.models.rt_detr_v2.generate import RTDetrV2Predictor
            from mlx_vlm.utils import load_model
            from transformers import AutoConfig
        except ImportError:
            raise ImportError(
                "mlx-vlm is required for the MLX object-detection engine. "
                "Install it with: pip install 'mlx-vlm>=0.4.3'"
            )

        if self.model_config is None or self.model_config.repo_id is None:
            raise ValueError(
                "MlxObjectDetectionEngine requires model_config with repo_id"
            )

        repo_id = self.model_config.repo_id
        revision = self.model_config.revision or "main"

        model_folder = self._resolve_model_folder(repo_id, revision)
        _log.debug("Using MLX model at %s", model_folder)

        # Load label mapping from config
        config = AutoConfig.from_pretrained(model_folder)
        self._id_to_label = {
            int(label_id): label_name
            for label_id, label_name in config.id2label.items()
        }
        _log.debug("Loaded label mapping with %d labels", len(self._id_to_label))

        # Load model via mlx-vlm
        model = load_model(model_folder)

        # Create predictor (labels are auto-resolved from model config)
        self._predictor = RTDetrV2Predictor(
            model, threshold=self.options.score_threshold
        )

        self._initialized = True
        _log.info("MLX object-detection engine ready (model=%s)", repo_id)

    def predict_batch(
        self, input_batch: List[ObjectDetectionEngineInput]
    ) -> List[ObjectDetectionEngineOutput]:
        """Run inference on a batch of inputs.

        MLX models are not thread-safe; a global lock serialises access.
        """
        if not input_batch:
            return []
        if self._predictor is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        outputs: List[ObjectDetectionEngineOutput] = []

        with _MLX_GLOBAL_LOCK:
            for item in input_batch:
                image = item.image
                if image.mode != "RGB":
                    image = image.convert("RGB")

                result = self._predictor.predict(image)

                # Map DetectionResult -> ObjectDetectionEngineOutput
                label_ids: List[int] = []
                scores: List[float] = []
                bboxes: List[List[float]] = []

                for label_id, score, box in zip(
                    result.labels, result.scores, result.boxes
                ):
                    label_ids.append(int(label_id))
                    scores.append(float(score))
                    bboxes.append([float(v) for v in box])

                outputs.append(
                    ObjectDetectionEngineOutput(
                        label_ids=label_ids,
                        scores=scores,
                        bboxes=bboxes,
                        metadata=item.metadata.copy(),
                    )
                )

        return outputs

    def get_label_mapping(self) -> Dict[int, str]:
        """Return the label ID -> name mapping loaded from model config."""
        return self._id_to_label
