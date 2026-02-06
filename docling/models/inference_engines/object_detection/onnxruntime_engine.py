"""ONNX Runtime implementation for RT-DETR style object-detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import onnxruntime as ort
from transformers import RTDetrImageProcessor

from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
    ObjectDetectionEnginePrediction,
)
from docling.models.utils.hf_model_download import download_hf_model

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class OnnxRuntimeObjectDetectionEngine(BaseObjectDetectionEngine):
    """ONNX Runtime engine for RT-DETR detectors.

    Uses HuggingFace RTDetrImageProcessor for preprocessing to ensure
    consistency with transformers-based models. This is the source of truth
    for preprocessing parameters.
    """

    def __init__(
        self,
        options: OnnxRuntimeObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
    ):
        """Initialize the ONNX Runtime engine.

        Args:
            options: ONNX Runtime-specific runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: OnnxRuntimeObjectDetectionEngineOptions = options
        self._session: Optional[ort.InferenceSession] = None
        self._processor: Optional[RTDetrImageProcessor] = None
        self._model_path: Optional[Path] = None

    def _resolve_model_path(self) -> Path:
        """Resolve model path from artifacts_path or HF download.

        Returns:
            Path to ONNX model file
        """
        if self.model_config is None:
            raise ValueError("model_config is required for ONNX engine")

        repo_id = self.model_config.repo_id
        revision = self.model_config.revision or "main"

        if repo_id is None:
            raise ValueError("model_config must provide repo_id")

        model_filename = self._resolve_model_filename()

        artifacts_root = self.options.artifacts_path
        if artifacts_root is not None:
            model_folder = artifacts_root / repo_id.replace("/", "--")
            candidate = model_folder / model_filename
            if candidate.exists():
                _log.info(f"Using ONNX model from artifacts_path: {candidate}")
                return candidate
            _log.warning(
                "Model not found in artifacts_path (%s), will download from HuggingFace",
                candidate,
            )

        # Download from HuggingFace
        _log.info(f"Downloading model from HuggingFace: {repo_id}@{revision}")
        base_path = download_hf_model(
            repo_id=repo_id,
            revision=str(revision),
            local_dir=None,
            force=False,
            progress=False,
        )

        candidate = base_path / model_filename
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected ONNX file '{model_filename}' not found in repo '{repo_id}'. "
                f"Searched in: {base_path}"
            )

        return candidate

    def _resolve_model_filename(self) -> str:
        """Determine which ONNX filename to load."""
        filename = self.options.model_filename
        if self.model_config is not None:
            extra_filename = self.model_config.extra_config.get("model_filename")
            if isinstance(extra_filename, str) and extra_filename:
                filename = extra_filename
        return filename

    def _load_preprocessor(self, model_folder: Path) -> RTDetrImageProcessor:
        """Load HuggingFace preprocessor from model folder.

        This is the source of truth for preprocessing parameters.

        Args:
            model_folder: Path to model folder

        Returns:
            RTDetrImageProcessor instance
        """
        # Try loading from local folder first
        if (model_folder / "preprocessor_config.json").exists():
            try:
                _log.debug(f"Loading preprocessor from {model_folder}")
                return RTDetrImageProcessor.from_pretrained(str(model_folder))
            except Exception as e:
                _log.debug(f"Could not load preprocessor from local: {e}")

        # Fall back to repo_id if available
        if self.model_config is not None and self.model_config.repo_id:
            repo_id = self.model_config.repo_id
            try:
                _log.debug(f"Loading preprocessor from HuggingFace: {repo_id}")
                return RTDetrImageProcessor.from_pretrained(repo_id)
            except Exception as e:
                _log.warning(
                    "Could not load preprocessor from %s: %s. Using default configuration.",
                    repo_id,
                    e,
                )

        # Last resort: use default preprocessor
        _log.warning("Using default RTDetrImageProcessor configuration")
        return RTDetrImageProcessor()

    def initialize(self) -> None:
        """Initialize ONNX session and preprocessor."""
        _log.info("Initializing ONNX Runtime object-detection engine")

        # Resolve model path
        self._model_path = self._resolve_model_path()
        model_folder = self._model_path.parent

        _log.debug(f"Using ONNX model at {self._model_path}")

        # Load preprocessor (source of truth for preprocessing)
        self._processor = self._load_preprocessor(model_folder)
        _log.debug(f"Loaded preprocessor with size: {self._processor.size}")

        # Create ONNX session
        sess_options = ort.SessionOptions()
        providers = self.options.providers or ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self._initialized = True
        _log.info(
            f"ONNX Runtime engine ready (providers={self._session.get_providers()})"
        )

    def predict_batch(
        self, input_batch: List[ObjectDetectionEngineInput]
    ) -> List[ObjectDetectionEngineOutput]:
        """Run inference on a batch of inputs.

        Args:
            input_batch: List of input images with metadata

        Returns:
            List of detection outputs
        """
        if not input_batch:
            return []
        if self._session is None or self._processor is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Preprocess images using HF processor (source of truth)
        images = [item.image.convert("RGB") for item in input_batch]
        inputs = self._processor(images=images, return_tensors="np")

        # Get original sizes for post-processing
        orig_sizes = np.array(
            [[img.width, img.height] for img in images], dtype=np.float32
        )

        # Run ONNX inference
        output_tensors = self._session.run(
            None,
            {
                self.options.image_input_name: inputs["pixel_values"],
                self.options.sizes_input_name: orig_sizes,
            },
        )

        if len(output_tensors) < 3:
            raise RuntimeError(
                "Expected ONNX model to return at least 3 outputs: "
                "[labels, boxes, scores]"
            )

        labels_batch, boxes_batch, scores_batch = output_tensors[:3]

        # Convert to structured outputs
        batch_outputs: List[ObjectDetectionEngineOutput] = []
        for idx, input_item in enumerate(input_batch):
            predictions: List[ObjectDetectionEnginePrediction] = []
            for label, box, score in zip(
                labels_batch[idx], boxes_batch[idx], scores_batch[idx]
            ):
                predictions.append(
                    ObjectDetectionEnginePrediction(
                        label_id=int(label),
                        score=float(score),
                        bbox=[float(v) for v in box],
                    )
                )

            batch_outputs.append(
                ObjectDetectionEngineOutput(
                    predictions=predictions, metadata=input_item.metadata.copy()
                )
            )

        return batch_outputs
