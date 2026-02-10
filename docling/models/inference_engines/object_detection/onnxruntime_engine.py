"""ONNX Runtime implementation for RT-DETR style object-detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, RTDetrImageProcessor

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.inference_engines.vlm._utils import resolve_model_artifacts_path
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class OnnxRuntimeObjectDetectionEngine(
    BaseObjectDetectionEngine, HuggingFaceModelDownloadMixin
):
    """ONNX Runtime engine for RT-DETR detectors.

    Uses HuggingFace RTDetrImageProcessor for preprocessing to ensure
    consistency with transformers-based models. This is the source of truth
    for preprocessing parameters.
    """

    def __init__(
        self,
        *,
        options: OnnxRuntimeObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        """Initialize the ONNX Runtime engine.

        Args:
            options: ONNX Runtime-specific runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        if model_config is None or model_config.repo_id is None:
            raise ValueError(
                "OnnxRuntimeObjectDetectionEngine requires model_config with repo_id"
            )

        repo_id = model_config.repo_id

        super().__init__(options, model_config=model_config)
        self.options: OnnxRuntimeObjectDetectionEngineOptions = options
        self._model_config: EngineModelConfig = model_config
        self._repo_id: str = repo_id
        self._accelerator_options = accelerator_options
        self._artifacts_path = (
            artifacts_path if artifacts_path is None else Path(artifacts_path)
        )
        self._session: Optional[ort.InferenceSession] = None
        self._processor: Optional[RTDetrImageProcessor] = None
        self._model_path: Optional[Path] = None
        self._id_to_label: Dict[int, str] = {}

    def _resolve_model_artifacts(self) -> tuple[Path, Path]:
        """Resolve model artifacts from artifacts_path or HF download.

        Returns:
            Tuple of (model_folder, model_path)
        """
        repo_id = self._repo_id
        revision = self._model_config.revision or "main"

        model_filename = self._resolve_model_filename()
        model_folder = self._resolve_model_folder(
            repo_id=repo_id,
            revision=str(revision),
        )
        model_path = model_folder / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model file '{model_filename}' not found: {model_path}"
            )

        return model_folder, model_path

    def _resolve_model_folder(self, repo_id: str, revision: str) -> Path:
        """Resolve model folder from artifacts_path or HF download."""

        def download_wrapper(download_repo_id: str, download_revision: str) -> Path:
            _log.info(
                "Downloading ONNX model from HuggingFace: %s@%s",
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

    def _resolve_model_filename(self) -> str:
        """Determine which ONNX filename to load."""
        filename = self.options.model_filename
        extra_filename = self._model_config.extra_config.get("model_filename")
        if extra_filename and isinstance(extra_filename, str):
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
        preprocessor_config = model_folder / "preprocessor_config.json"
        if not preprocessor_config.exists():
            raise FileNotFoundError(
                f"RTDetrImageProcessor config not found: {preprocessor_config}"
            )

        try:
            _log.debug(f"Loading preprocessor from {model_folder}")
            return RTDetrImageProcessor.from_pretrained(str(model_folder))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RTDetrImageProcessor from local model folder {model_folder}: {e}"
            )

    def _load_label_mapping(self, model_folder: Path) -> Dict[int, str]:
        """Load label mapping from HuggingFace model config.

        Args:
            model_folder: Path to model folder containing config.json

        Returns:
            Dictionary mapping label IDs to label names

        Raises:
            RuntimeError: If config cannot be loaded
        """
        try:
            config = AutoConfig.from_pretrained(str(model_folder))
            return {
                int(label_id): label_name
                for label_id, label_name in config.id2label.items()
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to load label mapping from model config at {model_folder}: {e}"
            )

    def get_label_mapping(self) -> Dict[int, str]:
        """Get the label mapping for this model.

        Returns:
            Dictionary mapping label IDs to label names
        """
        return self._id_to_label

    def initialize(self) -> None:
        """Initialize ONNX session and preprocessor."""
        _log.info("Initializing ONNX Runtime object-detection engine")

        # Resolve model folder and model path in one step
        model_folder, self._model_path = self._resolve_model_artifacts()

        _log.debug(f"Using ONNX model at {self._model_path}")

        # Load preprocessor (source of truth for preprocessing)
        self._processor = self._load_preprocessor(model_folder)
        _log.debug(f"Loaded preprocessor with size: {self._processor.size}")

        # Load label mapping from config
        self._id_to_label = self._load_label_mapping(model_folder)
        _log.debug(f"Loaded label mapping with {len(self._id_to_label)} labels")

        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._accelerator_options.num_threads
        providers = self._resolve_providers()

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self._initialized = True
        _log.info(
            f"ONNX Runtime engine ready (providers={self._session.get_providers()})"
        )

    def _resolve_providers(self) -> List[str]:
        """Resolve ONNX Runtime providers from accelerator and engine options."""
        configured_providers = self.options.providers or ["CPUExecutionProvider"]
        if configured_providers != ["CPUExecutionProvider"]:
            return configured_providers

        device = decide_device(
            self._accelerator_options.device,
            supported_devices=[AcceleratorDevice.CPU, AcceleratorDevice.CUDA],
        )

        if device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if device != AcceleratorDevice.CPU.value:
            _log.warning(
                "Unsupported ONNX device '%s' for object detection. Falling back to CPU provider.",
                device,
            )
        return ["CPUExecutionProvider"]

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
            [[img.width, img.height] for img in images], dtype=np.int64
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

        batch_outputs: List[ObjectDetectionEngineOutput] = []
        for idx, input_item in enumerate(input_batch):
            # Filter detections by score threshold
            filtered_labels = []
            filtered_boxes = []
            filtered_scores = []

            for label, box, score in zip(
                labels_batch[idx], boxes_batch[idx], scores_batch[idx]
            ):
                score_float = float(score)
                if score_float >= self.options.score_threshold:
                    filtered_labels.append(int(label))
                    filtered_boxes.append([float(v) for v in box])
                    filtered_scores.append(score_float)

            batch_outputs.append(
                ObjectDetectionEngineOutput(
                    label_ids=filtered_labels,
                    scores=filtered_scores,
                    bboxes=filtered_boxes,
                    metadata=input_item.metadata.copy(),
                )
            )

        return batch_outputs
