"""Transformers implementation for object-detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForObjectDetection

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    TransformersObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.inference_engines.object_detection.hf_base import (
    HfObjectDetectionEngineBase,
)
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class TransformersObjectDetectionEngine(HfObjectDetectionEngineBase):
    """Transformers engine for object detection models.

    Uses HuggingFace Transformers and PyTorch for inference.
    Supports AutoModelForObjectDetection-compatible models.
    """

    def __init__(
        self,
        *,
        options: TransformersObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        """Initialize the Transformers engine.

        Args:
            options: Transformers-specific runtime options
            model_config: Model configuration (repo_id, revision, extra_config)
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
        """
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.options: TransformersObjectDetectionEngineOptions = options
        self._model: Optional[AutoModelForObjectDetection] = None
        self._device: Optional[torch.device] = None

    def _resolve_device(self) -> torch.device:
        """Resolve PyTorch device from accelerator options."""
        import torch

        device_str = decide_device(
            self._accelerator_options.device,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.MPS,
            ],
        )

        # Map to PyTorch device
        if device_str.startswith("cuda"):
            return torch.device(device_str)
        elif device_str == AcceleratorDevice.MPS.value:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _resolve_torch_dtype(self) -> Optional[torch.dtype]:
        """Resolve PyTorch dtype from options or model config."""
        import torch

        # Priority: options > model_config > None (auto)
        dtype_str = self.options.torch_dtype or self._model_config.torch_dtype

        if dtype_str is None:
            return None

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            _log.warning(
                f"Unknown torch_dtype '{dtype_str}', using auto dtype detection"
            )
        return dtype

    def initialize(self) -> None:
        """Initialize PyTorch model and preprocessor."""
        import torch
        from transformers import AutoModelForObjectDetection

        _log.info("Initializing Transformers object-detection engine")

        revision = self._model_config.revision or "main"
        model_folder = self._resolve_model_folder(
            repo_id=self._repo_id,
            revision=revision,
        )

        _log.debug(f"Using model at {model_folder}")

        # Resolve device and dtype
        self._device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype()

        # Set num_threads for CPU inference
        if self._device.type == "cpu":
            torch.set_num_threads(self._accelerator_options.num_threads)

        # Load preprocessor (source of truth for preprocessing)
        self._processor = self._load_preprocessor(model_folder)
        _log.debug(f"Loaded preprocessor with size: {self._processor.size}")  # type: ignore[attr-defined]

        # Load label mapping from config
        self._id_to_label = self._load_label_mapping(model_folder)
        _log.debug(f"Loaded label mapping with {len(self._id_to_label)} labels")

        # Load model
        _log.debug(f"Loading model from {model_folder} to device {self._device}")
        try:
            self._model = AutoModelForObjectDetection.from_pretrained(
                str(model_folder),
                torch_dtype=torch_dtype,
            )
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_folder}: {e}")

        self._initialized = True
        _log.info(
            f"Transformers engine ready (device={self._device}, dtype={self._model.dtype})"  # type: ignore[union-attr]
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
        import torch

        if not input_batch:
            return []
        if self._model is None or self._processor is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Preprocess images using HF processor
        images = [item.image.convert("RGB") for item in input_batch]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)

        # Get target sizes for post-processing
        target_sizes = torch.tensor(
            [[img.height, img.width] for img in images], device=self._device
        )

        # Run inference
        with torch.inference_mode():
            outputs = self._model(**inputs)  # type: ignore[operator]

        # Post-process using HuggingFace processor
        results = self._processor.post_process_object_detection(  # type: ignore[attr-defined]
            outputs,
            target_sizes=target_sizes,  # type: ignore[arg-type]
            threshold=self.options.score_threshold,
        )

        # Convert to our output format
        batch_outputs: List[ObjectDetectionEngineOutput] = []
        for input_item, result in zip(input_batch, results):
            batch_outputs.append(
                self._build_output(
                    input_item=input_item,
                    labels=result["labels"],
                    scores=result["scores"],
                    boxes=result["boxes"],
                )
            )

        return batch_outputs
