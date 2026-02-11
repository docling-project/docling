"""Transformers implementation for image-classification models."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from packaging import version

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForImageClassification

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.image_classification_engine_options import (
    TransformersImageClassificationEngineOptions,
)
from docling.models.inference_engines.image_classification.base import (
    ImageClassificationEngineInput,
    ImageClassificationEngineOutput,
)
from docling.models.inference_engines.image_classification.hf_base import (
    HfImageClassificationEngineBase,
)
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class TransformersImageClassificationEngine(HfImageClassificationEngineBase):
    """Transformers engine for image-classification models."""

    def __init__(
        self,
        *,
        options: TransformersImageClassificationEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        """Initialize the Transformers image-classification engine."""
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.options: TransformersImageClassificationEngineOptions = options
        self._model: Optional[AutoModelForImageClassification] = None
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

        if device_str.startswith("cuda"):
            return torch.device(device_str)
        if device_str == AcceleratorDevice.MPS.value:
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_torch_dtype(self) -> Optional[torch.dtype]:
        """Resolve PyTorch dtype from options or model config."""
        import torch

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
                "Unknown torch_dtype '%s', using auto dtype detection",
                dtype_str,
            )
        return dtype

    def initialize(self) -> None:
        """Initialize PyTorch model and preprocessor."""
        import torch
        from transformers import AutoModelForImageClassification

        _log.info("Initializing Transformers image-classification engine")

        revision = self._model_config.revision or "main"
        model_folder = self._resolve_model_folder(
            repo_id=self._repo_id,
            revision=revision,
        )

        _log.debug("Using model at %s", model_folder)

        self._device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype()

        if self._device.type == "cpu":
            torch.set_num_threads(self._accelerator_options.num_threads)

        self._processor = self._load_preprocessor(model_folder)
        self._id_to_label = self._load_label_mapping(model_folder)

        _log.debug("Loading model from %s to device %s", model_folder, self._device)
        try:
            self._model = AutoModelForImageClassification.from_pretrained(
                str(model_folder),
                torch_dtype=torch_dtype,
            )
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]

            # Optionally compile model for better performance (model must be in eval mode first)
            # Works for Python < 3.14 with any torch 2.x
            # Works for Python >= 3.14 with torch >= 2.10
            if self.options.compile_model:
                if sys.version_info < (3, 14):
                    self._model = torch.compile(self._model)  # type: ignore[arg-type,assignment]
                    _log.debug("Model compiled with torch.compile()")
                elif version.parse(torch.__version__) >= version.parse("2.10"):
                    self._model = torch.compile(self._model)  # type: ignore[arg-type,assignment]
                    _log.debug("Model compiled with torch.compile()")
                else:
                    _log.warning(
                        "Model compilation requested but not available "
                        "(requires Python < 3.14 or torch >= 2.10 for Python 3.14+)"
                    )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_folder}: {exc}")

        self._initialized = True
        _log.info(
            "Transformers image-classification engine ready (device=%s, dtype=%s)",
            self._device,
            self._model.dtype,  # type: ignore[union-attr]
        )

    def predict_batch(
        self, input_batch: List[ImageClassificationEngineInput]
    ) -> List[ImageClassificationEngineOutput]:
        """Run inference on a batch of inputs."""
        import torch

        if not input_batch:
            return []
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        images = [item.image.convert("RGB") for item in input_batch]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)  # type: ignore[operator]
            probs_batch = torch.softmax(outputs.logits, dim=-1)

        batch_outputs: List[ImageClassificationEngineOutput] = []
        for input_item, probs_vector in zip(input_batch, probs_batch):
            # Use topk for efficiency when top_k is specified
            if self.options.top_k is not None:
                k = min(self.options.top_k, len(probs_vector))
                scores, labels = torch.topk(probs_vector, k=k)
            else:
                scores, labels = torch.sort(probs_vector, descending=True)

            batch_outputs.append(
                self._build_output(
                    input_item=input_item,
                    labels=labels,
                    scores=scores,
                )
            )

        return batch_outputs
