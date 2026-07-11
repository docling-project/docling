"""Shared HuggingFace helpers for vision inference engine families."""

from __future__ import annotations

import json
import logging
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.vlm._utils import resolve_model_artifacts_path
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class NumpyImageProcessor:
    """Dependency-light, torch-free stand-in for a transformers image processor.

    Reproduces the resize/rescale/normalize preprocessing described by a
    ``preprocessor_config.json`` using only numpy + Pillow and returns
    ``{"pixel_values": np.ndarray}`` shaped ``(N, C, H, W)``. It exists so the ONNX
    Runtime object-detection engine can preprocess inputs in environments without
    torch/torchvision: transformers >= 5 gates its image-processor classes behind
    those backends, so ``AutoImageProcessor.from_pretrained`` is unavailable there.

    For the default docling layout presets (RT-DETR: resize to a fixed square, no
    rescale/normalize) the output is byte-identical to the transformers slow image
    processor.
    """

    # PIL resampling filter for the integer codes stored by transformers/PIL.
    _RESAMPLE = {
        0: Image.Resampling.NEAREST,
        1: Image.Resampling.LANCZOS,
        2: Image.Resampling.BILINEAR,
        3: Image.Resampling.BICUBIC,
        4: Image.Resampling.BOX,
        5: Image.Resampling.HAMMING,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.size: Optional[Dict[str, int]] = config.get("size")
        self.do_resize: bool = bool(config.get("do_resize", False))
        self.do_rescale: bool = bool(config.get("do_rescale", False))
        self.do_normalize: bool = bool(config.get("do_normalize", False))
        self.rescale_factor: float = float(config.get("rescale_factor", 1.0 / 255.0))
        self.image_mean: Optional[List[float]] = config.get("image_mean")
        self.image_std: Optional[List[float]] = config.get("image_std")
        self.resample = self._RESAMPLE.get(
            int(config.get("resample", 2)), Image.Resampling.BILINEAR
        )

    @classmethod
    def from_config_file(cls, path: Path) -> NumpyImageProcessor:
        with path.open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def _target_hw(self) -> Optional[Tuple[int, int]]:
        size = self.size or {}
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        edge = size.get("shortest_edge") or size.get("longest_edge")
        if edge is not None:
            return int(edge), int(edge)
        return None

    def __call__(
        self, images: Any, return_tensors: str = "np", **_: Any
    ) -> Dict[str, np.ndarray]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        target = self._target_hw()
        frames: List[np.ndarray] = []
        for image in images:
            pil = image.convert("RGB")
            if self.do_resize and target is not None:
                pil = pil.resize((target[1], target[0]), resample=self.resample)
            frames.append(np.asarray(pil))
        batch = np.stack(frames, axis=0)  # (N, H, W, C), uint8
        if self.do_rescale or self.do_normalize:
            batch = batch.astype(np.float32)
        if self.do_rescale:
            batch = batch * self.rescale_factor
        if (
            self.do_normalize
            and self.image_mean is not None
            and self.image_std is not None
        ):
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            batch = (batch - mean) / std
        pixel_values = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        return {"pixel_values": pixel_values}


class HfVisionModelMixin(HuggingFaceModelDownloadMixin):
    """Shared utility mixin for HF vision model loading and label conversion."""

    def _init_hf_vision_model(
        self,
        *,
        model_config: Optional[EngineModelConfig],
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]],
        model_family_name: str,
    ) -> None:
        if model_config is None or model_config.repo_id is None:
            raise ValueError(
                f"{type(self).__name__} requires model_config with repo_id"
            )

        self._model_config: EngineModelConfig = model_config
        self._repo_id: str = model_config.repo_id
        self._accelerator_options = accelerator_options
        self._artifacts_path = (
            artifacts_path if artifacts_path is None else Path(artifacts_path)
        )
        self._model_family_name = model_family_name
        self._processor: Optional[BaseImageProcessor] = None
        self._id_to_label: Dict[int, str] = {}

    def _resolve_model_folder(self, repo_id: str, revision: str) -> Path:
        """Resolve model folder from artifacts_path or HF download."""

        def download_wrapper(download_repo_id: str, download_revision: str) -> Path:
            _log.info(
                "Downloading %s model from HuggingFace: %s@%s",
                self._model_family_name,
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
        except ImportError:
            # transformers >= 5 gates its image-processor classes behind the
            # torch/torchvision backends. In a torch-free environment the ONNX
            # object-detection engine still needs preprocessing, so fall back to a
            # dependency-light numpy processor built from preprocessor_config.json.
            _log.info(
                "transformers image processor unavailable (torch-free environment); "
                "using numpy preprocessing fallback for %s",
                model_folder,
            )
            return NumpyImageProcessor.from_config_file(preprocessor_config)
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

        raise TypeError(f"Unsupported score value type: {type(value)!r}")

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
