# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""MLX-based object-detection engine for Apple Silicon."""

from __future__ import annotations

import importlib
import json
import logging
import platform
import struct
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    MlxObjectDetectionEngineOptions,
)
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.inference_engines.object_detection.hf_base import (
    HfObjectDetectionEngineBase,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)

# MLX models are not thread-safe. Each engine family serializes its own calls;
# object detection and VLM inference do not share a lock.
_MLX_GLOBAL_LOCK = threading.Lock()


def _is_apple_silicon() -> bool:
    """Return True on macOS running on Apple Silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _safetensors_metadata(path: Path) -> Dict[str, str]:
    """Read the metadata block from a single-file safetensors checkpoint.

    The container starts with an 8-byte little-endian header length followed
    by a JSON header. Reading it directly avoids a ``safetensors`` dependency.
    """
    try:
        with path.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
    except (OSError, ValueError, struct.error):
        return {}

    metadata = header.get("__metadata__", {})
    return metadata if isinstance(metadata, dict) else {}


def _checkpoint_metadata(model_folder: Path) -> Dict[str, str]:
    """Read safetensors metadata from a single-file or sharded checkpoint."""
    model_file = model_folder / "model.safetensors"
    if model_file.is_file():
        return _safetensors_metadata(model_file)

    index_file = model_folder / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return {}
        shard_names = sorted(
            {name for name in weight_map.values() if isinstance(name, str)}
        )
        if not shard_names:
            raise FileNotFoundError(
                f"Sharded safetensors index in {model_folder} does not reference "
                "any shard files."
            )
        for shard_name in shard_names:
            shard_path = model_folder / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Sharded safetensors index references missing shard: {shard_path}"
                )
            metadata = _safetensors_metadata(shard_path)
            if metadata:
                return metadata
        return {}

    raise FileNotFoundError(
        f"No safetensors checkpoint found in {model_folder}. Expected "
        "'model.safetensors' or 'model.safetensors.index.json'."
    )


def _identity_weights(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Return weights unchanged, to bypass ``mlx-vlm`` sanitization."""
    return weights


@contextmanager
def _bypass_mlx_sanitizer(model_cls: Any) -> Iterator[None]:
    """Temporarily disable ``Model.sanitize`` for pre-converted MLX weights.

    ``mlx-vlm`` 0.7.x calls ``sanitize_weights`` unconditionally, even when the
    safetensors metadata records ``{"format": "mlx"}``. Pre-converted RT-DETR
    v2 checkpoints are already in MLX layout, so sanitizing them transposes the
    convolution weights a second time. Version 0.5.0 honored the marker; the
    behavior regressed in 0.7.x.
    """
    original = model_cls.__dict__.get("sanitize")
    if original is None and getattr(model_cls, "sanitize", None) is None:
        raise RuntimeError(
            "Cannot bypass mlx-vlm sanitization: the loaded Model class has no "
            "sanitize method. The installed mlx-vlm version is incompatible "
            "with MLX-format RT-DETR v2 checkpoints."
        )

    had_own_sanitize = original is not None
    model_cls.sanitize = staticmethod(_identity_weights)
    try:
        yield
    finally:
        if had_own_sanitize:
            model_cls.sanitize = original
        else:
            del model_cls.sanitize


class MlxObjectDetectionEngine(HfObjectDetectionEngineBase):
    """MLX engine for object-detection inference on Apple Silicon.

    Uses ``mlx_vlm.models.rt_detr_v2`` to load and run RT-DETR v2 models
    converted to the MLX safetensors format.
    """

    def __init__(
        self,
        *,
        options: MlxObjectDetectionEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: Optional[AcceleratorOptions] = None,
        artifacts_path: Optional[Union[Path, str]] = None,
    ) -> None:
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options or AcceleratorOptions(),
            artifacts_path=artifacts_path,
        )
        self.options: MlxObjectDetectionEngineOptions = options
        self._predictor: Any = None

    def initialize(self) -> None:
        """Load the MLX model, processor, and predictor."""
        if self._initialized:
            return

        if not _is_apple_silicon():
            raise RuntimeError(
                "The MLX layout engine requires macOS on Apple Silicon "
                f"(detected {platform.system()} {platform.machine()})."
            )

        _log.info("Initializing MLX object-detection engine")

        try:
            importlib.import_module("mlx_vlm.models.rt_detr_v2")
            from mlx_vlm.models.rt_detr_v2.generate import RTDetrV2Predictor
            from mlx_vlm.models.rt_detr_v2.processing_rt_detr_v2 import (
                RTDetrV2Processor,
            )
            from mlx_vlm.models.rt_detr_v2.rt_detr_v2 import Model
            from mlx_vlm.utils import load_model
        except ModuleNotFoundError as exc:
            if exc.name == "transformers":
                raise ImportError(
                    "transformers is required for the MLX layout engine. "
                    "Install the models-layout-mlx extra."
                ) from exc
            if exc.name and exc.name.startswith("mlx_vlm"):
                raise ImportError(
                    "mlx-vlm>=0.7.0 is required for the MLX layout engine. "
                    "Install it with: pip install 'docling[layout-mlx]' or "
                    "'docling-slim[models-layout-mlx]'."
                ) from exc
            raise

        repo_id = self._model_config.repo_id
        revision = self._model_config.revision or "main"

        model_folder = self._resolve_model_folder(repo_id, revision)
        _log.debug("Using MLX model at %s", model_folder)

        self._id_to_label = self._load_label_mapping(model_folder)
        _log.debug("Loaded label mapping with %d labels", len(self._id_to_label))

        metadata = _checkpoint_metadata(model_folder)
        if metadata.get("format") == "mlx":
            with _MLX_GLOBAL_LOCK, _bypass_mlx_sanitizer(Model):
                model = load_model(model_folder)
        else:
            model = load_model(model_folder)

        processor = RTDetrV2Processor.from_pretrained(model_folder)
        self._predictor = RTDetrV2Predictor(
            model,
            processor=processor,
            threshold=self.options.score_threshold,
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

        images = [
            item.image if item.image.mode == "RGB" else item.image.convert("RGB")
            for item in input_batch
        ]

        with _MLX_GLOBAL_LOCK:
            results = self._predictor.predict_batch(images)

        if len(results) != len(input_batch):
            raise RuntimeError(
                f"MLX predictor returned {len(results)} results for "
                f"{len(input_batch)} inputs"
            )

        return [
            self._build_output(
                input_item=item,
                labels=result.labels,
                scores=result.scores,
                boxes=result.boxes,
                apply_score_threshold=False,
            )
            for item, result in zip(input_batch, results)
        ]
