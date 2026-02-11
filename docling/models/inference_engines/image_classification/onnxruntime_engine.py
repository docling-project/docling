"""ONNX Runtime implementation for image-classification models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.image_classification_engine_options import (
    OnnxRuntimeImageClassificationEngineOptions,
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


class OnnxRuntimeImageClassificationEngine(HfImageClassificationEngineBase):
    """ONNX Runtime engine for image-classification models."""

    def __init__(
        self,
        *,
        options: OnnxRuntimeImageClassificationEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        """Initialize the ONNX Runtime image-classification engine."""
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.options: OnnxRuntimeImageClassificationEngineOptions = options
        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[Path] = None
        self._input_name: Optional[str] = options.input_name
        self._output_name: Optional[str] = options.output_name

    def _resolve_model_artifacts(self) -> tuple[Path, Path]:
        """Resolve model artifacts from artifacts_path or HF download."""
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

    def _resolve_model_filename(self) -> str:
        """Determine which ONNX filename to load."""
        filename = self.options.model_filename
        extra_filename = self._model_config.extra_config.get("model_filename")
        if extra_filename and isinstance(extra_filename, str):
            filename = extra_filename
        return filename

    def _resolve_configured_input_name(self) -> Optional[str]:
        """Determine configured ONNX input name from options/model overrides."""
        input_name = self.options.input_name
        extra_input_name = self._model_config.extra_config.get("input_name")
        if extra_input_name and isinstance(extra_input_name, str):
            input_name = extra_input_name
        return input_name

    def _resolve_configured_output_name(self) -> Optional[str]:
        """Determine configured ONNX output name from options/model overrides."""
        output_name = self.options.output_name
        extra_output_name = self._model_config.extra_config.get("output_name")
        if extra_output_name and isinstance(extra_output_name, str):
            output_name = extra_output_name
        return output_name

    def _resolve_input_name(self, session: ort.InferenceSession) -> str:
        """Resolve ONNX input name, with session introspection fallback."""
        input_nodes = session.get_inputs()
        if not input_nodes:
            raise RuntimeError("ONNX model exposes no inputs")

        available_input_names = {node.name for node in input_nodes}
        configured_input_name = self._resolve_configured_input_name()
        if configured_input_name is not None:
            if configured_input_name in available_input_names:
                return configured_input_name
            raise RuntimeError(
                "Configured ONNX input name "
                f"'{configured_input_name}' not found in model inputs "
                f"{sorted(available_input_names)}"
            )

        return input_nodes[0].name

    def _resolve_output_name(self, session: ort.InferenceSession) -> str:
        """Resolve ONNX output name, with session introspection fallback."""
        output_nodes = session.get_outputs()
        if not output_nodes:
            raise RuntimeError("ONNX model exposes no outputs")

        available_output_names = {node.name for node in output_nodes}
        configured_output_name = self._resolve_configured_output_name()
        if configured_output_name is not None:
            if configured_output_name in available_output_names:
                return configured_output_name
            raise RuntimeError(
                "Configured ONNX output name "
                f"'{configured_output_name}' not found in model outputs "
                f"{sorted(available_output_names)}"
            )

        return output_nodes[0].name

    def initialize(self) -> None:
        """Initialize ONNX session and preprocessor."""
        import onnxruntime as ort

        _log.info("Initializing ONNX Runtime image-classification engine")

        model_folder, self._model_path = self._resolve_model_artifacts()
        _log.debug("Using ONNX model at %s", self._model_path)

        self._processor = self._load_preprocessor(model_folder)
        self._id_to_label = self._load_label_mapping(model_folder)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._accelerator_options.num_threads
        providers = self._resolve_providers()

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self._input_name = self._resolve_input_name(self._session)
        self._output_name = self._resolve_output_name(self._session)
        _log.debug("Using ONNX input name: %s", self._input_name)
        _log.debug("Using ONNX output name: %s", self._output_name)

        self._initialized = True
        _log.info(
            "ONNX Runtime image-classification engine ready (providers=%s)",
            self._session.get_providers(),
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
                "Unsupported ONNX device '%s' for image classification. Falling back to CPU provider.",
                device,
            )
        return ["CPUExecutionProvider"]

    def predict_batch(
        self, input_batch: List[ImageClassificationEngineInput]
    ) -> List[ImageClassificationEngineOutput]:
        """Run inference on a batch of inputs."""
        if not input_batch:
            return []
        if (
            self._session is None
            or self._processor is None
            or self._input_name is None
            or self._output_name is None
        ):
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        images = [item.image.convert("RGB") for item in input_batch]
        inputs = self._processor(images=images, return_tensors="np")
        input_tensor = np.asarray(inputs["pixel_values"], dtype=np.float32)

        output_tensors = self._session.run(
            [self._output_name],
            {
                self._input_name: input_tensor,
            },
        )

        if len(output_tensors) < 1:
            raise RuntimeError(
                "Expected ONNX model to return at least 1 output containing logits"
            )

        logits_batch = np.asarray(output_tensors[0], dtype=np.float32)
        if logits_batch.ndim != 2:
            raise RuntimeError(
                "Expected ONNX logits output shape [batch_size, num_classes], "
                f"got shape={logits_batch.shape}"
            )

        probs_batch = self._softmax(logits_batch)

        batch_outputs: List[ImageClassificationEngineOutput] = []
        for idx, input_item in enumerate(input_batch):
            probs = probs_batch[idx]

            # Use argpartition for efficiency when top_k is specified
            if self.options.top_k is not None:
                k = min(self.options.top_k, len(probs))
                # argpartition finds top k elements (unsorted), then we sort just those
                top_k_indices = np.argpartition(-probs, k - 1)[:k]
                top_k_indices_sorted = top_k_indices[np.argsort(-probs[top_k_indices])]
                labels = top_k_indices_sorted
                scores = probs[labels]
            else:
                labels = np.argsort(-probs)
                scores = probs[labels]

            batch_outputs.append(
                self._build_output(
                    input_item=input_item,
                    labels=labels,
                    scores=scores,
                )
            )

        return batch_outputs

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
