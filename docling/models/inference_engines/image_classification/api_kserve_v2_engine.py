"""KServe v2 remote implementation for image-classification models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.image_classification_engine_options import (
    ApiKserveV2ImageClassificationEngineOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.common import KserveV2HttpClient
from docling.models.inference_engines.image_classification.base import (
    ImageClassificationEngineInput,
    ImageClassificationEngineOutput,
)
from docling.models.inference_engines.image_classification.hf_base import (
    HfImageClassificationEngineBase,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class ApiKserveV2ImageClassificationEngine(HfImageClassificationEngineBase):
    """Remote image-classification engine backed by KServe v2-compatible serving."""

    def __init__(
        self,
        *,
        enable_remote_services: bool,
        options: ApiKserveV2ImageClassificationEngineOptions,
        model_config: Optional[EngineModelConfig] = None,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]] = None,
    ):
        super().__init__(
            options=options,
            model_config=model_config,
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.options: ApiKserveV2ImageClassificationEngineOptions = options
        self._kserve_client: Optional[KserveV2HttpClient] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

        if not enable_remote_services:
            raise OperationNotAllowed(
                "Connections to remote services are only allowed when set explicitly. "
                "pipeline_options.enable_remote_services=True."
            )

    def _resolve_model_name(self) -> str:
        if self.options.model_name:
            return self.options.model_name

        return self._repo_id.replace("/", "--")

    def _resolve_model_version(self) -> Optional[str]:
        return self.options.model_version

    def _resolve_tensor_names(self) -> tuple[str, str]:
        if self._kserve_client is None:
            raise RuntimeError("KServe v2 client is not initialized.")

        metadata = self._kserve_client.get_model_metadata()
        if not metadata.inputs:
            raise RuntimeError(
                f"Expected image-classification model metadata to expose at least 1 input, "
                f"got {len(metadata.inputs)} inputs."
            )
        if not metadata.outputs:
            raise RuntimeError(
                f"Expected image-classification model metadata to expose at least 1 output, "
                f"got {len(metadata.outputs)} outputs."
            )

        input_name = metadata.inputs[0].name
        output_name = metadata.outputs[0].name
        return input_name, output_name

    def initialize(self) -> None:
        """Initialize preprocessor/labels and prepare remote client."""
        _log.info("Initializing KServe v2 image-classification engine")

        revision = self._model_config.revision or "main"
        model_folder = self._resolve_model_folder(
            repo_id=self._repo_id, revision=revision
        )

        self._processor = self._load_preprocessor(model_folder)
        self._id_to_label = self._load_label_mapping(model_folder)

        self._kserve_client = KserveV2HttpClient(
            base_url=str(self.options.url),
            model_name=self._resolve_model_name(),
            model_version=self._resolve_model_version(),
            timeout=self.options.timeout,
            headers=self.options.headers,
        )
        self._input_name, self._output_name = self._resolve_tensor_names()

        self._initialized = True
        _log.info(
            "KServe v2 image-classification engine ready (input=%s, output=%s)",
            self._input_name,
            self._output_name,
        )

    def predict_batch(
        self, input_batch: List[ImageClassificationEngineInput]
    ) -> List[ImageClassificationEngineOutput]:
        """Run inference on a batch of images against a KServe v2 endpoint."""
        if not input_batch:
            return []
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Type narrowing: _initialized guarantees these are non-None
        assert self._processor is not None
        assert self._kserve_client is not None
        assert self._input_name is not None
        assert self._output_name is not None

        images = [item.image.convert("RGB") for item in input_batch]
        processed_inputs = self._processor(images=images, return_tensors="np")
        pixel_values = np.asarray(processed_inputs["pixel_values"], dtype=np.float32)

        outputs = self._kserve_client.infer(
            inputs={self._input_name: pixel_values},
            output_names=[self._output_name],
            request_parameters=self.options.request_parameters,
        )
        try:
            logits_batch = outputs[self._output_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing expected KServe v2 output: {self._output_name}"
            ) from exc

        logits_batch = np.asarray(logits_batch, dtype=np.float32)
        if logits_batch.ndim != 2:
            raise RuntimeError(
                "Expected logits output shape [batch_size, num_classes], "
                f"got shape={logits_batch.shape}"
            )

        probs_batch = self._softmax(logits_batch)
        return self._build_batch_outputs_from_probabilities(
            input_batch=input_batch,
            probs_batch=probs_batch,
        )
