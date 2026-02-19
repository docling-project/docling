"""KServe v2 remote implementation for object-detection models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    ApiKserveV2ObjectDetectionEngineOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.common import KserveV2HttpClient
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


class ApiKserveV2ObjectDetectionEngine(HfObjectDetectionEngineBase):
    """Remote object-detection engine backed by KServe v2-compatible serving."""

    def __init__(
        self,
        *,
        enable_remote_services: bool,
        options: ApiKserveV2ObjectDetectionEngineOptions,
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
        self.options: ApiKserveV2ObjectDetectionEngineOptions = options
        self._kserve_client: Optional[KserveV2HttpClient] = None
        self._input_images_name: Optional[str] = None
        self._input_orig_target_sizes_name: Optional[str] = None
        self._output_labels_name: Optional[str] = None
        self._output_boxes_name: Optional[str] = None
        self._output_scores_name: Optional[str] = None

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

    def _resolve_tensor_names(self) -> tuple[str, str, str, str, str]:
        if self._kserve_client is None:
            raise RuntimeError("KServe v2 client is not initialized.")

        metadata = self._kserve_client.get_model_metadata()
        if len(metadata.inputs) < 2:
            raise RuntimeError(
                "Expected object-detection model metadata to expose at least 2 inputs "
                f"(images, orig_target_sizes), got {len(metadata.inputs)}."
            )
        if len(metadata.outputs) < 3:
            raise RuntimeError(
                "Expected object-detection model metadata to expose at least 3 outputs "
                f"(labels, boxes, scores), got {len(metadata.outputs)}."
            )

        input_images_name = metadata.inputs[0].name
        input_orig_target_sizes_name = metadata.inputs[1].name
        output_labels_name = metadata.outputs[0].name
        output_boxes_name = metadata.outputs[1].name
        output_scores_name = metadata.outputs[2].name

        return (
            input_images_name,
            input_orig_target_sizes_name,
            output_labels_name,
            output_boxes_name,
            output_scores_name,
        )

    def initialize(self) -> None:
        """Initialize preprocessor/labels and prepare remote client."""
        _log.info("Initializing KServe v2 object-detection engine")

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
        (
            self._input_images_name,
            self._input_orig_target_sizes_name,
            self._output_labels_name,
            self._output_boxes_name,
            self._output_scores_name,
        ) = self._resolve_tensor_names()

        self._initialized = True
        _log.info(
            "KServe v2 object-detection engine ready (inputs=[%s, %s], outputs=[%s, %s, %s])",
            self._input_images_name,
            self._input_orig_target_sizes_name,
            self._output_labels_name,
            self._output_boxes_name,
            self._output_scores_name,
        )

    def predict_batch(
        self, input_batch: List[ObjectDetectionEngineInput]
    ) -> List[ObjectDetectionEngineOutput]:
        """Run inference on a batch of images against a KServe v2 endpoint."""
        if not input_batch:
            return []
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Type narrowing: _initialized guarantees these are non-None
        assert self._processor is not None
        assert self._kserve_client is not None
        assert self._input_images_name is not None
        assert self._input_orig_target_sizes_name is not None
        assert self._output_labels_name is not None
        assert self._output_boxes_name is not None
        assert self._output_scores_name is not None

        images = [item.image.convert("RGB") for item in input_batch]
        processed_inputs = self._processor(images=images, return_tensors="np")

        pixel_values = np.asarray(processed_inputs["pixel_values"], dtype=np.float32)
        orig_sizes = np.asarray(
            [[image.width, image.height] for image in images],
            dtype=np.int64,
        )

        outputs = self._kserve_client.infer(
            inputs={
                self._input_images_name: pixel_values,
                self._input_orig_target_sizes_name: orig_sizes,
            },
            output_names=[
                self._output_labels_name,
                self._output_boxes_name,
                self._output_scores_name,
            ],
            request_parameters=self.options.request_parameters,
        )
        try:
            labels_batch = outputs[self._output_labels_name]
            boxes_batch = outputs[self._output_boxes_name]
            scores_batch = outputs[self._output_scores_name]
        except KeyError as exc:
            raise RuntimeError(
                "Missing one or more expected KServe v2 outputs: "
                f"{self._output_labels_name}, "
                f"{self._output_boxes_name}, "
                f"{self._output_scores_name}"
            ) from exc

        if len(labels_batch) != len(input_batch):
            raise RuntimeError(
                "KServe v2 output batch size mismatch for labels: "
                f"expected {len(input_batch)}, got {len(labels_batch)}"
            )

        batch_outputs: List[ObjectDetectionEngineOutput] = []
        for idx, input_item in enumerate(input_batch):
            batch_outputs.append(
                self._build_output(
                    input_item=input_item,
                    labels=labels_batch[idx],
                    scores=scores_batch[idx],
                    boxes=boxes_batch[idx],
                    apply_score_threshold=True,
                )
            )

        return batch_outputs
