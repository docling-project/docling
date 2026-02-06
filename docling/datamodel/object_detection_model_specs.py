"""Object detection model specifications - constant definitions."""

from docling.datamodel.stage_model_specs import (
    EngineModelConfig,
    ObjectDetectionModelSpec,
    ObjectDetectionStagePreset,
)
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineType,
)

# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

# Table Model (ONNX)
TABLE_MODEL_RTDETR = ObjectDetectionModelSpec(
    name="table_model_rtdetr",
    repo_id="docling-project/table_model_2026_02_02",
    revision="main",
    engine_overrides={
        ObjectDetectionEngineType.ONNXRUNTIME: EngineModelConfig(
            extra_config={"model_filename": "model.onnx"}
        )
    },
)

# Layout Heron (ONNX) - TBD: update repo_id when published to HuggingFace
LAYOUT_HERON_ONNX = ObjectDetectionModelSpec(
    name="layout_heron_onnx",
    repo_id="<TBD-local-or-future-hf-repo>",  # Update when available
    revision="main",
    engine_overrides={
        ObjectDetectionEngineType.ONNXRUNTIME: EngineModelConfig(
            extra_config={"model_filename": "model.onnx"}
        )
    },
)


# =============================================================================
# PRESETS
# =============================================================================

LAYOUT_OBJECT_DETECTION_PRESET = ObjectDetectionStagePreset(
    preset_id="layout_objdet_heron",
    name="Layout Heron (ONNX)",
    description="RT-DETR layout detection using Heron ONNX export",
    model_spec=LAYOUT_HERON_ONNX,
    default_engine_type=ObjectDetectionEngineType.ONNXRUNTIME,
)

TABLE_OBJECT_DETECTION_PRESET = ObjectDetectionStagePreset(
    preset_id="table_rtdetr",
    name="RT-DETR Table Structure",
    description="RT-DETRv2 ONNX model for table structure detection",
    model_spec=TABLE_MODEL_RTDETR,
    default_engine_type=ObjectDetectionEngineType.ONNXRUNTIME,
)
