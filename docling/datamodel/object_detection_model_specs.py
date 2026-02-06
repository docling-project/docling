"""Object detection model specifications - constant definitions."""

from docling.datamodel.stage_model_specs import (
    EngineModelConfig,
    ObjectDetectionModelSpec,
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
