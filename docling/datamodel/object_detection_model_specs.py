"""Object detection model specifications - constant definitions."""

from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec

# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

# Table Model (ONNX)
TABLE_MODEL_RTDETR = ObjectDetectionModelSpec(
    name="table_model_rtdetr",
    repo_id="docling-project/table_model_2026_02_02",
    revision="main",
)

# Layout Heron (ONNX) - TBD: update repo_id when published to HuggingFace
LAYOUT_HERON_ONNX = ObjectDetectionModelSpec(
    name="layout_heron_onnx",
    repo_id="<TBD-local-or-future-hf-repo>",  # Update when available
    revision="main",
)
