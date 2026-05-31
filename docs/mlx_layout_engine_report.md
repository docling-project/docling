# MLX Layout Engine Integration Report

## Overview

This document summarizes the integration of a native MLX object-detection engine into Docling's standard PDF pipeline, enabling Apple Silicon-accelerated layout analysis using the `mlx-community/docling-layout-heron-101-mlx-bf16` model.

## Implementation Plan

The goal was to add a new `MLX` engine type to Docling's pluggable object-detection system, allowing the RT-DETR v2 layout model (converted to MLX bfloat16 format) to run natively on Apple Silicon via the Metal backend.

### Files Changed

| File | Description |
|------|-------------|
| `docling/models/inference_engines/object_detection/base.py` | Added `MLX = "mlx"` to the `ObjectDetectionEngineType` enum |
| `docling/datamodel/object_detection_engine_options.py` | Added `MlxObjectDetectionEngineOptions` with Pydantic auto-registration |
| `docling/models/inference_engines/object_detection/mlx_engine.py` | New engine class using `mlx_vlm.models.rt_detr_v2` for inference |
| `docling/models/inference_engines/object_detection/factory.py` | Added MLX dispatch branch to the engine factory |
| `docling/datamodel/stage_model_specs.py` | Added `layout_heron_mlx` preset and MLX support in `from_preset()` |
| `docling/datamodel/pipeline_options.py` | Registered the new preset |
| `docling/models/stages/layout/layout_object_detection_model.py` | Fixed label mapping to handle hyphens and spaces in model label names |
| `pyproject.toml` | Added `models-layout-mlx` optional dependency group |

### Architecture

The engine follows the same pattern as the existing Transformers and ONNX Runtime engines:

```
LayoutObjectDetectionOptions.from_preset("layout_heron_mlx")
    -> ObjectDetectionStagePresetMixin resolves preset
    -> MlxObjectDetectionEngineOptions created
    -> Factory dispatches to MlxObjectDetectionEngine
    -> Engine loads model via mlx_vlm.models.rt_detr_v2
    -> RTDetrV2Predictor handles inference
    -> DetectionResult mapped to ObjectDetectionEngineOutput
```

Thread safety is enforced via a global lock (`_MLX_GLOBAL_LOCK`), consistent with the existing VLM MLX engine.

## Issues Encountered

### 1. Module Naming Mismatch in `mlx-vlm`

**Problem:** The HuggingFace model card documented usage with `mlx_vlm.models.rt_detr_v2`, but `mlx-vlm` v0.5.0 (the latest PyPI release) only ships an `rfdetr` module -- a different architecture. The `rt_detr_v2` module exists on the `mlx-vlm` GitHub main branch but has not been released to PyPI.

**Resolution:** Install `mlx-vlm` from the GitHub main branch:

```bash
pip install 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main'
```

The `pyproject.toml` dependency spec uses `>=0.5.0` with a comment documenting the git install requirement until the next PyPI release.

### 2. Weight Format Incompatibility with `rfdetr` Module

**Problem:** When attempting to use the `rfdetr` module (available in v0.5.0) as a fallback, the model weights failed to load. The MLX safetensors file uses RT-DETR v2 weight naming conventions (e.g., `decoder.bbox_embed`, `vision.backbone.encoder.stages`) that are incompatible with the RF-DETR architecture's expected parameter names.

**Resolution:** The `rt_detr_v2` module from the GitHub main branch correctly matches the weight format. No model re-conversion was necessary.

### 3. `mlx-vlm` `load_model()` Dispatch Failure

**Problem:** The `mlx_vlm.utils.load_model()` function dispatches to model modules based on `model_type` in `config.json`. The model's config declares `model_type: "rt_detr_v2"`, but v0.5.0's `MODEL_REMAPPING` dict only maps `"rf-detr"` to `"rfdetr"` -- it has no entry for `"rt_detr_v2"`.

**Resolution:** Resolved by the same fix as Issue 1. The GitHub main branch includes the `rt_detr_v2` module, which is discovered automatically by `load_model()` when iterating over `mlx_vlm.models.*`.

### 4. Label Name Format Mismatch

**Problem:** The MLX model's `config.json` uses mixed-case hyphenated label names (e.g., `"List-item"`, `"Key-Value Region"`), while Docling's `DocItemLabel` enum uses uppercase underscore format (`LIST_ITEM`, `KEY_VALUE_REGION`). The existing `_build_label_map()` method only called `.upper()`, producing `"LIST-ITEM"` which failed the enum lookup.

**Resolution:** Updated `_build_label_map()` in `layout_object_detection_model.py` to normalize labels by replacing hyphens and spaces with underscores before the uppercase conversion. This fix is backward-compatible with models that already use underscore-separated names.

### 5. Module Boundary Violation (`tach`)

**Problem:** The initial implementation imported `resolve_model_artifacts_path` from `docling.models.inference_engines.vlm._utils`, which violated Docling's module boundary rules enforced by `tach`. The object-detection module is not allowed to depend on the VLM module.

**Resolution:** Replaced the cross-module import with equivalent inline logic in the MLX engine's `_resolve_model_folder()` method: check artifacts cache, fall back to HuggingFace download via `HuggingFaceModelDownloadMixin`.

## Deployment Instructions (macOS / Apple Silicon)

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- `uv` package manager

### Setup

Clone the fork and install with the MLX layout extras:

```bash
git clone https://github.com/eusip/docling.git
cd docling
git checkout feat/mlx-layout-engine
uv sync --all-extras
```

Install `mlx-vlm` from GitHub main (required until the next PyPI release includes the `rt_detr_v2` module):

```bash
uv pip install 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main'
```

### Usage

```python
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    ThreadedPdfPipelineOptions,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

# Configure the MLX layout preset
layout_opts = LayoutObjectDetectionOptions.from_preset("layout_heron_mlx")
pipeline_options = ThreadedPdfPipelineOptions(layout_options=layout_opts)

# Create converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

# Convert a PDF
result = converter.convert("document.pdf")
print(result.document.export_to_markdown())
```

The model (~153 MB) is downloaded automatically from HuggingFace on first use and cached locally.

### Verification

```bash
# Run project checks
make validate

# Run tests
make test
```
