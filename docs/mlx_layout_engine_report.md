# MLX Layout Engine Integration Report

## Overview

This document summarizes the native MLX object-detection engine for Docling's
standard PDF pipeline. The engine runs RT-DETR v2 layout analysis on Apple
Silicon through Metal, using the
`mlx-community/docling-layout-heron-101-mlx-bf16` checkpoint.

## Architecture

The engine plugs into the existing object-detection system:

```
LayoutObjectDetectionOptions.from_preset("layout_heron_mlx")
    -> ObjectDetectionStagePresetMixin resolves the preset
    -> MlxObjectDetectionEngineOptions is created
    -> the factory dispatches to MlxObjectDetectionEngine
    -> the engine loads the model through mlx_vlm.models.rt_detr_v2
    -> RTDetrV2Predictor runs inference
    -> DetectionResult is mapped to ObjectDetectionEngineOutput
```

The engine shares the Hugging Face model resolver and label-map loader with the
Transformers and ONNX Runtime engines. Each MLX engine family serializes its own
calls with its own lock. The object-detection lock is not shared with the VLM
engine.

## Installation

`uv.lock` pins `mlx-vlm>=0.7.0,<1.0.0` from PyPI. No git install is required.

```sh
pip install "docling[layout-mlx]"
# or, for a slim install:
pip install "docling-slim[models-layout-mlx]"
```

The `mlx-vlm` dependency is gated to macOS on Apple Silicon. The
`docling[vlm]` and `docling-slim[all]` bundles also install it through
`models-vlm-inline`.

## Usage

CLI:

```sh
docling document.pdf --layout-preset layout_heron_mlx --to md --output /tmp/output
```

Python SDK:

```python
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
)

pipeline_options = PdfPipelineOptions(
    layout_options=LayoutObjectDetectionOptions.from_preset("layout_heron_mlx"),
)
```

The model is downloaded from Hugging Face on first use and cached locally. For
offline use, prefetch it with `docling-tools models download layout_mlx` and
pass `--artifacts-path`. The checkpoint is stored as
`mlx-community--docling-layout-heron-101-mlx-bf16`.

## Implementation notes

- **Checkpoint format.** The model is pre-converted to MLX bfloat16 safetensors.
  Docling does not convert PyTorch weights.
- **Sanitize bypass.** The checkpoint records `{"format": "mlx"}` in its
  safetensors metadata. `mlx-vlm` 0.7.x calls `sanitize_weights` unconditionally,
  so Docling temporarily disables `Model.sanitize` while loading. Version 0.5.0
  honored the marker; the behavior regressed in 0.7.x.
- **Sharded checkpoints.** The engine reads metadata from
  `model.safetensors.index.json` when the checkpoint is sharded. A broken index
  raises a clear error.
- **Preprocessing.** The engine loads `RTDetrV2Processor.from_pretrained`, so
  resize and normalization follow the checkpoint's `preprocessor_config.json`.
- **Batching.** Inference uses one `predict_batch` call per page batch under a
  global lock.

## Limitations

- The engine requires macOS on Apple Silicon. Other platforms get a clear
  `RuntimeError` during initialization.
- There is no automatic fallback to the Transformers engine.
- `accelerator_options`, `--device`, and `--num-threads` do not affect the MLX
  engine. It always uses Metal.
- `--layout-preset` applies only to the standard and legacy PDF pipelines. The
  CLI warns and ignores it for the native, VLM, and ASR pipelines.

## Verification

```sh
uv run python -c "from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions as L; print(L.list_preset_ids())"
uv run pytest tests/test_mlx_object_detection_engine.py tests/test_layout_migration.py tests/test_model_downloader_layout.py -q
make validate
```
