# Layout migration: implemented state

**Status:** implemented in PR #3914.
**Authority:** the implementation and regenerated ground truth are authoritative; this document
records the resulting design rather than the earlier migration proposal.

## Result

`LayoutObjectDetectionModel` is the only implementation that runs layout inference. It uses the
pluggable object-detection engines and reads labels from each model repository's `config.json`.
Nothing under `docling/` imports `docling_ibm_models.layoutmodel` anymore.

`LayoutObjectDetectionOptions` is the default for `PdfPipelineOptions`:

```python
layout_options: Annotated[BaseLayoutOptions, Field(...)] = Field(
    default_factory=LayoutObjectDetectionOptions
)
```

The factory is important: it constructs the nested engine options at pipeline-option creation time,
so `settings.inference.compile_torch_models = False` is honored. A preconstructed class-level default
would freeze the earlier compile setting.

`LayoutOptions` and `LayoutModel` remain only as a deprecated public compatibility boundary.
`LayoutModel` subclasses `LayoutObjectDetectionModel`, translates `LayoutOptions`, and inherits the
actual inference implementation. New and internal code uses `LayoutObjectDetectionOptions` directly.

## Options and model presets

`BaseLayoutOptions` owns the three post-processing controls shared by both option types:

- `keep_empty_clusters=False`
- `skip_cell_assignment=False`
- `create_orphan_clusters=True`

The common `create_orphan_clusters=True` default removes the previous behavioral drift and lets the
standard, legacy, and threaded VLM pipelines read the option directly without type checks.

`ObjectDetectionEngineOptionsMixin.engine_options` has a default factory returning
`TransformersObjectDetectionEngineOptions`. Consequently, `LayoutObjectDetectionOptions()` is valid
without an explicit engine configuration.

The supported presets are:

| preset | repository | default engine |
|---|---|---|
| `layout_heron_default` | `docling-project/docling-layout-heron` | Transformers; ONNX override available |
| `layout_heron_101` | `docling-project/docling-layout-heron-101` | Transformers |
| `layout_egret_medium` | `docling-project/docling-layout-egret-medium` | Transformers |
| `layout_egret_large` | `docling-project/docling-layout-egret-large` | Transformers |
| `layout_egret_xlarge` | `docling-project/docling-layout-egret-xlarge` | Transformers |

All five repositories must expose an `id2label` map whose values resolve to `DocItemLabel`. The
config-only parametrized test in `tests/test_layout_migration.py` is the permanent guard for that
external contract.

`DOCLING_LAYOUT_V2` is unsupported. Deprecated `LayoutOptions` callers selecting it receive a
`DeprecationWarning` and are translated to Heron.

The deprecated compatibility surface was deliberately reduced to what the shim needs:

- `LayoutModelConfig` contains only `name`, `repo_id`, and `revision`.
- `LayoutModelType`, `model_path`, `supported_devices`, and `model_repo_folder` were removed.
- `LayoutModel.download_models` and the old direct `artifacts_path/<model_path>` branch were removed.
- The layout label groups moved to module-level constants in `base_layout_model.py`; deprecated
  `LayoutModel` class aliases were not retained.

User-defined `LayoutModelConfig` instances still translate by copying those three fields into an
`ObjectDetectionModelSpec`. There is no preset-name lookup.

## Inference behavior

`LayoutObjectDetectionModel` now:

1. Collects all valid pages with images and calls `engine.predict_batch(...)` once. This makes
   `layout_batch_size` effective and records one `TimeRecorder(..., "layout")` interval per batch.
2. Preserves page ordering and existing empty predictions when invalid pages or missing images are
   skipped.
3. Converts image-pixel boxes into page space and clamps every coordinate to the page bounds.
4. Drops detections whose label id is absent from the model's own `id2label`, warning once per
   unmapped id instead of silently converting it to `TEXT`.
5. Implements `settings.debug.visualize_raw_layout` on the shared inference path.

Page-space scaling is intentional. Raster dimensions are integral while PDF page dimensions may be
fractional, so some boxes move by less than one point compared with the old unscaled path.

The Transformers engine supports CPU, CUDA, MPS, and XPU. XPU resolves to `torch.device("xpu")`.

## Pipeline and download integration

- `StandardPdfPipeline` and `LegacyStandardPdfPipeline` resolve layout models through the factory
  and pass the shared post-processing fields through directly.
- `ThreadedLayoutVlmPipelineOptions.layout_options` is concretely typed as
  `LayoutObjectDetectionOptions` and uses a default factory with `skip_cell_assignment=True`.
  It does not accept or preserve `LayoutOptions` for compatibility.
- `ThreadedLayoutVlmPipeline` also resolves the model through the layout factory and no longer
  constructs `LayoutModel` directly.
- Layout model prefetch uses the default object-detection spec and downloads both its base repository
  and every engine override repository, including the Heron ONNX repository.
- Documentation names the five presets, marks layout-v2 unsupported, and uses
  `LayoutObjectDetectionOptions.from_preset(...)` in examples.

## Deprecated shim behavior

Construction of `LayoutOptions` and `LayoutModel` emits `DeprecationWarning`.
`LayoutFactory` still registers the exact mappings required for deprecated public callers:

- `LayoutOptions` -> `LayoutModel`
- `LayoutObjectDetectionOptions` -> `LayoutObjectDetectionModel`

The translation preserves `keep_empty_clusters`, `skip_cell_assignment`, and
`create_orphan_clusters`, preserves custom repositories and revisions, and uses the current default
engine options. No inference logic remains in the shim.

## Serialization

`ThreadedLayoutVlmPipelineOptions` round-trips through `model_dump()` / `model_validate()` with a
concrete `LayoutObjectDetectionOptions`, including its model spec and `skip_cell_assignment=True`.

The broader `PdfPipelineOptions` and `ThreadedPdfPipelineOptions` fields remain annotated as
`BaseLayoutOptions`. Their generic Pydantic round-trip still reconstructs `BaseLayoutOptions`; this
migration does not introduce a discriminated layout-options union.

## Verification and reference data

`tests/test_layout_migration.py` guards:

- the new default type and Heron model;
- compile opt-out evaluation at options-construction time;
- experimental threaded VLM serialization;
- translation of all five supported legacy specs and custom specs;
- post-processing flag preservation and layout-v2 fallback;
- exact factory dispatch for both public option types;
- all five repository label maps;
- one batched engine call with stable handling of unusable pages;
- picture-internal non-caption text retention; and
- absence of `docling_ibm_models.layoutmodel` imports under `docling/`.

Ground truth regeneration must run on Linux. The checked-in reference data is authoritative; the
earlier expectation that no regeneration would be needed is withdrawn. The migration updates:

- `tests/data/docx/groundtruth/textbox.docx.{itxt,json,md}`
- `tests/data/pdf/groundtruth/right_to_left_03.{doctags.txt,json}`
