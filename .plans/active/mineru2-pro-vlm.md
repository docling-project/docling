# MinerU2-Pro VLM integration

**Status:** implementation in progress

## Objective

Add the `opendatalab/MinerU2.5-Pro-2604-1.2B` document-parsing model to the
stage-model preset system and expose it through the standard VLM CLI as
`--vlm-model mineru2_pro`.

Support the current VLM runtime with:

- Hugging Face Transformers;
- MLX through a compatible converted checkpoint; and
- generic/OpenAI-compatible API engines.

Produce structured `DoclingDocument` output with page provenance, semantic
labels, formulas, pictures, and OTSL tables.

## Model contract

MinerU2-Pro full-page conversion is a two-step operation:

1. Run `Layout Detection:` against a 1036 x 1036 page image. The model emits
   normalized bounding boxes, semantic region types, and optional rotation and
   paragraph-continuation markers.
2. Crop recognized regions from the original rendered page and run the
   type-specific prompt (`Text Recognition:`, `Table Recognition:`, or
   `Formula Recognition:`).

Standalone recognition of an already-cropped region remains possible through
the existing raw `process_images()` API, but the `mineru2_pro` document-convert
preset must use the two-step operation.

The first layout stage remains MinerU-native. Reusing
`experimental/pipeline/threaded_layout_vlm_pipeline.py` is out of scope: that
pipeline uses a separate Docling layout model, injects its boxes into a single
full-page DOCTAGS request, and is tied to the legacy VLM option/model stack.

## Public configuration

Add `ResponseFormat.MINERU2 = "mineru2"`.

Add `VLM_CONVERT_MINERU2_PRO` with:

- preset id `mineru2_pro`;
- official Transformers/API repository
  `opendatalab/MinerU2.5-Pro-2604-1.2B`;
- a compatible BF16 MLX conversion;
- Qwen2-VL image-to-text loading through Transformers;
- the official layout prompt and deterministic generation defaults; and
- explicit Transformers, MLX, generic API, OpenAI-compatible API, and LM Studio
  support.

The selected MLX conversion stores `tie_word_embeddings=true` only in its
nested text configuration, while the mlx-vlm Qwen2-VL loader reads the root
setting. Enable the preset's scoped `mlx_tied_word_embeddings` compatibility
mode so the omitted `lm_head.weight` is correctly backed by the token embedding
matrix rather than treated as a malformed checkpoint.

Register the preset on `VlmConvertOptions`. The CLI already derives its
`--vlm-model` choices from that registry, so registration is the CLI exposure
point.

## Implementation

### 1. MinerU utilities

Create `docling/utils/mineru_utils.py` containing model-specific, runtime-neutral
helpers:

- parse and validate the native layout envelope;
- normalize 0-1000 coordinates to 0-1 page regions;
- preserve rotation and paragraph-continuation metadata;
- discard invalid/unsupported regions and table-internal duplicate regions;
- create and rotate region crops;
- select the recognition prompt by region type;
- serialize the completed region list for the page prediction;
- parse that region list into `DoclingDocument`; and
- convert MinerU OTSL table output into `TableData`.

Keep image/chart analysis disabled initially, matching the official client's
default. Picture and chart regions are retained as pictures without issuing an
extra description request.

### 2. Two-step stage execution

Extend `VlmConvertModel` only when the selected response format is
`ResponseFormat.MINERU2`:

1. prepare the square layout images;
2. run a batched layout request through the already-selected `BaseVlmEngine`;
3. parse regions and prepare all region crops;
4. run the recognition crops as a second engine batch;
5. associate recognition output with its source page/region;
6. aggregate stop reason, token count, and generation timing; and
7. attach the serialized structured response to each page.

All other VLM presets retain the existing one-pass path. Transformers, MLX, and
API behavior share the same orchestration and engine interface.

### 3. Pipeline finalization

Add a `ResponseFormat.MINERU2` branch to `VlmPipeline._finalize_page_document`
that invokes the MinerU parser and then uses the existing page finalization and
image-cropping behavior.

Malformed layout or serialized region data should be handled defensively: warn,
preserve valid regions, and return an empty page document when no valid regions
remain rather than crashing on individual bad blocks.

### 4. Tests

Add focused tests for:

- preset registration and CLI-visible preset id;
- Transformers, MLX, and API repository/configuration resolution;
- layout parsing, coordinate normalization, rotations, continuation markers,
  and invalid-region rejection;
- nested table-region filtering and crop/prompt preparation;
- two-step VLM stage batching and page/region association;
- MinerU JSON-to-`DoclingDocument` semantic mapping and provenance; and
- OTSL tables, including row/column spans.

Update the response-format consistency test to include `MINERU2`.

## Documentation

Update:

- `docs/usage/model_catalog.md` with the preset, supported runtimes, native
  two-step output, and MLX checkpoint note; and
- `docs/reference/cli.md` so the documented `--vlm-model` choices include
  `mineru2_pro`.

## Verification

Run, in order:

1. the new MinerU-focused tests;
2. existing VLM preset/runtime and VLM conversion-model tests;
3. relevant CLI tests or a CLI help smoke test;
4. `make validate`, reviewing and retaining any formatter-generated changes;
5. rerun affected tests after validation; and
6. `make check` if validation or the broader changes indicate an additional
   read-only verification pass is useful.

Do not download model weights or run heavyweight end-to-end inference as part
of the default test pass.

## Risks and constraints

- The MLX checkpoint is community-published rather than an official OpenDataLab
  artifact; document that distinction and pin the selected repository explicitly.
- Region fan-out can make a page substantially more expensive than ordinary
  one-pass VLM conversion. Batch all crops across the incoming page batch.
- API servers must expose the raw model through an OpenAI-compatible image
  endpoint; a server-side MinerU wrapper is not assumed.
- The initial implementation covers the official default with image/chart
  analysis disabled. Cross-page table merging and image analysis are not part
  of this integration.
