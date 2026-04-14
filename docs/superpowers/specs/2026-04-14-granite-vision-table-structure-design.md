# Granite Vision Table Structure Model — Design Spec

**Date:** 2026-04-14
**Author:** Eli Schwartz
**Status:** Approved

---

## Overview

Add a new `GraniteVisionTableStructureModel` that uses `ibm-granite/granite-4.0-3b-vision` with the `<tables_otsl>` prompt tag to extract table structure and cell text from document pages. This replaces the TableFormer dependency for users who prefer a VLM-based approach.

---

## Motivation

The main branch has no VLM-based table extraction. TableFormer (V1 and V2) is the only option. `granite-4.0-3b-vision` natively supports OTSL table output via a special `<tables_otsl>` prompt token — the same model already used for chart extraction in docling. This design adds a focused, minimal implementation following the same pattern.

---

## Architecture

### New files

- `docling/models/stages/table_structure/table_structure_model_granite_vision.py`

### Modified files

- `docling/datamodel/pipeline_options.py` — add `GraniteVisionTableStructureOptions`
- `docling/models/plugins/defaults.py` — register new engine in `table_structure_engines()`

### Class hierarchy

```
BaseTableStructureModel
  └── GraniteVisionTableStructureModel
```

No shared base with chart extraction — the two models have different parent classes (`BaseTableStructureModel` vs `BaseItemAndImageEnrichmentModel`) and different call signatures.

---

## Components

### `GraniteVisionTableStructureOptions`

```python
class GraniteVisionTableStructureOptions(BaseTableStructureOptions):
    kind: ClassVar[str] = "granite_vision_table"
```

No extra fields in V1.

### `GraniteVisionTableStructureModel`

| Attribute | Value |
|---|---|
| `_model_repo_id` | `"ibm-granite/granite-4.0-3b-vision"` |
| `_model_repo_folder` | `"ibm-granite--granite-4.0-3b-vision"` |
| `_model_repo_revision` | `"f0d034897bae1cd438c961c8c170a3a3089ebf01"` |
| Supported devices | CPU, CUDA (MPS excluded — same restriction as chart extraction) |

---

## Data Flow

For each page, `predict_tables()`:

1. **Get layout clusters** — filter `page.predictions.layout.clusters` for `TABLE` and `DOCUMENT_INDEX` labels
2. **Crop table images** — call `page.get_image(scale=1.0)` to get the full page image, then crop using the cluster bbox (which is in page coordinates at scale=1.0) → PIL `Image`
3. **Batch inference** — build one conversation per crop with prompt `"<tables_otsl>"`, call `model.generate()` in a single batched call (`padding=True`, `do_pad=True`, `max_new_tokens=model_max_length`)
4. **Decode outputs** — slice off input prompt tokens, decode generated tokens only (same pattern as `ChartExtractionModelGraniteVisionV4`)
5. **Parse OTSL** — call `_parse_otsl_output()` on each decoded string
6. **Build `Table` objects** — wrap parsed results into `Table` and insert into `TableStructurePrediction.table_map`

---

## OTSL Parsing

### Input format

The model produces inline-text OTSL like:
```
<ched>Name</ched><ched>Q1</ched><ched>Q2</ched><nl><fcel>Revenue</fcel><fcel>100</fcel><fcel>120</fcel><nl>
```

Token reference (from `docling_core.types.doc.tokens.TableToken`):

| Token | Meaning |
|---|---|
| `<fcel>text</fcel>` | Data cell with content |
| `<ecel/>` or `<ecel></ecel>` | Empty cell |
| `<ched>text</ched>` | Column header cell |
| `<rhed>text</rhed>` | Row header cell |
| `<srow>text</srow>` | Section row cell |
| `<lcel>` | Left-extension cell (colspan continuation) |
| `<ucel>` | Up-extension cell (rowspan continuation) |
| `<xcel>` | 2D extension cell (both) |
| `<nl>` | Row separator |

### Parser: `_parse_otsl_output(text: str) -> tuple[list[str], list[TableCell], int, int]`

Returns `(otsl_seq, table_cells, num_rows, num_cols)`.

**Algorithm:**
1. Use regex to extract `(tag_name, inner_text)` pairs from the string
2. Build `otsl_seq` as list of bare tag names (strip angle brackets): `["ched", "ched", "nl", "fcel", ...]`
3. Split sequence into rows on `"nl"` tokens
4. Walk grid to compute row/col indices and span detection:
   - `lcel` to the right → colspan
   - `ucel` below → rowspan
5. For each content-bearing cell (`fcel`, `ecel`, `ched`, `rhed`, `srow`), construct a `TableCell`:
   - `text` — inner text from the tag (empty string for `ecel`)
   - `start_row_offset_idx`, `end_row_offset_idx` — from row position + rowspan
   - `start_col_offset_idx`, `end_col_offset_idx` — from col position + colspan
   - `column_header` — `True` for `ched`
   - `row_header` — `True` for `rhed`
   - `row_section` — `True` for `srow`
   - `bbox` — `None` (VLM does not predict bounding boxes)

### Error handling

If `_parse_otsl_output()` raises, log a warning and emit an empty `Table` for that cluster (defensive pattern matching `_post_process` in chart extraction).

---

## Model Loading

Identical to `ChartExtractionModelGraniteVisionV4._load_model()`:

```python
self._processor = AutoProcessor.from_pretrained(artifacts_path, trust_remote_code=True)
self._model_max_length = self._processor.tokenizer.model_max_length
self._model = AutoModelForImageTextToText.from_pretrained(
    artifacts_path,
    device_map=self.device,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
cast(Any, self._model).merge_lora_adapters()
self._model.eval()
```

`download_models()` calls `download_hf_model(repo_id=..., revision=...)` — same utility used by all other models.

---

## Registration

### `pipeline_options.py`

Add `GraniteVisionTableStructureOptions` alongside `TableStructureOptions` and `TableStructureV2Options`.

### `defaults.py`

```python
def table_structure_engines():
    ...
    return {
        "table_structure_engines": [
            TableStructureModel,
            TableStructureModelV2,
            GraniteVisionTableStructureModel,  # new
        ]
    }
```

---

## User-Facing API

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    GraniteVisionTableStructureOptions,
)

pipeline_options = PdfPipelineOptions()
pipeline_options.table_structure_options = GraniteVisionTableStructureOptions()
```

---

## Out of Scope

- MPS support (excluded, same as existing chart extraction)
- Cell bounding box prediction (VLM does not produce them; `bbox=None`)
- PDF text cell matching / `do_cell_matching` (cell text comes from VLM output)
- VLM engine abstraction layer (`BaseVlmEngine`) — future refactor, separate task
- Batching across pages (batching is per-page, across tables within a page)
