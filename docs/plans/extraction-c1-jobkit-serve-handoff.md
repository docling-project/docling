# Handoff: align docling-jobkit + docling-serve to the C1 extraction contract

**Status: closed, historical.** Nothing left to do here.

Drafted 2026-09-22. **Applied 2026-09-22:** jobkit `e75bb73`, serve `367f3d2`.
The list below is kept as the record of what changed. It aligns
`docling-jobkit` and `docling-serve` (both on branch `cau/extract-endpoint`) to
the C1 change already made in `docling` on `cau/extraction-api-service-models`.

## Why this is needed

C1 moved the extraction **contract** out of the operational options bag:

| | before | after |
|---|---|---|
| contract (what to extract) | `ExtractSourcesRequest.options.target: ExtractionTarget` | `ExtractSourcesRequest.extraction_target: ExtractionTarget` |
| operational config | `ExtractDocumentsOptions` (with required `target`) | `ExtractDocumentsOptions` (no `target`; all fields default) |

`ExtractDocumentsOptions` no longer has a `target` field (it is `extra="forbid"`,
so any code or payload that still sets `options.target` now **fails validation**),
and `ExtractSourcesRequest` now **requires** a top-level `extraction_target`.

Both repos read `options.target` today, so extraction is broken against the new
docling until the edits below land. The fix threads `extraction_target` from the
request, through the task and orchestrator, to the worker's `extract_documents`
call — the operational `extract_options` plumbing stays exactly as-is.

`resolve_extraction_model` does **not** need changing — it only reads the
operational fields (`extraction_preset`, `extraction_custom_config`,
`output_mode`, `input_channels`), never `target`.

## Prerequisite

`cau/extract-endpoint` in both repos must depend on a docling that includes C1
(the `cau/extraction-api-service-models` change). Bump the docling pin / use an
editable install of that branch before running the suites below, otherwise the
imports still resolve to the old model and nothing here is testable.

---

## docling-jobkit

### 1. `docling_jobkit/datamodel/task.py` — carry the contract on the task

`Task` currently has `extract_options` (~line 109) but no home for the contract.
Add a sibling field:

```python
extract_target: Optional[ExtractionTarget] = None
```

Import `ExtractionTarget` from `docling.datamodel.extraction`. Keep
`extract_options` as-is (still holds preset/output_mode/channel/page_range).

### 2. `docling_jobkit/orchestrators/base_orchestrator.py` — enqueue signature

`enqueue(...)` (abstract, ~line 108) lists `extract_options`. Add:

```python
extract_target: ExtractionTarget | None = None,
```

### 3. The three concrete orchestrators — accept + persist the field

Each mirrors the base signature and builds the task dict. In all three add the
`extract_target` parameter next to `extract_options`, and add
`"extract_target": extract_target,` to the `validate_task({...})` payload right
beside the existing `"extract_options": extract_options,`:

- `orchestrators/local/orchestrator.py` — signature ~line 76, task dict ~line 97.
- `orchestrators/ray/orchestrator.py` — signature ~line 645, task dict ~line 717.
- `orchestrators/rq/orchestrator.py` — signature ~line 171, task dict ~line 207.
  (RQ serializes the task to Redis; `ExtractionTarget` is a plain pydantic model,
  so it round-trips through `validate_task` like `extract_options` already does —
  no extra codec work.)

### 4. `docling_jobkit/orchestrators/ray/serve_deployment.py` — pass it to the worker

In the `ExtractPassthroughRequest` branch (~lines 807–843):

- After the existing `extract_options is None` guard, read the contract and guard
  it too:
  ```python
  extract_target = request.task.extract_target
  if extract_target is None:
      raise RuntimeError("Extraction task is missing extract_target.")
  ```
- In the `extract_documents(...)` call (~line 836) add the argument:
  ```python
  self._get_extraction_manager().extract_documents(
      sources=extract_sources,
      extraction_target=extract_target,
      options=extract_options,
      headers=headers,
  )
  ```

### 5. `docling_jobkit/convert/extraction_manager.py` — consume the contract

`extract_documents` (~line 159):

- Add a parameter: `extraction_target: ExtractionTarget` (import the type).
- Change the `extractor.extract_all(...)` call (~line 182) from
  `target=options.target` to `target=extraction_target`.

`resolve_extraction_model` — leave unchanged.

### 6. jobkit tests — `tests/test_extraction_manager.py`

Every `ExtractDocumentsOptions(target=_target(), ...)` must drop `target=` and the
contract must move to the call/enqueue site:

- Model-resolution cases (~lines 90, 135, 149, 157, 167, 183, 218, 270): remove
  `target=_target()` from `ExtractDocumentsOptions(...)`; where the test then needs
  a target it belongs on `extract_documents(..., extraction_target=_target())`.
- `extract_documents` calls (~lines 242–243, 261): pass
  `extraction_target=_target()` explicitly; the assertions on `call["target"]`
  (~lines 248–250) now compare against the passed `extraction_target`, not
  `options.target`.
- `enqueue(..., extract_options=ExtractDocumentsOptions(target=_target()))` cases
  (~lines 342, 387, 446, 556, 616, 678) and the fake `extract_documents(**kwargs)`
  (~line 646): move the contract to `enqueue(..., extract_target=_target(),
  extract_options=ExtractDocumentsOptions(...))` and update the fake to accept
  `extraction_target`.

---

## docling-serve

### 7. `docling_serve/policy.py` — validation

`validate_extract_request` (~line 672) calls
`prepare_target(request.options.target, resolved.model_spec)`. Change to:

```python
prepare_target(request.extraction_target, resolved.model_spec)
```

`resolve_extraction_model(request.options)` (~line 666) stays as-is.

`build_extract_request_model` (~line 264) needs **no change**: it subclasses
`ExtractSourcesRequest` and only overrides the `sources` and `target` fields, so
it inherits the new required `extraction_target` automatically. (Sanity-check the
generated OpenAPI shows `extraction_target` as required after the docling bump.)

### 8. `docling_serve/app.py` — enqueue

`_enqueue_extract` (~line 610) calls `orchestrator.enqueue(...)` (~line 642) with
`extract_options=request.options`. Add the contract alongside it:

```python
return await orchestrator.enqueue(
    task_type=TaskType.EXTRACT,
    sources=sources,
    extract_target=request.extraction_target,
    extract_options=request.options,
    targets=[target],
    callbacks=request.callbacks,
    metadata=task_metadata,
)
```

The endpoint handler `extract_url_async` (~line 1192) is unchanged — it already
takes the whole `ExtractSourcesRequestModel`, which now carries `extraction_target`.

### 9. serve tests + smoke scripts — request construction

Move `target=` out of `options=ExtractDocumentsOptions(...)` and up to
`extraction_target=` on the request in every construction site:

- `tests/test_service_policy.py` — six `ExtractSourcesRequest(...)` (~lines 592,
  610, 633, 658, 675, 699), each with an inner `options=ExtractDocumentsOptions(...)`.
- `tests/test_extraction_admission.py` — any `ExtractSourcesRequest` / payloads
  that set `options.target` (grep it).
- `scripts/smoke_extraction_lmstudio.py` (~line 64),
  `scripts/local_smoke/extraction_s3_to_s3_lmstudio.py` (~line 83),
  `scripts/local_smoke/extraction_matrix.py` (~line 192).

Also grep both repos for any remaining `options["target"]` / `.options.target` /
`"options": {"target"` in fixtures and JSON payloads.

---

## Suggested order

1. Bump docling pin to the C1 branch in both repos.
2. jobkit: task field → base signature → three orchestrators → serve_deployment →
   extraction_manager → jobkit tests. `pytest tests/test_extraction_manager.py`.
3. serve: policy → app → serve tests/scripts.
   `pytest tests/test_service_policy.py tests/test_extraction_admission.py`.
4. End-to-end: run one extraction smoke script against a live/LM-Studio endpoint
   to confirm the contract reaches the model unchanged.

## Done when

- No reference to `options.target` / `ExtractDocumentsOptions(target=...)` remains
  in either repo (grep clean).
- jobkit and serve extraction suites pass against the C1 docling.
- A source extraction round-trips: request `extraction_target` → task
  `extract_target` → `extract_documents(extraction_target=...)` → model, with
  operational `extract_options` still applied.
