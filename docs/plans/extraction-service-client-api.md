# Extraction in the Docling Service Client: design + low-level critique

**Status: closed, historical (2026-09-23).** Everything below is implemented: C1–C3 in `19b9638f`, then the review fixes in `809fb75d`..`1d3f63d2` (see [the review handoff](extraction-service-client-review-handoff.md)), downstream in jobkit `e75bb73` and serve `367f3d2`. Some details changed after this proposal: `extract()`/`extract_all()` take the contract as `target=` and return the local `DocumentExtractionResult`, and `extract_all` runs one job per source. The code and the shipped `references/service-client.md` are authoritative, not this document.

Drafted 2026-09-22. Scope: give `DoclingServiceClient` /
`AsyncDoclingServiceClient` a user-facing extraction API that aligns with how we
work with conversion, and — per the request — challenge whether the already-built
low-level plumbing (`submit_extract`, `ExtractSourcesRequest`,
`ExtractDocumentsOptions`, `ExtractDocumentResponse`) is designed the right way.

Nothing here is committed. This is a proposal to react to before code.

## What already exists

- `submit_extract(request: ExtractSourcesRequest)` on both
  [client.py:1127](../../docling/service_client/client.py) and
  [_async_client.py:399](../../docling/service_client/_async_client.py). Returns a
  `ConversionJob[ExtractDocumentResponse | RawServiceResult]` with the same
  `.result()/.watch()/.poll()` surface as convert/chunk jobs.
- Serve endpoint: **`POST /v1/extract/source/async` only**. There is deliberately
  no sync `/v1/extract/source` and no multipart `/v1/extract/file`
  (`docling-serve` `test_service_policy.py` asserts their absence). Local files
  ride inline as base64 `FileSourceRequest` in the JSON body.
- Data models: `ExtractSourcesRequest`, `ExtractDocumentsOptions`,
  `ExtractDocumentResponse`, `ExtractionDocumentResult`, `ExtractionTaskResult`.

**The gap:** no high-level `extract()` / `extract_all()` (the analog of
`convert()` / `convert_all()`), no source ergonomics (a local
`Path`/`DocumentStream`/URL → a source item), and none of the extraction models
are exported from `service_client/__init__.py`.

## Answers to the three questions (established from the code, not opinion)

### 1. Why "service-only knobs" on extraction, but convert doesn't feel this?

They are **not service-invented**. `ExtractDocumentsOptions` fields map straight
onto the real model config `ExtractionVlmOptions`
([extraction_options.py](../../docling/datamodel/extraction_options.py)):

- `extraction_preset` / `extraction_custom_config` → *which model* (operator
  allow-listed; resolved by `DocumentExtractionManager.resolve_extraction_model`
  in docling-jobkit).
- `output_mode` → `ExtractionVlmOptions.output_mode` (see Q2).
- `input_channels` → `VlmExtractionPipelineOptions.input_channels`.
- `page_range` → limits.

Locally you set all of these by *constructing* a
`DocumentExtractor(extraction_format_options=...)`. On the service you cannot hand
the server a model object, so these become a **narrow per-call window** into
operator-gated model config. Convert has exactly the same shape (operator
configures the pipeline; `ConvertDocumentsRequestOptions` is the caller's window).
The only reason extraction *feels* different is not the knobs — it is that the
**semantic contract (`target` = what to extract) is buried inside the same knobs
bag.** That fusion is the real smell, and it is what makes "pass both `target=`
and `options` (with its own target)" even expressible. Fix addressed in the
low-level critique below.

### 2. What is `output_mode` for?

A genuine generation-control knob, not a service concept
(`prepare_output_target` in
[prompt_utils.py:162](../../docling/models/extraction/prompt_utils.py)):

- `prompt_only` (default): the schema/template is injected as *guidance*; the
  model emits free-form JSON that is parsed (and optionally validated) afterward.
- `schema_constrained`: turns on **constrained/guided decoding** against
  `output_schema`. Only valid on the vLLM **API** engine and only when an
  `output_schema` is present; otherwise it raises. It changes the generation
  contract, so it legitimately belongs in the operational options — but it is
  operational, *not* part of the extraction contract.

### 3. Why would a single `extract()` return a multi-document type? Where is `ExtractDocumentResponse` used?

`ExtractDocumentResponse` is used in exactly one place: the return of the
low-level `submit_extract` fetch path, plus one contract test
(`test_extraction_service_contract.py`). It is the **task envelope**
(`num_succeeded`/… counts + `documents: list[ExtractionDocumentResult]`).

Fan-out to >1 document happens **only for connector sources** (an S3 prefix, a
Google Drive folder). A `FileSource` or `HttpSource` yields exactly one document.
So a single-source convenience returning a multi-doc envelope is indeed wrong for
the common case. See the return-shape decision below.

## Low-level critique (challenge the existing design)

The endpoint is unreleased WIP, so changing the request contract now is cheap and
correct. Three concrete issues:

### C1 — `ExtractDocumentsOptions` fuses the contract with operational config

`options.target: ExtractionTarget` (the *what*) sits next to `output_mode` /
`extraction_preset` / `input_channels` (the *how/which model*). This is what
forces the ugly composition and lets a caller specify the contract in two places.

**Recommendation:** hoist the extraction contract to a top-level field of
`ExtractSourcesRequest`, sibling of `sources`/`target`/`options`, and make
`options` purely operational.

```python
class ExtractSourcesRequest(BaseModel):
    extraction_target: ExtractionTarget          # the "what" — was options.target
    sources: list[ExtractSourceRequestItem]
    options: ExtractDocumentsOptions = ExtractDocumentsOptions()  # now purely operational, all-defaulted
    target: ExtractTargetRequest = InBodyTarget()  # the destination
    callbacks: list[CallbackSpec] = []
```

`ExtractDocumentsOptions` loses `target`; everything else stays and becomes
optional with server defaults. One place for the contract, no ambiguity, and
`options` can be omitted entirely for the common case.

- Naming: request-level `target` already means *destination* (consistent with
  convert). The extraction contract is therefore `extraction_target` to avoid the
  clash. (Bikeshed: `schema`/`extraction`/`contract` — `extraction_target` keeps
  the existing `ExtractionTarget` vocabulary. Open to a rename.)
- Cost: touches the serve request model + `resolve_extraction_model`
  (reads `options.target` today) + the contract test + smoke scripts. All in the
  active branch series; no released consumer.

### C2 — `submit_extract` is inconsistent with its siblings

`submit()` and `submit_batch()` take **unpacked** friendly args
(`source`/`sources`, `target`, `options`, `headers`). `submit_extract` is the only
low-level submitter that demands a **prebuilt `ExtractSourcesRequest`**. That
forces callers to hand-assemble base64 `FileSourceRequest`s and the options
wrapper.

**Recommendation:** give `submit_extract` the same unpacked shape:

```python
def submit_extract(
    self,
    source: SourceType | Iterable[SourceType] | Sequence[ExtractSourceRequestItem],
    extraction_target: ExtractionTarget | None = None,
    template: ExtractionTemplateType | None = None,
    options: ExtractDocumentsOptions | None = None,
    target: ExtractTargetRequest | None = None,   # destination; default InBody
    headers: dict[str, str] | None = None,
    callbacks: list[CallbackSpec] | None = None,
) -> ConversionJob[ExtractDocumentResponse | RawServiceResult]: ...
```

It builds the `ExtractSourcesRequest` internally (source→item mapping, contract
normalization via `normalize_extraction_call`). This is the escape hatch for
storage targets, callbacks, and multi-source batches — parity with `submit_batch`.

### C3 — Return shape of the high-level convenience

`ExtractDocumentResponse` (the counts envelope) is the right return for a *job*
(`submit_extract`), but wrong for a single-source convenience. Per Q3, only
connectors fan out.

**Recommendation:**
- `extract(source, ...)` → a single `ExtractionDocumentResult` (mirrors local
  `DocumentExtractor.extract()`). If the response carries ≠1 document (a connector
  source was passed), raise `ExtractionError` pointing at `extract_all`.
- `extract_all(sources, ...)` → `Iterator[ExtractionDocumentResult]`, flattening
  connector fan-out (mirrors `convert_all`).
- The counts envelope stays reachable via `submit_extract(...).result()`.

## Proposed high-level API

Mirrors the **local** `DocumentExtractor.extract()` (not `convert()`), because the
extraction contract is `ExtractionTarget` and the high-level form has no
destination arg (same as `convert()` has none). Swapping a local extractor for the
service client becomes near drop-in. Reuse `normalize_extraction_call(template,
target)` so `template=` deprecation + "exactly one of target=/template=" behave
identically to the local path.

```python
def extract(
    self,
    source: SourceType,
    template: ExtractionTemplateType | None = None,
    headers: dict[str, str] | None = None,
    raises_on_error: bool = True,
    max_num_pages: int | None = None,
    max_file_size: int | None = None,
    page_range: PageRange | None = None,
    *,
    target: ExtractionTarget | None = None,        # the contract (what to extract)
    options: ExtractDocumentsOptions | None = None,  # operational only (post-C1)
) -> ExtractionDocumentResult: ...

def extract_all(
    self,
    source: Iterable[SourceType],
    template: ExtractionTemplateType | None = None,
    ...same knobs...,
    *,
    target: ExtractionTarget | None = None,
    options: ExtractDocumentsOptions | None = None,
) -> Iterator[ExtractionDocumentResult]: ...
```

- With C1 done, there is no `target`/`options.target` collision to guard: `target`
  (or `template`) is the *only* place for the contract; `options` is operational.
- Internally: `normalize_extraction_call` → build request with `InBodyTarget` →
  `submit_extract(...).result(timeout=self._job_timeout)` → return document(s).
- `raises_on_error=True` raises `ExtractionError` when a document status is a
  failure — same pattern as `convert()` raising `ConversionError`.
- Async mirrors on `AsyncDoclingServiceClient`.

### Source → item mapping (new helper)

`_source_to_extract_item(source)`: `Path`/`DocumentStream` →
`FileSourceRequest(base64_string=..., filename=...)`; URL/`HttpSourceRequest` →
`AnyHttpSourceRequest`; connector request models passed through. Known ceiling:

```python
# ponytail: whole-file base64 into memory — the extract endpoint is source/JSON
# only, no multipart streaming exists server-side. Fine for typical docs; revisit
# if large-file extraction becomes a use case.
```

## Work plan

Ordered so the contract change lands before the ergonomic layer builds on it.

1. **C1 (contract hoist).** Move `target` → `extraction_target` on
   `ExtractSourcesRequest`; strip `target` from `ExtractDocumentsOptions` (all
   remaining fields optional). Update `resolve_extraction_model` (docling-jobkit),
   serve request model, contract test, smoke scripts. *Decision gate — this is a
   wire change; confirm before doing it.*
2. **C2 (unpack `submit_extract`).** Friendly args + internal request assembly +
   `_source_to_extract_item`. Async mirror.
3. **C3 + high-level.** `extract()` / `extract_all()` on both clients;
   `ExtractionError` in `exceptions.py`.
4. **Exports.** Add to `service_client/__init__.py`: `ExtractSourcesRequest`,
   `ExtractDocumentResponse`, `ExtractionDocumentResult`, `ExtractDocumentsOptions`,
   `ExtractionTarget`, `ExtractionTemplate`, `ExtractTargetRequest`,
   `ExtractionError`.
5. **Docs/examples.** `docs/examples/service_client/extract.py` mirroring
   `convert.py`; update the usage skill
   [`docling/.agents/skills/docling/`](../../docling/.agents/skills/docling/SKILL.md)
   per the AGENTS.md "keep in sync" rule.
6. **Tests.** Extend `test_service_client_fake_service.py` (single-doc return,
   connector fan-out → `extract_all`, `raises_on_error`, base64 file source,
   `template=` vs `target=`, operational-knob passthrough) and
   `test_service_client_payload_fidelity.py` (assembled request payload, incl. the
   new `extraction_target` placement).
7. `make validate` + targeted tests.

## Resolution (2026-09-22, implemented in docling-second)

Decided and built on `cau/extraction-api-service-models`:

- **C1 done.** `ExtractSourcesRequest.extraction_target` is the contract;
  `ExtractDocumentsOptions` is operational-only and fully defaulted (`target`
  removed). Field named `extraction_target`; destination stays `target`.
- **C2 done.** `submit_extract` now takes unpacked args
  (`source, extraction_target, options=, target=, headers=, callbacks=`) and
  builds the request internally — parity with `submit`/`submit_batch`.
- **C3 done.** `extract()` → one `ExtractionDocumentResult` (raises
  `ExtractionError` on connector fan-out or, with `raises_on_error`, on failure);
  `extract_all()` → iterator; both in-body. Async mirrors added.
- **`template=` dropped from the service client.** The local `template=`
  shorthand serializes against a specific model spec (NuExtract) client-side,
  which is wrong when the server picks the model by preset. The service API takes
  an explicit `ExtractionTarget` only (the non-deprecated local path), which also
  keeps the client free of `docling.models.*` imports.
- Source→item mapping added: local `Path`/`DocumentStream` → base64
  `FileSourceRequest` (endpoint is source/JSON only); URL → `AnyHttpSourceRequest`;
  prebuilt connector items pass through. Exports + skill docs + `extract.py`
  example updated; tests added and green.

**Downstream (done 2026-09-22).** serve `367f3d2` reads
`request.extraction_target`, and jobkit `e75bb73` threads it through task →
orchestrators → worker. C1 changed the wire model that `docling-jobkit` and
`docling-serve` (both on `cau/extract-endpoint`) consumed via `options.target`.
The change touched:

- `docling-serve`: `policy.py` (`request.options.target` → `request.extraction_target`),
  `app.py` `_enqueue_extract` (pass `extract_target=request.extraction_target`).
- `docling-jobkit`: `datamodel/task.py` (`extract_target` field), the four
  orchestrators' `enqueue` (carry it in the task payload),
  `ray/serve_deployment.py` (read it), and `convert/extraction_manager.py`
  (`extract_documents(..., extraction_target=...)`; `resolve_extraction_model`
  needs no change — it never read `target`).

They ship on `cau/extract-endpoint` alongside the docling pin bump.

## Open decisions (all resolved 2026-09-22)

Resolved: C1 yes, field name `extraction_target`, C2 unpacked. Kept for the record:

- **C1 wire change: yes/no.** Cleanest fix for the fusion + double-target problem,
  cheap now (unreleased), but it is a contract edit spanning serve + jobkit. If
  "no", fall back to: keep `target` inside `options`, and in the high-level API
  forbid passing both `target=` and an `options` whose `.target` is set (raise).
- **Name for the hoisted contract field** (`extraction_target` vs `schema`/
  `extraction`).
- **Should `submit_extract` be unpacked (C2)** or left as the one prebuilt-request
  escape hatch?
