# Handoff: extraction service-client review findings

Drafted 2026-09-22. This is a review of the extraction SDK on
`DoclingServiceClient` / `AsyncDoclingServiceClient` as built in commit
`19b9638f` from [extraction-service-client-api.md](extraction-service-client-api.md).
The question was whether it matches `convert` in design, validation, error
handling, and result shapes. Each finding below stands on its own so it can be
fixed in a separate session.

## Repos and branches

| Repo | Path | Branch | Relevant head |
|---|---|---|---|
| docling (this worktree) | `/Users/cau/Documents/Development/docling-second` | `cau/extraction-api-service-models` | `d10c1309` (on top of `19b9638f`) |
| docling-serve | `/Users/cau/Documents/Development/docling-serve` | `cau/extract-endpoint` | `8303ebb` |
| docling-jobkit | `/Users/cau/Documents/Development/docling-jobkit` | `cau/extract-endpoint` | `6d6a3ae` |

The C1 contract change (moving `options.target` to `extraction_target`) is
already done downstream: serve `367f3d2` reads `request.extraction_target`, and
jobkit `e75bb73` passes it through task → orchestrators → worker. The
"Downstream still required" section in
[extraction-service-client-api.md:265](extraction-service-client-api.md) is out
of date. See finding 7.

## Key code locations (docling)

- Sync client [client.py](../../docling/service_client/client.py):
  - `extract` :1221, `extract_all` :1250, `submit_extract` :1273
  - `_coerce_extract_sources` :360, `_source_to_extract_item` :374
  - `_as_extract_response` :403, `_single_extraction_document` :413,
    `_extraction_failure_message` :425
  - Convert counterparts to align with: `convert` :993, `convert_all` :1018,
    `submit` :1132, `_normalize_source` :629, `_preflight_limits` :1834,
    `_make_convert_fetch_result_handler` :1629, `_fetch_presigned_result` :1796,
    `_fetch_presigned_document_result` :1808, `_convert_all_async` :2369
- Async client [_async_client.py](../../docling/service_client/_async_client.py):
  `extract` :407, `extract_all` :432, `submit_extract` :454. Bounded fan-out for
  `convert_all` uses `_run_bounded` (:657; defined in
  [_scheduler.py:31](../../docling/service_client/_scheduler.py)).
- Wire models: `ExtractSourcesRequest`
  [requests.py:256](../../docling/datamodel/service/requests.py),
  `ExtractDocumentsOptions`
  [options.py:1206](../../docling/datamodel/service/options.py),
  `ExtractDocumentResponse` / `ExtractionDocumentResult`
  [responses.py:213,325](../../docling/datamodel/service/responses.py),
  `ExtractionTarget` [extraction.py:82](../../docling/datamodel/extraction.py).
- Tests: [tests/test_extraction_service_contract.py](../../tests/test_extraction_service_contract.py)
  (client tests from :490, built on `httpx.MockTransport` through `_result_transport`).

## Key code locations (downstream)

- docling-serve
  - `docling_serve/app.py:610` `_enqueue_extract`, `:1189` `POST /v1/extract/source/async`
  - `docling_serve/policy.py:603` `validate_extract_request` (source/target
    kinds, custom-config gate, allowed formats, callbacks,
    `max_sources_per_request`, presigned requires artifact storage)
  - `docling_serve/settings.py:150` `max_sources_per_request: int = 3` (default)
  - `docling_serve/response_preparation.py:38`: result type → response model
- docling-jobkit
  - `docling_jobkit/convert/extraction_results.py:236`: presigned →
    `PresignedArtifactResult`, storage → `RemoteTargetResult`, in-body →
    `ExtractionTaskResult`
  - `docling_jobkit/convert/extraction_manager.py:164` `extract_documents`

## What already matches `convert` (keep it)

- **Submission:** `submit_extract` uses `_request_with_retry` and
  `_raise_for_generic_http_error`, like `convert`.
- **Result fetch:** it goes through `_fetch_result_response`, so a 404 or expired
  result, a task failure, and a schema mismatch are all handled the same way as
  for `convert`.
- **Status check:** `raises_on_error` uses the same `SUCCESS_CONVERSION_STATUSES`
  set as `convert`, so `PARTIAL_SUCCESS` passes.
- **Secrets:** secret values are restored before submission.
- **C1:** `extraction_target` is the contract and `ExtractDocumentsOptions` is
  operational only.

## Findings

Ordered by severity. The suggested fix order is at the end.

### 1. `extract_all` sends every source as one job and breaks at the server default

**Status: done.**

**Problem.** `extract_all` passes all sources to a single `submit_extract`. Serve
rejects more than `max_sources_per_request` sources (default **3**) with a 422.
So `extract_all` on 4 local files fails on a default deployment. It also:

- base64-encodes every file into one JSON body held in memory,
- yields nothing until the whole batch is done (the iterator only looks like streaming),
- loses every result if the task fails, instead of just one.

`convert_all` runs one job per source with bounded concurrency
(`max_concurrency`, `_run_bounded`) and yields each result as it completes.

**Fix.**
- Run one `submit_extract` job per input source through `_run_bounded`.
- Add a `max_concurrency: int | None = None` argument, resolved with
  `_effective_concurrency`.
- Yield each job's documents as the job completes. A connector source still
  expands inside its own job, and all of its documents are yielded.
- A source whose job fails should come back as a failed
  `ExtractionDocumentResult` built with the right `source_index`, the way
  `convert_all` builds a failed `ConversionResult`. It should not end the
  iterator. Check how `_convert_all_async` handles this and copy it.
- Note that `source_index` on the server is per request (it is always 0 for
  single-source jobs). Rewrite it to the caller's input index, or document that
  callers should use `source_uri`.
- Mirror the change in the async client.

**Test.** Push more than 3 sources through `extract_all` against a mock transport
that returns 422 for more than 3 sources per request, and assert that every
result comes back. Also test that one failing job does not stop the others.

### 2. Storage and presigned results come back as raw bytes, though the server sends typed JSON

**Status: done.**

**Problem.** `submit_extract` returns `RawServiceResult` (bytes) for any target
other than in-body. For extract tasks, serve returns JSON models:
`PresignedUrlConvertResponse` for `PresignedUrlTarget`, and
`PresignedUrlConvertDocumentResponse` for S3/Azure/GCS/Drive. `submit` already
parses both of these for convert.

**Fix.**
- In `submit_extract`, dispatch on the target the way
  `_make_convert_fetch_result_handler` does:
  - in-body → `ExtractDocumentResponse`
  - `PresignedUrlTarget` → `_fetch_presigned_result`
  - storage target (`_is_storage_target`) → `_fetch_presigned_document_result`
- The return type becomes
  `ConversionJob[ExtractDocumentResponse | PresignedUrlConvertResponse | PresignedUrlConvertDocumentResponse]`.
- `_as_extract_response` stays as the type narrowing for `extract` /
  `extract_all`, since they always submit in-body.
- Update the async mirror, the skill reference
  (`docling/.agents/skills/docling/references/service-client.md`), and
  `test_sync_extraction_payload_and_result` / `test_async_extraction_payload_and_result`
  (they currently cover the storage case).

### 3. `extract()` accepts expandable sources and finds out after the job has run

**Status: done.**

**Problem.** `extract()` accepts `ExtractSourceRequestItem` (connectors included)
and, through `_coerce_extract_sources`, iterables too. The "expanded to N"
check (`_single_extraction_document`) runs after the job has finished. An S3
prefix with 500 documents is fully extracted, which costs money, and then
discarded with an error. `convert()` only accepts `SourceType`.

**Fix.**
- Restrict `extract()` to one non-expandable source: `SourceType`,
  `FileSourceRequest`, or `AnyHttpSourceRequest`.
- Raise `TypeError` / `ValueError` before submission for iterables and connector
  items, and point the message at `extract_all`.
- Keep the post-hoc length check as a defensive guard. (Correction: the original
  example, "a ZIP URL can still expand server-side", is wrong. Nothing unpacks
  ZIP input; see finding 4.)
- Update the type hints on both clients.

### 4. ZIP inputs are not rejected consistently, and dicts give confusing errors

**Status: done.** docling changes are uncommitted in this worktree; jobkit
`tests/test_connector_factory.py` is edited but uncommitted on `cau/extract-endpoint`.

**Corrected 2026-09-22.** The first version of this finding assumed extraction
supports ZIP URLs. It does not, and no endpoint does. The serve plans
(`plan_batch_convert_endpoint.md` §2, `plan_E_batch_endpoint.md`) assumed a ZIP
URL "can silently expand to many documents" and deliberately allowed ZIP URLs
on batch via `AnyHttpSourceRequest`. That expansion was never built: a local
check converting a ZIP of two PDFs with `DocumentConverter` gives one `SKIPPED`
result with format `None`. The docling tests
`test_any_http_source_request_allows_zip_urls` and
`test_batch_convert_sources_request_allows_zip_http_urls` encoded that plan and
are replaced.

**Facts.**
- Nothing in docling, jobkit, or serve unpacks a ZIP archive as input. jobkit's
  `HttpSourceProcessor` handles both `FileSourceRequest` and
  `AnyHttpSourceRequest` with `is_expandable() -> False`. It passes the URL or
  the file bytes straight to the converter. Docling's format detection only
  looks inside a ZIP for office containers (docx/xlsx/pptx/pages/epub), so a
  plain `bundle.zip` gets no format and fails in the worker after queueing
  (`tests/test_backend_dclx.py::test_dclx_not_guessed_without_dclx_extension`
  shows `archive.zip` → `None`).
- The only ZIP guard is `HttpSourceRequest.reject_zip_url` (convert, #3519).
  `AnyHttpSourceRequest` (batch convert, extract) and `FileSourceRequest` (all
  endpoints) have none. Serve's `validate_extract_request` allowed-formats check
  does not catch it either, because `.zip` is not in `FormatToExtensions`.
- Dict sources (`{"kind": "sharepoint", ...}`) come from #3841 (`4d825450`).
  They let `submit_batch` reach connectors that have no typed model in docling
  (jobkit's box, sharepoint, filenet, databricks_volumes, and plugins), through
  `GenericSourceRequest` and `Mapping` in `BatchSourceRequestInput`. The extract
  wire model (`ExtractSourceRequestItem`) accepts `GenericSourceRequest` the same
  way, and that stays. The extraction client never supported plain dicts. The
  errors below are just what happens when a dict falls through the code.

Reproduced with `_coerce_extract_sources` (before this fix):

| Input | Result |
|---|---|
| `"https://example.com/bundle.zip"` | `FileNotFoundError: 'https:/example.com/bundle.zip'` (convert gets the same) |
| `{"kind": "s3", ...}` (single dict) | `FileNotFoundError: 'kind'` (the dict is iterated as its keys) |
| `[{"kind": "s3", ...}]` | `TypeError: Unsupported extraction source` |

**Fix.**
- **ZIP guard in the shared models (decided).** Move `reject_zip_url` from
  `HttpSourceRequest` to `AnyHttpSourceRequest`, and add a matching `.zip`
  filename check on `FileSourceRequest`. Check the extension, not the contents,
  because office formats are ZIPs by magic bytes. `HttpSourceRequest` then only
  keeps its docstring. Use one message for both, for example "ZIP archives are
  not accepted as input sources". Serve then returns 422 at submission, and the
  client fails before sending.
- **`_normalize_source`.** Only fall back to `Path` when the string has no
  `http://` / `https://` scheme. For an http(s) string, let the
  `ValidationError` (ZIP or otherwise) propagate. This fixes row 1 for convert
  and extract.
- **Dicts (decided: same as `submit_batch`).** `submit_extract` (and through it
  `extract_all`) accepts dict sources. `_coerce_extract_sources` treats a
  `Mapping` as one source, `_source_to_extract_item` passes it through, and
  `ExtractSourcesRequest` validation (`_validate_batch_source`) coerces it by
  `kind`. The new alias `ExtractSourceRequestInput`
  (`ExtractSourceRequestItem | Mapping[str, Any]`) mirrors
  `BatchSourceRequestInput`. `extract()` still takes only one non-expandable
  source (finding 3).
- Replace the stale comment at client.py:433 ("A ZIP URL can still expand
  server-side…"). The post-hoc check is now only a defensive guard.
- **Downstream.** Update jobkit
  `tests/test_connector_factory.py::test_registry_validates_filenet_and_http_canonical_models`.
  It validates `{"kind": "http", "url": ".../archive.zip"}` as
  `AnyHttpSourceRequest`, which will now raise, so change the URL to a non-ZIP
  one. Serve and jobkit pick the guard up after relocking the docling pin.
- **Tests.** One per table row, plus `FileSourceRequest(filename="x.zip")` and
  `AnyHttpSourceRequest(url=".../x.zip")` rejected in the model tests. Check that
  `x.docx` still validates.

### 5. `schema_constrained` without `output_schema` fails only in the worker

**Status: done.** Validator `_schema_constrained_needs_schema` on
`ExtractSourcesRequest`; test in `tests/test_service_datamodels.py`.

**Problem.** `ExtractDocumentsOptions(output_mode="schema_constrained")` with an
`ExtractionTarget` that has only a template is accepted by the client and by
serve. It then fails at execution time
([prompt_utils.py:173](../../docling/models/extraction/prompt_utils.py)), after
queueing.

**Fix.** Add a `model_validator(mode="after")` on `ExtractSourcesRequest`. It
should raise if `options.output_mode == "schema_constrained"` and
`extraction_target.output_schema is None`. This is possible now that C1 put both
on the same model. The client then raises `ValueError`, and serve returns a 422
with no serve change (serve uses the same model). Add one test. The engine-type
part of the check (vLLM API only) depends on the preset the server resolves, so
it stays server-side.

### 6. Parity with local extraction: decide, then document

**Status: done. Decided 2026-09-22 (Christoph), overriding the recommendation
below:**
- `extract()` / `extract_all()` return the local `DocumentExtractionResult`,
  built with `_build_input_document` the way `convert()` builds
  `ConversionResult` (filename + format guessed from it). The two models carry
  the same payload (status, errors, items) and differ only in identity
  (`InputDocument` vs `source_index` / `source_uri` / `filename`), so returning
  the wire type made two near-identical names public. `source_index` /
  `source_uri` are dropped there; `extract_all` results match by
  `input.file.name`, as `convert_all`. `submit_extract` keeps the wire
  `ExtractDocumentResponse`.
- `extract()` / `extract_all()` take the contract as `target=`, like the local
  extractor. `submit_extract` keeps the wire names (`extraction_target=` contract,
  `target=` destination), as `submit(target=)` and `ExtractSourcesRequest`.
- Top-level `page_range=` on `extract()` / `extract_all()` overrides
  `options.page_range` (jobkit honors it). `max_num_pages` / `max_file_size` are
  not added: they would need new `ExtractDocumentsOptions` fields; the server
  applies its own `max_num_pages` / `max_file_size`.

Original analysis:

`convert()` returns the local `ConversionResult`, so the service client can
replace the local converter. `extract()` differs from local
`DocumentExtractor.extract()` in three ways:

- It returns the wire model `ExtractionDocumentResult`, where local returns
  `DocumentExtractionResult` (which has an `InputDocument`).
- The contract argument is `extraction_target`, where local uses `target=`.
- It has no top-level `page_range` / `max_num_pages` / `max_file_size`.
  `page_range` exists on `ExtractDocumentsOptions`.

**Recommendation: keep it as it is.**
- `source_index` and `source_uri` are what make `extract_all` results traceable
  to their source. `DocumentExtractionResult`'s `InputDocument` would be made up.
- Using `target=` would clash with `submit_extract(target=)`, which means the
  destination, as it does in `submit`.

If kept, state in the skill reference and the `extract.py` example that this is
not a drop-in replacement for the local extractor. **This needs a decision from
Christoph before any change.**

### 7. Minor

**Status: done.** `max_file_size=` on `extract` / `extract_all` (SKIPPED before
the read, `_preflight_extract_size`); async `submit_extract` builds the request
in `asyncio.to_thread`; shared `_build_extract_request` and
`_extract_submission_status` (which logs the submission, source count only);
both plans updated. Side catch:
`test_polymorphic_option_fields_are_serialized_as_any` broke in `d4083dec`,
which made `ExtractionVlmModelSpec` a subclass of `VlmModelSpec`; the three
`model_spec: VlmModelSpec` fields in `pipeline_options.py` are now
`SerializeAsAny`.

- **No file-size check.** There is no size check before the whole-file base64
  read in `_source_to_extract_item`. `convert` has `max_file_size` plus
  `_preflight_limits`. Consider an optional `max_file_size` on `extract` /
  `extract_all` that fails before reading. Serve has its own `max_file_size`
  setting.
- **Blocking read in async.** The async path calls the shared
  `_source_to_extract_item`, which does a blocking `Path.read_bytes()`. Wrap it
  in `asyncio.to_thread`, or accept it and mark it with a `ponytail:` comment.
- **Duplicated request building.** The `ExtractSourcesRequest(...)` construction
  is copied in the sync and async `submit_extract`. Move it into a base
  `_build_extract_request(...)`.
- **No logging.** There is no submit log line; `_submit_convert_task` logs
  source and task id at info level.
- **Stale plan.** Update the "Downstream still required" section of
  `extraction-service-client-api.md` (and check
  `extraction-c1-jobkit-serve-handoff.md`) to record that serve `367f3d2` and
  jobkit `e75bb73` made the change.

## Suggested order

1. **Finding 4:** the shared-model ZIP guard, the `_normalize_source` fix (shared
   with convert), and a clear rejection of dicts.
2. **Finding 5:** the validator (one model change, and serve benefits too).
3. **Finding 3:** restrict `extract()` sources.
4. **Finding 2:** typed results for storage and presigned targets.
5. **Finding 1:** per-source `extract_all` with bounded concurrency. This is the
   largest change, and it builds on 3 and 4.
6. **Finding 6:** decide, then update docs.
7. **Finding 7:** minor cleanups.

Findings 4 (ZIP guard on `AnyHttpSourceRequest` / `FileSourceRequest`) and 5
(validator on `ExtractSourcesRequest`) change shared models. No other finding
changes the wire contract. Serve picks it up after the docling pin is relocked on
`cau/extract-endpoint`.

## Conventions for the follow-up session

- Follow `AGENTS.md`: run `make validate` before finishing, run the affected tests
  (`uv run pytest tests/test_extraction_service_contract.py tests/test_service_datamodels.py`),
  and keep the usage skill (`docling/.agents/skills/docling/references/`) and
  `docs/examples/service_client/extract.py` in sync with user-facing changes.
- Keep sync and async clients mirrored.
- Commit as Christoph Auer with his own `Signed-off-by`, and no Claude
  attribution lines. Commit only when asked.
