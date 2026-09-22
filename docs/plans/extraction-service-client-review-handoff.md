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
- Keep the post-hoc length check as a guard (for example, a ZIP URL can still
  expand server-side).
- Update the type hints on both clients.

### 4. Source conversion bugs

Reproduced with `_coerce_extract_sources`:

| Input | Result |
|---|---|
| `"https://example.com/bundle.zip"` | `FileNotFoundError: 'https:/example.com/bundle.zip'` |
| `{"kind": "s3", ...}` (single dict) | `FileNotFoundError: 'kind'` (the dict is iterated as its keys) |
| `[{"kind": "s3", ...}]` | `TypeError: Unsupported extraction source` |

**Root causes.**
- **ZIP URL:** `_normalize_source` (client.py:629) builds an `HttpSourceRequest`,
  whose validator rejects `.zip` URLs. The `ValidationError` is caught and the
  string falls through to `Path`. Extraction accepts ZIP URLs through
  `AnyHttpSourceRequest`, so a supported input is blocked. `convert` gets the
  same misleading error message.
- **Dicts:** `_coerce_extract_sources` treats any `Iterable` as a list of
  sources, and `_source_to_extract_item` has no `Mapping` branch. `submit_batch`
  accepts `Mapping` sources (`BatchSourceRequestInput`).

**Fix.**
- In `_normalize_source`, only fall back to `Path` when the string is not an
  http(s) URL. If it is one, let the ZIP rejection (or any other URL validation
  error) propagate for convert.
- Give extraction its own URL path that builds `AnyHttpSourceRequest` directly
  from the string, so ZIP URLs work there.
- In `_coerce_extract_sources`, treat `Mapping` as a single item. In
  `_source_to_extract_item`, pass `Mapping` through and let
  `ExtractSourcesRequest` validation (`_validate_batch_source`) coerce it.
- Add a regression test for each row of the table.

### 5. `schema_constrained` without `output_schema` fails only in the worker

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

1. **Finding 4:** the `_normalize_source` fix (shared with convert, small, removes
   a misleading error everywhere), plus `Mapping` handling.
2. **Finding 5:** the validator (one model change, and serve benefits too).
3. **Finding 3:** restrict `extract()` sources.
4. **Finding 2:** typed results for storage and presigned targets.
5. **Finding 1:** per-source `extract_all` with bounded concurrency. This is the
   largest change, and it builds on 3 and 4.
6. **Finding 6:** decide, then update docs.
7. **Finding 7:** minor cleanups.

No wire change is needed except finding 5, which adds a validator on a shared
model. Serve picks it up after the docling pin is relocked on
`cau/extract-endpoint`.

## Conventions for the follow-up session

- Follow `AGENTS.md`: run `make validate` before finishing, run the affected tests
  (`uv run pytest tests/test_extraction_service_contract.py tests/test_service_datamodels.py`),
  and keep the usage skill (`docling/.agents/skills/docling/references/`) and
  `docs/examples/service_client/extract.py` in sync with user-facing changes.
- Keep sync and async clients mirrored.
- Commit as Christoph Auer with his own `Signed-off-by`, and no Claude
  attribution lines. Commit only when asked.
