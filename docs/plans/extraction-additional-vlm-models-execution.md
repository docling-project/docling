# Extraction implementation ledger

Updated 2026-10-01. Implementation and stage 11 are complete. This is the current cross-repository ledger; old resumption prompts are historical. Final delivery work is listed in [the finalization assessment](extraction-api-finalization.md).

## Current contract

- `ExtractSourcesRequest.extraction_target`: schema, tagged template and instructions.
- `options`: preset/custom configuration, output mode, channels and page range.
- Outer `target`: in-body, presigned or direct artifact storage.
- Sync/async `extract(source, target, ...)`, bounded per-source `extract_all`, and `submit_extract(source, extraction_target, options=..., target=...)` are implemented.
- Durable documents contain source identity, status, document errors and ordered items. Items contain page/document scope, raw text, extracted data, validation status, structured errors and inference metadata.
- Serve execution is Ray-only. Local/RQ return 501. Expandable sources require direct artifact storage; S3→in-body/presigned return 422.
- Operator-defined presets are independent of client permission to supply custom configuration.
- Published Core is sufficient. Jobkit/Serve still pin unreleased upstream Git branches until package publication.

## Completion record

| Stage | Scope | State / checkpoint |
|---|---|---|
| 0–1 | Target/content preparation | Complete; `0e53ddc` |
| 2 | Unified inference adapters | Complete; `b0e888f` |
| 3 | Local extractor target SDK and durable items | Complete; `cd55041` |
| 4 | NuExtract3 | Complete; `ef5a8a9`; API route verified with GGUF through llama-server |
| 5 | Lift | Implementation and offline contracts complete; `6f7beea`; live run remains optional follow-up |
| 6–7 | Qwen3.5/Gemma | Deferred by the user |
| 8 | Docling service/client contract | Complete; `c2d5347`, followed by C1/client review fixes through `1d3f63d2` |
| 9 | Jobkit forwarding, identity, storage, callbacks | Complete; `7f641bf`, followed by contract/preset/storage fixes |
| 10 | Serve admission/endpoint | Complete; `00cd751c`, followed by C1/preset/source-target fixes |
| 11 | Gap closure and final matrix | Completed 2026-09-23; plan updates committed in `804c4014`; smoke runners committed and pushed on 2026-10-01 |

PR #4201's head is an ancestor of #4218. Its API-engine/multi-format/channel work is already included. #4218 now targets main directly; #4201 can be marked superseded.

`NuExtractTransformersModel` removal is accepted: it was not a public-facing import. No compatibility shim or further decision is required.

## Verified behavior

Historical live runs on September 23 covered Granite through LM Studio and NuExtract3 GGUF through llama-server: PDF/DOCX source-target matrices, Markdown/HTML, multi-document S3 prefixes, encrypted input, page range 2–3, forced schema failure and artifact availability before document callbacks. The GGUF run verifies the NuExtract3 API route. NuExtract3's named LM Studio engine cannot deliver the required per-call template kwargs; generic API through llama-server works.

Callbacks use the shared lifecycle events. Delivery uses independent threads, so arrival order is not guaranteed. The smoke checks event counts and artifact availability rather than assuming arrival order.

The September 24 transport-debug fixes landed in Docling `7f237215`, Jobkit `af845006` and Serve `b85ead7`. Published October 1 fixes also gate raw HTTP error bodies, preserve scoped item reasons in SDK exceptions/callbacks, and classify wrapped read timeouts safely. Branches and downstream locks are pushed; main conflicts and Docling DCO are resolved. Include these fixes in releases/deployment.

## Tests and remaining work

Run the affected committed tests with a compatible environment; use each repository's own environment when possible. No custom Core or sibling source override is required for installed released dependencies. The Docling checkout's Python 3.14 environment currently aborts on optional MLX import; the October 1 source checks used Jobkit's Python 3.12 environment.

- Docling: extraction suites, service contract/client suites, API request and KServe HTTP tests.
- Jobkit: extraction manager and presigned result tests.
- Serve: extraction admission, environment parsing and smoke assertion tests.

Historical stage 11 broad runs also had unrelated parser/cv2, MinIO bucket and async-fixture/config/OTEL failures; retain that distinction when describing coverage. Original temporary live logs are no longer available. Tests and fresh finalization results are summarized in the assessment.

Publish Docling→Jobkit→Serve and replace branch sources with release floors. Rerun the formerly timing-out SaaS request at its new 60-second server timeout and then the corrected 147-case benchmark. The harness lives in the standalone [docling-extractbench repository](https://github.ibm.com/docling-project/docling-extractbench/), which follows Docling's extraction branch and uses ExtractBench as a pinned dependency. Extraction billing work units/operation identification and metrics remain a separate scoped follow-up.
