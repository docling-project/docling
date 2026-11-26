# Extraction service-client contract

Implemented. Updated 2026-10-01. This replaces the completed design/resumption instructions; history remains in Git.

```python
client.extract(source, target, options=None, raises_on_error=True)
client.extract_all(sources, target, options=None, max_concurrency=None)
client.submit_extract(source, extraction_target, options=None, target=None, callbacks=None)
```

`extract` and `extract_all` return local `DocumentExtractionResult` envelopes and use in-body output. Single-source `extract` rejects connector sources that can expand into multiple documents. `extract_all` submits one job per source with bounded concurrency and yields failed documents rather than aborting the iterator. Sync and async APIs share validation and result helpers.

`submit_extract` supports storage targets. `extraction_target` carries output schema, tagged model template and instructions; `options` carries operational configuration; outer `target` chooses output storage. Presigned/direct storage responses are typed. This top-level wire contract is implemented in all three PRs.

Schema validation, ZIP rejection, dict-source coercion and maximum local file size are validated before submission. Item errors are retained in results and included in raised extraction failure messages. Task/job errors remain distinct from per-document/item failures.

For the final delivery checklist, see [the assessment](extraction-api-finalization.md). For runnable examples, see [the service-client example](../examples/service_client/extract.py).
