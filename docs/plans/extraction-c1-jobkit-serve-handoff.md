# C1 downstream contract migration: closed

Applied 2026-09-22 in Jobkit `e75bb73` and Serve `367f3d2`. `ExtractSourcesRequest.extraction_target` is forwarded through task → orchestrator → worker → `DocumentExtractor`; options contain operational settings only. The outer `target` selects result storage.

No work remains under this handoff. See [the current ledger](extraction-additional-vlm-models-execution.md).
