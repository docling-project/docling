# Extraction service-client review: closed

Updated 2026-10-01. All seven original findings are implemented and committed through `1d3f63d2`:

1. Bounded per-source `extract_all`, yielding failed documents without ending the iterator.
2. Typed presigned/storage responses.
3. Early rejection of expandable sources by single-source `extract`.
4. ZIP rejection and dict-source coercion aligned with conversion.
5. Early `schema_constrained` validation requiring an output schema.
6. API parity with local extraction: positional `target`, operational options separately.
7. `max_file_size`, async request building off-loop, and polymorphic `VlmModelSpec` serialization.

C1's top-level `extraction_target` is implemented downstream in Jobkit `e75bb73` and Serve `367f3d2`. No downstream C1 work remains.

Published October 1 fixes preserve item-level failure reasons in raised SDK exceptions; see [the finalization assessment](extraction-api-finalization.md). Original review details remain in Git history.
