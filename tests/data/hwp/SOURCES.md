# HWP/HWPX test fixture sources

These sample documents exercise the experimental HWP/HWPX backend end-to-end.

## Provenance & license

All files are copied verbatim from the **rhwp** parser project's sample corpus:

- Repository: <https://github.com/edwardkim/rhwp>
- License: **MIT** (Copyright (c) 2025-2026 Edward Kim) — redistribution and
  bundling in a downstream test suite is permitted.

| File | Upstream path | Exercises |
|------|---------------|-----------|
| `para-001.hwp` | `samples/para-001.hwp` | paragraphs, inline runs, mixed Korean/Hanja |
| `para-001.hwpx` | `samples/hwpx/para-001.hwpx` | HWP5↔HWPX equivalence pair (same content) |
| `table-001.hwp` | `samples/table-001.hwp` | a single table with header cells and column spans |
