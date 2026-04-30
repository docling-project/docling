# Docling

This file provides guidance to AI coding agents when working with code in this
repository.

## Project overview

Docling is a Python SDK and CLI for converting PDFs, Office files, HTML,
Markdown, audio, images, XML, and other formats into a unified
`DoclingDocument` representation for downstream AI workflows.

## Project structure

```text
docling/                 # main Python package
packages/docling/        # full docling meta-package
packages/docling-slim/   # slim package readme
tests/                   # pytest suite and test data
docs/                    # MkDocs documentation and examples
scripts/                 # project maintenance scripts
```

## Key commands

```bash
just setup          # install CI-style dev environment
just test           # run pytest
just check          # run read-only local checks
just validate       # run mutating hooks on the current changeset
```

## Code standards

- Keep public APIs typed and compatible with Python 3.10+.
- Use `uv add` or project-local dependency patterns when dependencies change.
- Add focused tests for behavior changes; regenerate reference data only when
  conversion outputs intentionally change.

## When making changes

1. Keep edits scoped and consistent with the surrounding module.
2. Update docs/examples when user-facing behavior changes.
3. Run targeted tests for touched behavior.
4. For reference output changes, use `DOCLING_GEN_TEST_DATA=1 uv run pytest`
   and review generated data carefully.

## Before finishing

Run `just validate` before considering a task complete. If hooks modify files,
review the changes and rerun `just validate` until it passes cleanly. Also run
the affected tests for the files or behavior you changed. Use `just check` when
you need a read-only verification pass.
