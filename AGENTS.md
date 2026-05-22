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
make setup          # install CI-style dev environment
make test           # run pytest
make check          # run read-only local checks
make validate       # run mutating hooks on the current changeset
```

## Code standards

- Keep public APIs typed and compatible with Python 3.10+.
- Use `uv add` or project-local dependency patterns when dependencies change.
- Add focused tests for behavior changes; regenerate reference data only when
  conversion outputs intentionally change.
- Prefer structured models over loose dictionaries for durable schema-like data. Use Pydantic models or dataclasses when data crosses module boundaries, is serialized, or represents a stable contract. Exceptions may apply for internal datatypes and trivial data structures.
- Prefer `pathlib.Path` for path-handling code. Use `Path` operations instead of `os.path` in new or edited code unless an API explicitly requires string paths.
- Avoid `hasattr(...)`, broad `getattr(...)`, and similar attribute-probing patterns. These usually hide interface uncertainty. If such a check is genuinely required for compatibility with a documented third-party API, keep it narrowly scoped and explain in a comment.
- Do not add trivial or self-validating tests. Tests should verify meaningful application behavior, regressions, or integration boundaries, not restate assumptions about well-established library functionality or implementation details introduced only to validate the agent’s own code changes. Avoid mock-heavy tests unless mocking is the clearest way to exercise a real contract or failure mode.

## When making changes

1. Keep edits scoped and consistent with the surrounding module.
2. Update docs/examples when user-facing behavior changes.
3. Run targeted tests for touched behavior.
4. For reference output changes, use `DOCLING_GEN_TEST_DATA=1 uv run pytest`
   and review generated data carefully.

## Before finishing

Run `make validate` before considering a task complete. If hooks modify files,
review the changes and rerun `make validate` until it passes cleanly. Also run
the affected tests for the files or behavior you changed. Use `make check` when
you need a read-only verification pass.
