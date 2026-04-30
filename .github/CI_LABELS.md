# CI labels

The pull request workflows recognize these optional maintainer labels:

- `tests:full`: run the full Linux CI matrix for the PR, including all ML
  suites and package compatibility lanes.
- `tests:heavy-examples`: run the heavy examples workflow for the PR.

Windows and macOS smoke lanes are intentionally not label-triggered. Run them
from the `Run CI` or `Run CI Main` workflow dispatch inputs when cross-platform
verification is needed.

## ML test segmentation

Expensive ML tests are selected with module-level pytest markers, not workflow
file globs:

- `pytest.mark.ml_ocr`
- `pytest.mark.ml_pdf_model`
- `pytest.mark.ml_vlm`
- `pytest.mark.ml_asr`

New tests run in the core lane by default. If a new test belongs in an ML lane,
add the matching module-level `pytestmark`; do not add per-test file globs to
the workflow.

The workflow intentionally uses a broad ML trigger for code, test, and tooling
changes. Tach performs the fine-grained affected-test selection inside the ML
lanes.
