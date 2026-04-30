# CI labels

The pull request workflows recognize these optional maintainer labels:

- `tests:full`: run the full Linux CI matrix for the PR, including all ML
  suites and package compatibility lanes.
- `tests:heavy-examples`: run the heavy examples workflow for the PR.

Windows and macOS smoke lanes are intentionally not label-triggered. Run them
from the `Run CI` or `Run CI Main` workflow dispatch inputs when cross-platform
verification is needed.
