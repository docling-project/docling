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

Path filters still decide whether a CI lane should be created at all. Pytest
markers only select which test modules run after a test lane has started.

## Cross-platform smoke tests

Windows and macOS smoke tests are selected with `pytest.mark.cross_platform`.
Use this marker for lightweight modules that should be exercised by the
workflow-dispatch cross-platform lanes; do not maintain a separate test-file
list in the workflow.

## AI triage labels

The `AI PR Triage` workflow (`.github/workflows/ai-pr-triage.yml`) runs on every
non-draft PR before a maintainer approves CI. It never checks out or runs PR
code. It reads the PR through the GitHub API, applies deterministic rules, and
asks Bob Shell in a read-only mode for an advisory assessment. It updates one
sticky comment and sets these labels:

- `ai:ci-safe`: no deterministic rule matched and the model found no concern.
- `ai:ci-needs-care`: a sensitive path changed (CI, build, dependency, or agent
  instruction files), or the model reported a concern. Read the comment before
  you approve CI.
- `ai:possible-duplicate`: the model rated an open or earlier PR as a duplicate.
  Candidates come from shared issue references and overlapping diff hunks.

- `ai:review-lgtm`: the AI first review found no `blocker` or `major` issue.
- `ai:review-changes`: the AI first review suggests changes. See its inline
  comments.

The first review is a `COMMENT` review with inline comments. It never approves
or requests changes. It does not run for PRs rated as duplicates, for PRs with
more than 60 source files or 2000 changed source lines, or for a commit that it
already reviewed. After a push, it reviews only the new commits, if the history
was not rewritten.

The triage also adds topic labels from the existing repository labels, for
example `bug`, `enhancement`, `docx`, `markdown`, `ocr`, or `table structure`.
Rules map the conventional-commit title (`fix` → `bug`, scope `(docx)` →
`docx`) and the changed source paths to labels, and the model can add up to
three more. The list is in `.github/scripts/pr_topic_labels.py`. The workflow
adds topic labels only on the first triage of a PR, never creates a label, and
never removes one, so maintainers can correct them.

### `/ai` commands

After a maintainer (`OWNER`, `MEMBER`, or `COLLABORATOR`) comments on or
reviews a PR, a push runs only the triage, not the AI review. Maintainers can
ask for more with a PR comment that starts with one of these commands:

- `/ai review` (or `/ai`): a new, full AI review of the current commit. It
  also runs for a commit that was already reviewed and for a PR rated as a
  duplicate. The size limits still apply.
- `/ai triage`: run the triage again.

The bot adds a 👀 reaction to confirm the command. Commands from other users
are ignored.

The comment also summarizes changed reference data in `tests/data/**/groundtruth/`.
It separates formatting-only and coordinate-only changes from text, table, and
structure changes.

The labels are advisory. They never add `tests:full` and never approve a workflow
run. A new push removes them and runs the triage again.

The workflow is never a blocking check. Every step uses `continue-on-error`, so
an error (for example a missing secret, an API outage, or an invalid model
answer) shows as a warning annotation on the run, and the PR check stays green.
If a step fails, the comment shows only the parts that finished.

The workflow needs the `BOB_API_KEY` secret (an API key with the Inference
scope). Optional repository variables: `BOB_TEAM_ID` (for `general` keys) and
`BOB_TRIAGE_MAX_COST` (default `2`), and `BOB_REVIEW_MAX_COST` (default `3`). Without the secret, the comment shows only
the deterministic checks.
