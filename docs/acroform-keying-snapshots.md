# AcroForm keying development snapshots

These four commits reconstruct checkpoints from saved working files. They are
created now, not presented as contemporaneous historical commits. Production
code is unchanged; the algorithms are offline experiments.

The worktree branch is `codex/acroform-keying-snapshots`, based on
`db0564fdda1c3df9587b3b86b1c80a82022436b0`.

## Checkpoint 1: frozen visual evidence and evaluation

Save the complete local-label review, native widget identities, frozen page
images and Docling snapshots, the evaluator, and the design handoff. The evaluator
imports the optimizer introduced in checkpoint 2; this first checkpoint can
independently validate reference identities with the stdlib validator.

The original PDFs remain external. Their hashes and source directory are in
`tests/data/groundtruth/acroform_keying/manifest.json`; replay also accepts
`--fixtures`. No document conversion or model download is needed for replay.
The 19 saved page images and snapshots are included in Git. Generated reports
are preserved byte-for-byte in later checkpoints so the original decisions can
be inspected without rerunning anything.

## Checkpoint 2: baseline and widget coverage

Save the accepted offline optimizer, focused tests, pre-coverage source copy,
coverage sweeps, baseline/coverage reports and failure audit. The change uses
intersection divided by widget area for inline clauses while keeping table/cell
ownership strict. The accepted result is 276 correct local labels; fragmented
captions and missed/truncated tables remain upstream defects.

## Checkpoint 3: first grouping experiment

Save conditional question/caption configurations, full-assignment alternatives,
tests, raw and filtered reports, the linear scoring control, and the subsequent
role investigation. This version was rejected: it creates caption-fragment
questions and loses correct local associations. Its README records the evidence
and scope of rejection. Baseline source and references remain frozen.

## Checkpoint 4: caption alternatives and nested references

Save the follow-up algorithm, tests, geometry checks, source fingerprints,
comparison, and raw/filtered reports. It retains alternative captions and makes
nested groups reference the parent option's caption. Across all 19 pages it
fixes one wrong local association without losing previous correct local labels.
Sami caption selection and the outer Yes/No boundary remain unresolved; this
version is experimental, not production integration.

Thirty focused tests pass, and every replay preserves native values/order and
all 503 table exclusions. Per-checkpoint commit messages record validation.
See each experiment README for commands and limitations. Run the two experiment
test modules separately: their historical helper module names overlap.

Historical hash manifests retain their original capture paths. When checking
a relocated worktree, resolve their repository-relative suffix under that
worktree. The file bytes and hashes remain unchanged. Replay/check scripts
may regenerate reports or path-bearing manifests; use `--out` for temporary
replay destinations when preserving these committed checkpoints.
