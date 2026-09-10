# Widget coverage and checkbox question investigation — September 10

[Updated coverage report](report/index.html) · [Original failure audit](../acroform-keying-failure-audit/index.html)

## Implemented: widget-area coverage

The offline prototype now uses `widget_bbox.intersection_over_self(caption_bbox) >= 0.8` for inline-clause proposals. This is intersection divided by widget area, using the existing Core utility. The threshold mirrors the production form stage's existing text-container coverage convention. No new geometry utility, package, text reconstruction, language-dependent signal or production-pipeline change was introduced.

This replaces exact containment in the existing caption's text envelope, not widget assignment to a whole FORM detection. Table/cell ownership still uses its existing strict containment rule. A qualifying overlap only creates a candidate; it does not independently prove that a paragraph labels every overlapping widget.

The full 19-page replay improved from 256 to 276 correct local-label matches, with no previously correct local labels lost. All 503 table exclusions and native widget identities/order are unchanged. The reference and snapshot hashes remain unchanged. The 23 changed predictions all occur on F1120: 20 now match the reference; three change to the correct enclosing clause but still do not recover the individual From/until component caption. Four component-label failures remain on that page overall. Existing reviewed group matches increased from 6 to 12, chiefly because shared inline clauses now exist; group truth remains partial.

Threshold sensitivity: 80% and 90% each correct 20 flagged labels; 95% corrects 15; 99% corrects 9. No tested threshold loses a previously correct local label. This is development-set evidence, not held-out validation. We chose the existing 80% convention, not a fixture-derived optimum.

Focused checks cover 80% containment, rejecting a neighboring control with only 10% overlap, scale/translation, changed caption text, fixed native sequence, and near-total overlap across a detected cell boundary. All 13 prototype tests passed with generation disabled. `make validate` passed in the existing isolated checkout; the three changed files were verified byte-identical to the workspace after hooks.

## Confirmed: question stealing also occurs above the options

- RC7190 page 1, native widget 1: the first option takes the Application type prompt on its left. Prompt bounds: x=34.97–259.52, y=236.24–256.59; widget bounds: x=263.97–275.97, y=243.16–255.16.
- Chinese F14446 page 1, native widget 6: option A takes the common prompt entirely above it. Prompt bounds: x=36.00–272.51, y=312.19–322.60; widget bounds: x=36.08–46.08, y=328–338. Option A's long explanation and intervening note also split it from B–E in the proposed group.

Neither case justifies a left-only guard. The next grouping work must support prompts above or beside options, without keyword or word-count assumptions.

## Tested and rejected: force adjacent checkbox captions onto a consistent side

A separate in-memory experiment adds a geometric penalty when adjacent native checkboxes in one column get captions on different sides. It does not inspect text. We tested penalties of 1 and 3 against all 19 pages. Both corrected the two stolen option captions on top of the 20 coverage improvements, without losing another previously correct local label on those pages. However, neither corrected the selected shared questions or group membership.

A counterexample with two independent checkboxes exposes the rule's flaw:

```text
AAAA                  [ ]
                      [ ] BBBB
```

Both captions are correctly assigned without the extra rule. With either penalty, the first checkbox loses its correct caption to abstention because the rule incorrectly couples the two independent choices. The complete input and results are saved in `independent-checkbox-counterexample.json`. This is why a better development-set score is insufficient to accept the change.

**Decision:** retain only widget-area coverage in the prototype. For checkbox groups, the next experiment should compare supported question-and-option configurations against an ungrouped alternative. Caption consistency should contribute within a chosen group, not across arbitrary consecutive checkboxes. Geometry alone may leave the grouping uncertain; it must not force a shared question merely because controls align. Existing alphabetic, short-text and whitespace-word-count assumptions need an explicit separate review before broader language-independence claims.

Fragmented captions and missed/truncated tables remain upstream layout defects. No reconstruction or missing-table recovery was added.

## Evidence and reproduction

- `change.diff`: implementation/test/documentation changes relative to the previous validated prototype.
- `baseline_algorithm.py`: exact pre-change prototype, retained for isolated experiments.
- `ablations.json`: four coverage-threshold replays, including every changed prediction and selected group.
- `option-side-experiments.json`: two geometric consistency experiments, not implemented in the prototype.
- `independent-checkbox-counterexample.json`: complete counterexample snapshot and results.
- `report/`: regenerated HTML/JSON report from the actual updated prototype.

From the repository root:

```bash
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-coverage-20260910/replay_thresholds.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-coverage-20260910/replay_option_sides.py
DOCLING_DEVICE=cpu .venv/bin/python -m scripts.replay_acroform_keying --out output/acroform-keying-coverage-20260910/report
DOCLING_DEVICE=cpu DOCLING_GEN_TEST_DATA=0 .venv/bin/python -m pytest tests/test_acroform_keying_prototype.py -q
```

The original audit/report remain the baseline. The original audit's caption-reconstruction recommendation was superseded by the September 10 scope decision.
