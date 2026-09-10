# Question-and-options experiment — September 10

**Decision: do not promote this version.** The accepted widget-coverage prototype, production code, and all references are unchanged. The experiment is implemented and replayable in this directory.

[Uncertainty-aware experimental report](uncertainty-report/index.html) · [Raw configuration choices](report/index.html) · [Accepted coverage baseline](../acroform-keying-coverage-20260910/report/index.html)

## What was tested

A group proposal specifies an existing prompt span, its member widgets, and a specific existing caption for each member. Selecting it requires those caption associations in the same global assignment. Ordinary individual assignments and abstention remain available without any caption-side consistency penalty. Caption candidates can sit on either side; prompts can sit above or to the left of the options. Native interleaving and column resets remain possible without changing exported order.

The new proposal code reads geometry, existing span membership, and layout roles. It does not inspect prompt words, alphabetic content, whitespace-word counts, native field-name meanings, or language identifiers. It does not join caption fragments or recover tables. The inherited singleton/component cost model still contains its previously identified alphabetic/short-text assumptions; this is not a claim that the entire inherited prototype is language independent.

The objective favors a repeated caption-to-checkbox arrangement, with costs for prompt separation, differences in caption offsets, intervening text blocks, and using an existing fragment instead of its existing complete span. A group's reward is conditional on selecting its specified captions. However, as the results show, this conditional reward can still favor the wrong segmentation.

To expose uncertainty, the experiment forbids each selected group in turn and resolves the whole assignment. If an alternative complete assignment is within 0.25 objective units, it withholds the disputed group and any local captions that differ between the alternatives. The margin is an experimental cost tolerance, not a calibrated probability. Exact ties are explicitly checked. Each page links to its alternative configurations and score differences. Solver status/objective in the report describe the raw solve before withholding; the displayed associations have the uncertainty filter applied.

## Concrete results

### RC7190: the intended behavior is recovered

Page 1 now uses Application type as the shared prompt for widgets 1–5, with Type 1A, Type 1B, Type 2, Type 3 and Type 5 as their individual captions. Widgets 6–8 remain a separate Type of rebate group. Both groups survive the alternative-assignment check. Their nearest alternatives cost about 0.50 and 0.52 more, respectively.

### Chinese F14446: ambiguity is represented instead of hidden

On page 1, two complete interpretations compete: all A–E options under the real prompt, or B–E under the intervening note while A consumes the real prompt. The latter still wins the raw objective, but the former is only 0.211 cost units away. The filtered report withholds the disputed group and widget 6's caption. This is an unresolved association, not a recovered correct label.

On page 3, the real consent prompt on the Yes/No row competes with the preceding legal paragraph. Their full-assignment difference is only 0.032. The question is withheld while the unchanged Yes/No captions remain. This exposes an error that the earlier green local-label score hid.

### Independent checkboxes remain independent

The previous counterexample, with one left-hand caption and one right-hand caption on independent checkboxes, keeps both correct associations. An ungrouped assignment pays no group-consistency penalty. Tests also cover above/left prompts, opposite caption sides, numeric and non-Latin prompt text, native interleaving, two separate groups in one column, and tied prompt interpretations. All ten structural checks pass.

### Real regressions prevent adoption

The raw configuration model fixes RC7190 widget 1 but makes two previously correct local labels wrong: GST111 page 1 widget 15 and F1120 page 1 widget 28. The latter's Yes caption is displaced by question 6 when the previous, itself incorrect, group hypothesis is removed. This demonstrates how even a bad group can previously have protected text from being stolen locally.

The language-preference form keeps its local captions but loses its correct 21-option question group. The correct full configuration **is present**; it loses to multiple smaller groups seeded by unused caption fragments, including Arabic text and a closing parenthesis. This is an association-role/scoring failure under fragmented input, not a request to reconstruct the captions or add a token blacklist.

The uncertainty filter withholds some wrong outputs, but does not repair these model defects. Compared with the accepted coverage run, it gains one correct local label, loses two previously correct ones, and turns two other wrong labels into explicit abstentions. The frozen local-label totals are 275 correct / 83 wrong / 76 unassigned, versus the baseline's 276 / 85 / 73. The separate no-label and ambiguous-reference cases are unchanged. Partial reviewed group matches fall from 12 to 10. These counts do not measure every false-positive group; the per-page group review is necessary.

## A control explains why more reward tuning is insufficient

A second scoring control counts supported option relationships linearly, with one unit for crossing an intervening block, instead of saturating the group reward. It restores the complete 21-option language group—but merges the two RC7190 questions into one eight-option group. Local-label totals alone do not reveal this regression.

The evidence therefore points to **question-span roles and group boundaries**, not a universal left/right rule or simply a stronger reward for grouping. The implementation of conditional prompt/caption selection and full-assignment alternatives is worth retaining as experimental machinery. The current proposal/scoring rules are not ready to adopt. A further revision should improve geometric evidence that text belongs to an option caption or to a separate question, while keeping competing boundary interpretations available. Unused caption fragments must not automatically become free-standing questions. None of this requires reconstructing their text or introducing language-specific rules.

## Scope and verification

- All 19 pages were replayed for the raw model, uncertainty filter, and scoring control.
- All retained native widget values/order and all 503 table exclusions match the accepted run exactly.
- Accepted prototype source and reference/snapshot hashes are unchanged.
- Ten structural tests pass; the active experimental Python files pass Ruff and scoped `make validate` in the isolated validation checkout.
- This is a development-set experiment with provisional geometry thresholds and an uncertainty margin. It is not held-out evidence of robustness.

## Reproduce

Run from the repository root:

```bash
.venv/bin/python output/acroform-keying-grouping-20260910/build_experiment.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-grouping-20260910/run_experiment.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-grouping-20260910/run_uncertainty.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-grouping-20260910/run_linear_control.py
DOCLING_DEVICE=cpu DOCLING_GEN_TEST_DATA=0 PYTHONPATH=.:output/acroform-keying-grouping-20260910 .venv/bin/python -m pytest output/acroform-keying-grouping-20260910/test_configurations.py -q
.venv/bin/python output/acroform-keying-grouping-20260910/compare_results.py
```

`baseline_algorithm.py` freezes the accepted implementation. `configurations.py` supplies the experimental geometry; `build_experiment.py` creates the isolated module and conditional constraints. `uncertainty.py` implements the alternative-assignment check. `comparison.json` lists every changed local prediction and every before/after choice group. Per-page `.group-audit.json` files record the actual alternatives, not just an uncertainty label.
