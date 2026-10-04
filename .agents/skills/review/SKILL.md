---
name: review
description: Review or re-review a Docling pull request with reproducible findings and explicit validation results. Use for code, tests, dependencies, documentation, or agent guidance.
---

# Review a Docling pull request

Use these steps for every review. Human reviewers can use the same steps.
Explain the affected contract so that a reviewer need not know Docling internals.
Read [AGENTS.md](../../../AGENTS.md) and load other applicable
[task routes](../skill-router.json). For document conversion without repository
changes, use the [package usage skill](../../../docling/.agents/skills/docling/SKILL.md).

## 1. Fix the review scope

Read the live PR description, diff, reviews, and CI results. Record the base and
head commit IDs. Use an isolated worktree; preserve other changes and staging.
For a re-review, read each open finding and the author's response.

**Exit:** The changed files and affected contracts are known. The reviewed head
is exact, and prior findings are listed for verification.

## 2. Check behavior and tests

Follow input through the affected code to the user-visible result.

- For conversion changes, check source content, `DoclingDocument` ownership and
  order, and the affected exports. Check nested and non-text content where relevant.
  Text presence or item counts alone do not prove correct structure.
- For a bug fix, run a small case on base and head when practical. The regression
  test must fail for the original defect and pass with the fix. State any limit.
- Check public types, Python 3.10 support, defaults, error paths, and optional
  dependencies where affected. Prefer explicit contracts to attribute probing.
- Inspect test assertions and changed reference data. Confirm that they detect
  the defect rather than accept the new output without a reason.
- For loops over document content, check how work grows with input size. Compare
  base and head on the same input when needed. Do not use noisy timing thresholds
  as a correctness gate.

**Exit:** Each affected contract has evidence, or an explicit unchecked limit.
Every prior finding is confirmed fixed or still reproducible on the new head.

## 3. Run the applicable checks

Use the commands in `AGENTS.md`. For changes you make, run `make validate`, inspect
hook edits, and repeat until it passes. For a read-only review, use `make check`
and targeted tests. Do not regenerate reference data during a read-only review.
Inspect CI on the exact reviewed head and explain failures that affect the verdict.

`python3 .github/scripts/check_skill_routes.py` checks contributor skill routes
and Codex/Claude links. Existing Ruff, ty, Tach, and lock checks check code and
dependency rules. These checks do not prove conversion quality, test adequacy,
or full ASD-STE100 compliance. Those decisions require review.

**Exit:** Record commands, results, reviewed commit, and checks not run. Never
describe skipped, pending, or failed checks as passed.

## 4. Give a precise verdict

Use one finding per defect. Include its location, trigger, actual and expected
behavior, user impact, and reproduction or test evidence. Link the affected code.
Use these severity terms consistently:

| Severity | Meaning |
| --- | --- |
| Blocker | Security defect, data loss, incorrect supported behavior, or a required check failure. Request changes. |
| Suggestion | A useful improvement with no demonstrated contract failure. Do not block on preference. |
| Question | Evidence is missing. Ask for the specific evidence; do not assert a defect. |

Approve only when no blocker remains and relevant validation is complete.
If required evidence is unavailable, state the limit and withhold approval.
Post reviews or comments only when the user has authorized that action. Before
posting, confirm that the PR head still matches the reviewed commit. If it changed,
check the new diff and repeat affected validation. Avoid duplicate findings.

**Exit:** The verdict identifies the reviewed head, blockers, validation results,
and limits. Follow the communication rule in `AGENTS.md`.
