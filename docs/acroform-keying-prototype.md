# Testable AcroForm pairing prototype

This is an offline experiment on the saved September 8 page snapshots. It does
not change the production PDF pipeline or produce production DCLX exports.

From the repository root, run:

```bash
uv run --no-sync python -m scripts.replay_acroform_keying
```

Open `output/acroform-keying-prototype/index.html`. Select a page and a native
widget to see the reviewed label and the proposed connection over the source
image. The per-page JSON preserves native values in the supplied page-wide
sequence; groups reference those values without moving or duplicating them.

For a shorter iteration:

```bash
uv run --no-sync python -m scripts.replay_acroform_keying --only f1040lep gst494
uv run --no-sync python -m pytest tests/test_acroform_keying_prototype.py -q
```

The replay requires `output/acroform-keying-review-20260908/` and the original
PDF directory recorded in the annotation manifest. Override these with
`--evidence` and `--fixtures`; use `--out` for separate experimental runs.
Fixture hashes, page dimensions, native sequence, reference geometry, and widget
type/name are checked before scoring. The small regression input is included
alongside the prototype so focused tests do not need external evidence or PDFs.

The experiment uses the existing NumPy, Pydantic, Core geometry, and SciPy
packages, without importing the conversion pipeline or loading native models.
It requires SciPy 1.9 or later for `milp`. The repository's production SciPy
minimum remains unchanged; the experimental tests skip on older SciPy.

## Table rule

- Outside detected tables: ordinary pairing.
- Inside a detected table: excluded, except for labels and values wholly inside
  the same unambiguous detected cell.
- No cross-cell pairing, row/column-header inference, or table recovery.
- If a table is not detected, the optimizer does nothing special about it.

Saved cell boxes sometimes tightly cover text instead of the full physical
cell. The prototype uses the available boxes conservatively; it does not
extrapolate a grid to enlarge the exception.

## What the first experiment does

It compares singleton labels, existing multi-line text spans and their fragments,
inline clauses, adjacent text-field components, and candidate choice questions.
All candidates share text ownership constraints. Choice questions are distinct
from option captions. Costs include distance, alignment, text-role evidence,
unrelated-widget obstruction, intersecting connections, and label movement
inconsistent with adjacent native values in a comparable lane. Column resets
are allowed. Values can remain unpaired; `--null-cost` controls that experimental
tradeoff. Weights are starting hypotheses, not calibrated probabilities.

The composite-field candidates use a common envelope, charge distance per value,
and reject groups with separate sibling captions. Choice candidates use compact
checkbox runs or shared native names as evidence; native names are never emitted
as visible labels. Candidate construction remains heuristic and needs more work.

Inline-clause proposals require at least 80% of each member widget's area inside
the existing caption's text envelope, using
`widget_bbox.intersection_over_self(caption_bbox)`. This mirrors the form stage's
text-container coverage threshold and tolerates small protrusions beyond text
bounds. It does not change detected table/cell ownership or reconstruct captions.
The coverage test only proposes a shared clause; the assignment still evaluates
competing uses of its text and widgets.

The solver minimizes the objective over the generated candidates. An optimal
solver status does not mean the associations are semantically correct. On a
solver failure or timeout, the page preserves all values without assignments;
the report records the failure and the command exits unsuccessfully.

## How to read the results

The main report shows correct, wrong, and unassigned reviewed local label links.
All 443 eligible widgets now have explicit visual reference decisions: 434
visible labels, eight with no visible label, and one ambiguous case. The 503
detected-table exclusions are separate. A no-label decision rewards abstention
and counts an assigned label as wrong. Ambiguous cases are displayed but not
scored. Missing, duplicated or stale reference records stop the replay.
Expected text
hints identify regions and are not exact-text targets. The automatic check uses
spatial overlap in both directions to reject oversized paragraph matches; it
does not certify complete text or resolve semantic ambiguity.

The page overlay shows detected forms in cyan, with detected tables cut out,
and tables in grey. Selected widgets show their form/table membership and the
reference reason. Some native widgets lie outside detected forms and remain
eligible. A grid that was not detected as TABLE remains eligible too; its shared
row caption can be a local-label reference without reconstructing a table.
The manifest pins snapshot and source-image hashes, so these decisions cannot
silently be applied to a different detection run.

Question/composite/clause membership is scored separately in `summary.json` and
per-page JSON. Form scopes and table row/column annotations are not pairing
targets. A correct option caption does not imply a correct common question.

Fragmented captions and missed/truncated tables are upstream layout defects;
caption reconstruction and missing-grid recovery are out of scope here.
Known association limits: reliable local label-side conventions, an independent
alternation objective, broader choice-group construction, and
integration with Core traversal/serializers remain future work. The saved tables
come from the baseline run, whose form stage preceded table inference; a future
pipeline integration must validate fresh table inputs after the stage reorder.
The development forms have already influenced this experiment; this is not a
held-out accuracy estimate.

The September 10 widget-coverage replay corrected 20 local-label failures across
the 19 development pages, with no previously correct local labels lost. Thresholds
of 80% and 90% gave the same local results; 80% follows the existing text-container
convention rather than selecting a threshold specific to these forms. Native
widget order and all 503 table exclusions stayed unchanged. This does not certify
group membership or resolve separate From/until components inside one clause.

Question stealing occurs both beside and above the checkbox column: RC7190 page 1
uses the left-hand Application type prompt as the first option caption; Chinese
F14446 page 1 uses the prompt above option A. A left-only guard cannot fix the
general failure. Existing alphabetic/length/whitespace-word checks are experimental
language-sensitive assumptions requiring separate review; this containment change
introduces no language-dependent signal.
