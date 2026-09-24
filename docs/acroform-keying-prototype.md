# AcroForm keying: replay and evaluation

The keying described here ships in `docling/models/stages/form_field/keying/`
and runs in the standard PDF pipeline when `extract_form_fields=True` (CLI:
`--extract-form-fields`). This document covers its offline evaluation on the
frozen page snapshots.

From the repository root, run:

```bash
uv run --no-sync python -m scripts.replay_acroform_keying \
    --evidence output/acroform-keying-handover-20260916/output/frozen-review-evidence \
    --fixtures output/acroform-keying-handover-20260916/fixtures
```

Open `output/acroform-keying-replay/index.html` (the `--out` directory). Select a page and a native
widget to see the reviewed label and the proposed connection over the source
image. The per-page JSON preserves native values in the supplied page-wide
sequence; groups reference those values without moving or duplicating them.

For a shorter iteration:

```bash
uv run --no-sync python -m scripts.replay_acroform_keying --only f1040lep gst494
uv run --no-sync python -m pytest tests/test_acroform_keying.py -q
```

The replay requires the frozen evidence directory (snapshots and page images)
and the original PDF directory recorded in the annotation manifest, passed
with `--evidence` and `--fixtures`; use `--out` for separate runs.
Fixture hashes, page dimensions, native sequence, reference geometry, and widget
type/name are checked before scoring. The small regression input is included
alongside the tests so focused tests do not need external evidence or PDFs.

The keying uses NumPy, Pydantic, Core geometry and SciPy (`milp`, imported
only when a page has widgets; SciPy 1.9 or later). The replay does not load
native models.

## Table rule

- Outside detected tables: ordinary pairing.
- Inside a detected table with cell structure: the value is keyed from the
  table's own cells, outside the free-form search. It sits in the row and
  column whose bands it overlaps most (a band is the extent of the single-span
  cells of that row or column, because saved cell boxes cover text, not the
  printed cell). The key is the lettered text of its own cell; otherwise the
  first lettered cell to its left (the row caption). The first column header
  above is kept as context and is the key only when there is no row caption.
  Cell text is used whole; codes and units without letters never key a value.
- Values wholly inside one unambiguous detected cell keep ordinary pairing
  with labels in that cell. A detected table without cell structure stays
  excluded.
- If a table is not detected, ordinary pairing applies; its row captions are
  reached through the visibility and sharing rules below.

## What the keying does

A single-value caption is a candidate only if it is the first lettered text met
on one side of the value (inside, above, below, left or right). Another text
line crossing the corridor hides whatever lies behind it; digits, codes and
symbols are transparent, as are other values along a row. Walking up or down a
column passes only like-sized values of the same kind. Diagonal captions are
left to group candidates.

The cost orders these candidates: a caption in the value's own printed cell
(inside, or touching it above or below) first, then an aligned caption along
the same line, then an aligned caption farther up or down. Misalignment costs
as much as a cell difference; proximity only breaks ties. Captions precede
their values in reading order, so a caption below costs slightly more, and text
to the right only captions a value when adjacent (an option caption after its
checkbox).

A caption may key a whole line of like-sized values of the same kind in one row
or column (a row caption over its amounts), and a checkbox line and a text line
of the same row may share it. Otherwise each text atom keys at most one field.
Captions the layout split into pieces are rebuilt, first along a text line and
then across lines, and the whole caption is preferred over its pieces. Pieces
of one line join when they are adjacent, of one height, with no value between
them and the same lines just above and below; sub-captions each over their own
box stay apart. Lines join when they are each other's only neighbour, share a
line height and a left edge or centre, and the upper line is not already beside
a value the lower one misses (a caption ends at its value's row). A line with a
value inside it is an inline caption and never joins.

Aligned like-sized neighbours in the native order that read their captions
from different sides pay a small cost (0.5, below any cell or alignment
difference), so symmetry only settles close calls. A value in a grid (like-sized
siblings in both its row and its column) keeps the best aligned caption along
the other axis as context, when a sibling along that line sees it too (a
column header, a group question). Context is reported, never scored as the key.

It also keeps existing multi-line text spans and their fragments, inline
clauses, adjacent text-field components, and candidate choice questions. Choice
questions are distinct from option captions. Pairwise costs penalize
intersecting connections and label movement inconsistent with adjacent native
values in a comparable lane. Column resets are allowed. Values can remain
unpaired; `--null-cost` controls that experimental tradeoff. Weights are
starting hypotheses, not calibrated probabilities.

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
solver failure or timeout, the page preserves all free-form values without
assignments (table keys do not depend on the solver); the report records the
failure and the command exits unsuccessfully.

## How to read the results

The main report shows correct, wrong, and unassigned reviewed local label links.
All 443 eligible widgets now have explicit visual reference decisions: 434
visible labels, eight with no visible label, and one ambiguous case. The 503
values in detected tables are reported separately (`table: …` statuses): the
60 inside annotated grids are scored by comparing the key with the annotated
row header text (the column header is context); the rest are keyed but
unreviewed. Only the primary key is scored; `summary.json` adds informative
`context` tallies, such as wrong keys whose context matches the reference. A no-label decision rewards abstention
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
per-page JSON. Form scopes are not pairing targets; annotated table row/column
headers score only values inside detected tables. A correct option caption
does not imply a correct common question.

Fragmented captions and missed/truncated tables are upstream layout defects.
Only captions split into pieces along and across text lines are rebuilt; other caption
reconstruction and missing-grid recovery are out of scope here.
Known association limits: reliable local label-side conventions, an independent
alternation objective, broader choice-group construction, and
integration with Core traversal/serializers remain future work. The saved tables
come from the baseline run, whose form stage preceded table inference; in the
pipeline the stage now runs after table structure, and
`tests/test_form_extraction.py` covers keys read from fresh table cells.
The development forms have already influenced the rules; this is not a
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

The September 18 replay measured each change in sequence against the accepted
276 correct / 85 wrong / 73 unassigned local labels (434 visible labels):

| Change | Correct | Wrong | Unassigned | Without T3MB |
|---|---:|---:|---:|---|
| Visible-side candidates, cell/alignment cost | 309 | 75 | 50 | 280 / 53 / 26 |
| Shared row and column captions | 371 | 61 | 2 | 307 / 50 / 2 |
| Captions split per line joined back | 392 | 40 | 2 | 325 / 32 / 2 |
| Table values keyed from cells | 392 | 40 | 2 | 325 / 32 / 2 |
| Captions split along a line joined back | 402 | 30 | 2 | 332 / 25 / 2 |
| Grid context (informative) | 402 | 30 | 2 | 332 / 25 / 2 |
| Symmetry between sibling values | 403 | 29 | 2 | 333 / 24 / 2 |

The seven correct abstentions and the 12 correct groups are unchanged;
F1040LEP, GST494 and F14446CN lose no label. Table keys match the annotated row
header for 60 of 60 values in the two annotated detected grids, and the column
context matches for all 60; 431 table values are keyed without a reference and
12 get no key. These rules were shaped on the same 19 pages; they are not a
held-out estimate. Row keys in the unannotated Italian tables are often line
codes such as `A1`, which the letter test does not treat as codes.
