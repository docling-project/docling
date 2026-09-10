# AcroForm keying: algorithm review and redesign handoff

Date: 2026-09-08. Reviewed branch: `feat/acroform-label-binding`, HEAD
`db0564fd`. This is an analysis and recommendation, not an implemented redesign.
Production code, dependency pins, and the previous handoff were not changed.

## Decision

**Redesign the association problem before tuning more constants.** Keep native
widget extraction, state normalization, and the existing field-item output
primitives. Replace the page-wide greedy label pass with a region-scoped
assignment over explicit candidate labels, choice groups, and abstentions.
Resolve table ownership separately using table structure. Treat overlap,
direction, order, and textual evidence as inputs to that assignment rather than
independent paths that commit to an answer before it runs.

**AcroForm value order is fixed, reliable input throughout this design.** The
optimizer assigns labels and chooses groups along that supplied sequence; it
does not infer, score the reliability of, or replace the value order.

The implementation is not especially large or slow. Its problem is that its
claimed guarantees are stronger than the algorithm, and its input/assignment
model cannot express several ordinary form structures. Replacing greedy with a
larger optimizer using the same inputs and distance cost would retain many errors.

There is no defensible set of universally validated weights from this corpus.
The measured ablations below identify useful signals and disprove several old
claims. The proposed weights are starting values for a new, explicitly defined
objective; they are not presented as calibrated production defaults.

## Evidence and reproducibility

The supplied directory contains **11 PDFs, 19 pages**, not 15 PDFs. All 19 source
page images were rendered and visually inspected directly. Expected labels were
inferred from those images; existing bindings and PDF tooltips were not used as
an oracle. Inspection covers all forms, but exhaustive manual scoring of every
widget was not performed.

All conversions ran **inside the sandbox with `DOCLING_DEVICE=cpu`**. The final
run used native form extraction, default OCR/table processing, and page images
at scale 2.0. All 11 conversions reported `SUCCESS`, with no missing pages.
There were table postprocessing warnings, preserved in the log.

- Native widgets before filtering: **970**.
- Native values retained after existing zero-size/push-button filtering: **946**.
- Predicted field items: **928**, of which **290 have a nonempty key**.
- Serialized DCLX values: **946**; embedded DCLX page images: **19**.
- Exact replay of the unchanged form-field stage reproduced every prediction on
  all **19 pages** from saved pre-stage layout/widget snapshots.
- Existing tests: **16 passed**, four warnings, across
  `tests/test_acroform_label_binding.py` and `tests/test_form_extraction.py`.
- Timed form-field stage: median **1.08 ms/page**, maximum **13.10 ms/page**,
  total **46.77 ms**. Complete conversions took **34.55 s**, including model
  initialization. These are one-run observations, not benchmark medians.

Counts of keyed items are coverage indicators, **not correctness scores**.
The previous handoff's `19/19` or `21/21` claims should not be read as semantic
accuracy measurements.

### Environment boundary

The ordinary `.venv` aborted during optional MLX import, even with CPU selected.
A temporary environment excluded MLX packages. Its installed AcroForm dependencies
also differed from the lock: the installed parser was on main and lacked the
required contract. Those failed attempts are excluded from the results above.

The successful run imported current Docling plus the exact locked dependency
sources:

| Component | Source used |
|---|---|
| Docling | `db0564fd` in this checkout |
| docling-core | `d21826a672b2ccd95d6b37a128002b7bba6ca505`, version 2.95.0 |
| docling-parse | `ea05b85c866515602d83619f5cd5c79768d92785`, version 7.16.0 |
| Python | 3.13.5 |
| Other reused installed packages | docling-ibm-models 4.0.2, torch 2.13.0, transformers 5.16.1 |

The pinned parser's C++ binding was rebuilt in `/private/tmp` for Python 3.13,
using existing local native static libraries. Its complete widget-contract test
passed before conversion. This is **not a fresh, fully frozen environment**;
installed distribution metadata alone misidentifies the source-overridden parser.
The manifest records that distinction. Reproduce in a clean frozen environment
before using these numbers as release acceptance criteria. Differences from the
September 4 counts cannot be attributed to the matcher alone.

### Saved evidence

Local evidence directory, retained alongside this handoff:

`output/acroform-keying-review-20260908/`

- `baseline/`: all 11 DCLX archives, Docling JSON, and Markdown exports.
- `pages/`: all 19 independently rendered source images.
- `snapshots/`: pre-binding layout and native widgets, post-binding predictions,
  and final table predictions, per page.
- `manifest.json`: source identities, options, fixture hashes, status and timing.
- `replay.json`: actual matcher inputs, bound cluster IDs, and item-to-native-index
  mappings, including widgets whose displayed checkbox bbox was enlarged.
- `gold.json`: **102 image-reviewed associations** on eight pages from five PDF
  families, with widget index, expected cluster ID, visible text, and bbox.
- `ablation.json`, `ablation.log`, `ablate.py`: diagnostic comparisons below.
- `replay.py`: asserts exact reproduction of all saved stage predictions.
- `check_counterexamples.py`: runnable proofs of two incorrect algorithm claims.
- `run_audit.py`, `build_native.py`, logs: conversion/instrumentation and native
  build procedure. The build helper retains machine-specific paths.

The evidence directory is local and untracked; include it when transferring this
handoff. Snapshot IDs are valid only with the recorded fixture hashes and model
outputs. PDF page numbers below are **positions in the thinned PDFs**, which can
differ from printed page numbers.

The 102-link sample scores assignment to an existing visible label/option-caption
cluster. It does not score question-group ownership, table cells, exact text
completeness, or validity of native annotations. For example, a correctly selected
Arabic caption cluster can still omit the separately extracted Arabic text. It
also excludes GST494's two Business Number widgets, whose desired shared key
cannot be represented by the current 1:1 matcher. This is a diagnostic development
sample, not a random sample or held-out accuracy estimate.

## Reusable visual ground truth

The reusable annotation set is saved at
`tests/data/groundtruth/acroform_keying/`, with its own README and versioned
manifest. It contains **147 visible label links, 26 groups/scopes, 18 table
regions, and explicit cell membership for 128 widgets in five grids**. The
original 102-link diagnostic subset remains marked separately, so the ablation
numbers above are unchanged. Five notes preserve unresolved judgments.

Fixture SHA-256, thinned-page number, native widget index and original rectangle
identify values. Expected spatial label anchors and semantic group/cell membership
are ground truth; extracted text hints and historical cluster IDs are only aids.
All 970 native references are saved, but unannotated references are not implicitly
correct, incorrect, or keyless. The set covers observations across all 11 forms
without claiming exhaustive semantic annotation. Its integrity checker verifies
all hashes, page/identity references, coordinate bounds and fixed group order.

## What the page images say versus the output

| Fixture | Pages | Values / keyed items | Direct visual interpretation and observed result |
|---|---:|---:|---|
| `gst111-fill-08e` | 2 | 84 / 35 | Identification fields above; fiscal dates have year/month/day components; institution type is a three-choice question. Page 2 has tax lines and two matrices with row/column headers. The output maps the postal-code value to `Telephone number`, the telephone value to dash marks, and a last table amount to `Page 4`. The exported-supplies matrix is not detected as a table and contributes numeric label errors. |
| `gst494-fill-09e` | 1 | 44 / 15 | Ordinary identity fields, two dates, one three-option reporting-period group, Schedule A matrix, certification. The 14 scored local captions are correct. But `RT` is a fixed Business Number component, not the semantic key: widgets #4 and #5 belong to `Business Number (BN)`. Reporting-period options lack their common question. The old ground truth accepted `RT` and omitted cases; it is not complete semantic ground truth. |
| `t3mb-fill-15e` | 2 | 83 / 19 | Tax calculation grids require row and bracket-column context. Page 1 emits no keys: its main form content is absorbed into a PICTURE cluster with 172 children, leaving only eight free label candidates. Page 2 binds 18 grid values to arithmetic signs and the contributions field to `A`. Neither page has a TABLE prediction. |
| `rc7190-ws` | 2 | 19 / 19 | `Application type` governs five options; `Type of rebate` governs three. Page 1 binds the first checkbox to the question instead of `Type 1A`, and line 1's amount to `1`. On page 2, amount fields bind to `2`, `3`, `4`, `5`, `6`, `$`, and `) × line 3:` instead of their prompts or operand roles. All 19 having keys is not success. |
| `agenziaentrate_e5017784` | 3 | 326 / 67 | Page 1 mixes identity fields and matrices; page 2 is mostly tax grids; page 3 contains repeated beneficiary/signature panels. Wrong local associations include `C.A.P.` to a minor-status checkbox and `NUM. CIVICO` to the postal-code value. Page 2 still has a section-header binding and `,00` keys where table detection leaves gaps. Page 3 gives six beneficiary-tax-code values dotted signature rules as keys. |
| `rf-1084s` | 1 | 63 / 4 | The top block has three rows of two label/value pairs. Native #1/#3/#5 belong to `Namma`, `Ealáhus- (kontor-) čujuhus`, and `Boastanr./-báiki`; instead they take the next column's labels. The right-column widgets stay keyless. The lower 56 widgets are inside the detected table and need structural interpretation. The top block is visibly ruled even though it is not a TABLE prediction. |
| `rf-1125s` | 1 | 96 / 48 | Identity fields are followed by a three-vehicle matrix A/B/C, including per-cell options and person details. The detected table ends at y≈530, above much of the matrix. Below that point, text fields repeatedly steal the following row's captions: e.g. #74 gets `Riegádannr.` although the field is in the `Namma` row. Whole-table exclusion by predicted bbox is insufficient. |
| `rf-1177s` | 2 | 129 / 12 | Page 1 contains two distinct choice questions and multiple side-by-side tables; page 2 is tabular. Header fields are generally sensible and table widgets mostly abstain. Options remain separate keyed fields rather than groups. Native indices #73–76 return to top-page choices after many lower table widgets, demonstrating why one page-wide y frontier is wrong. |
| `f14446cn` | 3 | 40 / 35 | Page 1 has six ordinary fields and options A–E; option A gets the common question instead of its caption. Page 2 has 11 clearly labeled large response boxes, all correctly associated. Page 3 contains two consent questions and two parallel signer blocks: five right-column fields are keyless and the right electronic signature is keyed `或者` (“or”). All six failures disappear in the no-frontier replay. |
| `f1040lep` | 1 | 23 / 15 | Two identity fields and one language-choice group with 21 options arranged in two columns. Eight right-column choices are keyless, and Traditional/Simplified Chinese captions are swapped. The no-frontier replay associates all 23 local captions correctly, but still does not construct the common language-choice group. |
| `f1120so` | 1 | 39 / 21 | Inline checkbox/date clauses are the useful existing case: 39 values are grouped into 21 keyed items, including multi-value clauses. Preserve this capability. However question-to-option hierarchy is still absent, and checkbox geometry is expanded to the enclosing option paragraph. A count of 21 keyed items does not validate those broader semantics. |

### Especially important corrections to the prior handoff

1. **Reliable native order is not a global y-sort.** F1040LEP's second column
   and F14446CN's second signer block are legitimate returns upward in the fixed
   supplied value sequence. That sequence is authoritative. The failure is the
   label matcher's geometric frontier, not the widget order. Compare candidate
   labels along the given value sequence, with scope-specific subsequences.
2. **`/TU` is not empty everywhere.** It is nonempty on **150/970 raw widgets**,
   covering rc7190, Manitoba, and F14446CN. Rc7190 has useful full prompts and
   native repeated field names identify its five-option application-type group.
   Conversely F14446CN's A/B tooltips say “social security number”/“documents”,
   inconsistent with the visible option titles. Use metadata as corroboration,
   retain its provenance, and do not display it as an inferred visible label.
3. **Same TABLE bbox is not same cell.** The current boundary prevents some
   cross-table links, but does not associate a widget with a row/column, and it
   does nothing where table detection is incomplete or absent. A ruled block
   can be a form with labels inside cells, a matrix, or a mixture of both.
4. **Distance minimization, order agreement, geometric non-crossing, and
   alternation are distinct objectives.** None can be substituted for all others.

## Measured ablations

All methods replay the saved, unchanged candidate pools on the same 102
image-reviewed links. No production changes or model reruns were used here.
“Wrong” means an assigned label differs from the annotated visible cluster;
“missing” means there was no assignment.

| Diagnostic method | Correct | Wrong | Missing |
|---|---:|---:|---:|
| Current greedy, 2.0 cap / 1.5 row band | 70 | 16 | 16 |
| Same greedy/cap, remove vertical frontier | 86 | 14 | 2 |
| Exact linear assignment, same gap/cap, explicit null cost, no frontier | 86 | 14 | 2 |
| Exact assignment with simple role/direction/text-quality costs | 90 | 12 | 0 |

The global-gap method uses `gap / h`, a null cost of 2, and excludes edges outside
`2h`; labels can be unused. This has the same feasible edges as no-frontier
greedy, and gains nothing on this sample. It is a relaxed comparison to current
greedy because it removes the frontier. It does not prove that all optimal
assignments equal greedy, or that global optimization is generally unnecessary.

The final diagnostic uses `log(1 + gap/h)`, null cost 3, alignment penalty 0.5,
nonalphabetic-label penalty 3, and direction penalty 1.5. For text widgets it
penalizes trailing-right labels; for checkboxes it penalizes left labels. It
removes the distance cap. These choices deliberately test a simple hypothesis;
they are **not the proposed final algorithm** and use no tooltip evidence.

It improves aggregate counts but regresses GST494 from 14/14 to 11/14 and
F14446CN page 1 from 10/11 to 9/11. Direction-weight sweeps of
`0, 0.5, 1, 1.5, 2, 3` yield respectively
`88, 90, 89, 90, 92, 92` correct out of 102. Even the highest aggregate scores
still contain ten wrong links. A single global “labels go on this side” weight
is therefore not the answer. A future optimizer needs local style and structure,
not another fixture-selected direction rule.

## Implementation findings

References are relative to this repository at the reviewed HEAD.

### 1. The core has neither its claimed global objective nor crossing guarantee

`form_field_model.py:37–113` (`_precedes`, `_match_labels`) scans all labels for
each widget, greedily consumes one, and remembers the last label's bbox. It does
not sort or model a label sequence, compare alternative complete assignments,
measure key-value segment intersections, or inspect intervening widget rectangles.

A runnable counterexample has widgets x=[10,20] and [30,40], labels A=[21,27]
and B=[4,7], all in the same y band. Greedy picks A for the first widget (gap 1)
and skips the second (cost 20 under the documented skip interpretation): total
21. B→first and A→second costs 3+3=6, satisfies the same row constraint, and is
geometrically non-crossing. The advertised minimization property does not hold.

The frontier also permits cumulative upward drift: labels at y=40,30,20,10 all
pass a tolerance of 15 when visited in that order. Comparing only against the
last label does not enforce global vertical monotonicity either.

**Recommendation:** delete the incorrect guarantees and replace the frontier with
explicit local ordering factors. The no-frontier ablation identifies a defect,
not a complete production patch.

### 2. Assignment is page-scoped even when FORM groups exist

`form_field_model.py:369–418` partitions only by table ID. Every non-table
widget and label shares the `None` partition, across different FORM containers,
columns, and sections. One wrong label can consume another form's candidate or
move its frontier. `_match_form` affects eventual output grouping, not matching
scope. A FORM label can also wrap almost the entire page and is not itself an
adequate semantic partition.

**Recommendation:** scope matching to local containers/lanes, retaining native
indices. Use layout parentage and visible separators, refined by candidate
connectivity. Do not hard-split a group merely because it spans columns, as
F1040LEP's question legitimately does.

### 3. The candidate pool loses essential labels and admits obvious distractors

`TEXT_ELEM_LABELS` includes headers, footers, formulas, and checkbox clusters.
`label_pool` is only the top-level text list minus inline containers. Actual
labels can be TABLE/PICTURE children; long labels may be split into separate
fragments; numbers, decimal suffixes, and dotted rules can be closer than a
prompt. `_median_line_height` measures cluster height, not line height, so
paragraph segmentation changes both thresholds.

RF1084S makes the cap problem concrete: `Namma` ends at x≈71 and its widget
starts at x≈139, while the incorrect next-column caption begins less than a
point after that widget ends. A cap based on a few short text heights excludes
the correct candidate before any optimizer can help.

**Recommendation:** preserve references to text atoms/lines inside containers;
form label-span candidates without duplicating table contents into the free
pool. Estimate local text scale from text cells/lines. Generate candidates by
structural neighborhood and aligned visibility as well as proximity; measure
candidate recall before adjusting scoring weights. Preserve numeric identifiers
as context, but distinguish a line number/unit from the full semantic prompt.
Do not add literal filters for `RT`, `ULUL`, `,00`, or named forms.

### 4. Three binding routes make incompatible early decisions

Checkbox overlap, enclosing text, and detached-label search each commit
independently (`form_field_model.py:205–363`). Detached matching is 1:1, whereas
an enclosing paragraph can group multiple values. A detected checkbox caption
is excluded from keyless matching as though having an option caption implied
having a common question. Those are different relations.

Additionally, promoted checkbox IDs are not explicitly removed from the later
label pool; they are only removed indirectly if also registered as text
containers. The code does not enforce one unified ownership constraint. That
is a structural risk, not a corpus-measured duplicate-label count.

**Recommendation:** keep overlap as strong evidence, not a separate irreversible
association policy. Model question→choice-group and option-caption→widget
separately. Constrain source text-span consumption once across all candidates.
Keep the successful F1120SO inline clauses as supported candidates.

### 5. Native identity and geometry disappear too early

`FieldValuePrediction` (`base_models.py:398`) does not retain widget index,
field name/identity, flags, or original native rectangle separately. At
`form_field_model.py:329`, the value bbox is replaced with the whole checkbox
cluster bbox. Later matching uses Python `id(value)` as temporary identity.
These choices prevent reliable downstream native-order checks, grouping by
native field, or obstruction tests against actual widget rectangles.

**Recommendation:** retain stable page-local native index and original bbox,
plus metadata needed for grouping/roles. Keep option-caption provenance
separate. Preserve `/AS` state precedence and lossless native values.
Native field-name equality can support grouping without displaying the name;
use actual field identity when available, since duplicate names need not prove
identity.

### 6. Table exclusion is partial and runs too early for semantic ownership

Pipeline order is `layout_postprocess → form_field → table → assemble`
(`standard_pdf_pipeline.py:790–792`). The current matcher has region bboxes,
not the completed grid. Center-in-smallest-table is useful as a preliminary
boundary, but cannot identify rows, columns, spans, or a cell-local form.
The early containment/checkbox routes also precede this boundary check.

**Recommendation:** keep native collection early, but finalize semantic
association after table structure exists and before assembly. A single final
ownership pass should resolve table cells and free form associations before
consuming labels. Do not delete text needed by the table model first. For
unresolved table-like regions, preserve widgets and abstain or mark structural
ownership unresolved; do not silently force a free-form key.

### 7. Binding and document reading order remain separate obligations

`page_assemble_model.py:198–333` materializes source-backed regions and unmatched
regions; `readingorder_model.py:561–592` emits each region's item list. Native
order survives within individual value accumulators, but widgets split between
inline items, FORM regions, and the unmatched region are not constrained by one
retained native sequence. The unmatched region still encloses all its widgets,
not tight key/value groups.

The F1040LEP export leaves unmatched right-column captions in the body before
the later block of bound left-column values. A correct link count would not
prove a correct document traversal.

**Recommendation:** preserve native order inside each resolved form scope and
let document reading order place the complete scope. Validate exported traversal,
not only prediction records. Do not add a page-level “form-dominated” override
as a substitute for resolving ownership.

### 8. Performance and tests do not justify the existing claims

The matcher is O(WL), not a linear one-pass algorithm; table partitioning adds
O((W+L)T), and containment scans add O(W(F+L+C)). These sizes are small here and
the measured stage cost is negligible relative to conversion. Avoid an index,
cache, GPU path, or elaborate solver framework unless larger workloads justify it.

The ten matcher tests exercise selected synthetic rules. The table test copies
the partitioning logic into the test, and the shared-header test codifies the
unsupported 1:many case as expected failure. All 16 current tests pass despite
the observed errors. Add image-derived integration regressions rather than
further tests that only restate the heuristic.

## Recommended optimization model

### Inputs and ownership

Use layout text spans, native widget rectangles/indices/types, container
relations, and completed table cells. Keep native values and visible text
lossless until a final assignment is selected.

Each widget has exactly one primary destination:

1. A field item with a visible key, possibly shared with sibling values.
2. A table cell, with row/column spans and references to header context.
3. An explicit unresolved/keyless item.

An option caption is not interchangeable with a group question. For a simple
choice group, existing `FieldItem(key=question, values=[checkboxes])` and nested
checkbox labels can represent both. For an option containing inline dates or
amounts, retain the option's own field item and its surrounding question context;
do not flatten all clauses into one undifferentiated key.

Table cells can themselves contain a local key/value or choice group. Solve that
inside the cell after cell ownership; never allow two widgets in different cells
to share a key merely because both are inside one TABLE bbox. If table structure
is absent, report unresolved structure separately from ordinary key abstention.

### Candidate groups

A candidate association is `(key span or span bundle, widget group, role)`.
Include singleton fields, native-supported choice groups, composite fields
such as Business Number, and inline clauses. Widget groups should be contiguous
in their **local filtered native sequence**, not necessarily globally contiguous:
RF1177S's top choices are separated by table widgets in the page sequence.

Require positive structural evidence before proposing a multi-value group:
shared native field identity, a bounded common prompt/option block, repeated
choice structure, or an inline clause. Merely being adjacent checkboxes is not
enough. Penalize unsupported grouping; otherwise one huge group can evade
crossing and alternation costs. Cap candidate group sizes from observed local
structure, not a fixture-specific number such as five.

### Constraints and objective

For candidate groups g, let x_g be binary, and let z_i denote abstention for each
non-table widget i. Use:

`sum(x_g for g containing widget i) + z_i = 1`.

Each leaf label span can be consumed once as a key/option caption. A common
question is consumed once by its group, not repeatedly copied onto its values.
Conflicting span bundles, crossing cell boundaries, and overlapping primary
widget ownership are hard constraints. Native widget order is an invariant of
output, not an option the optimizer can change to improve its score.

Minimize a scalarized multi-objective cost comprising:

- Local label/group distance and alignment along the fixed value sequence.
- Intervening **unrelated** widget obstacles and hard separators.
- Label-order disagreement with the local widget traversal.
- Violations of expected `key → value-group` alternation.
- Label-role uncertainty and unsupported grouping.
- Inconsistency with the local block's above/below/left/right convention.
- Abstention cost; optionally a small reward for corroborating native metadata.

Normalize units before tuning. Use local line height h for
`log(1 + edge_gap/h)`, bounded [0,1] alignment/role/style terms, and per-transition
order/alternation terms. Compare scores on the same fixed eligible units; do
not divide by the number of links an optimizer elects to emit. Account for
multi-value group size explicitly so neither large groups nor many singleton
groups receive a free scoring advantage.

### How the user's three objectives should work

**Geometric obstruction:** connect facing label/widget boundary anchors rather
than bbox centers; a center line through a large paragraph is misleading.
Count interior intersections with unrelated native widget rectangles, excluding
the target and other values in the candidate's own legitimate group. Use a small
text-scale tolerance for touching borders. A checkbox group is not exempt from
crossing another question's values. Hard cell/separator crossings should normally
be forbidden; uncertain geometric obstacles should be penalized.

Also distinguish a link passing through a widget from two links intersecting.
The first is usually a unary candidate cost once group membership is known; the
second is a pairwise interaction. RF1084S's wrong adjacent-column caption can
have no intervening obstacle at all, so obstruction alone cannot fix it.

**Order agreement:** use the fixed sequence `v1, v2, ..., vn` as the backbone.
The unknowns are each value's label and the boundaries/membership of semantic
groups. A reset to the top of a second column is valid because the supplied
sequence says so. Evaluate label transitions and intervening content between
successive values in that sequence. Layout hierarchy, lanes, and repeated
structure help locate labels; they must not propose alternative value orders.
Where label placement is uncertain, compare alternative label assignments or
local label-side conventions while holding the value sequence unchanged.

**Alternation:** reward `K, V+` at field-group level, and option-caption/checkbox
adjacency inside choice groups. Treat below-field captions as a legitimate local
style as well. Table cells are excluded from free-form alternation scoring.
Alternation is weak supporting evidence: a wrong but regularly alternating
assignment is still wrong. Avoid double-counting the same signal through both
order and alternation weights.

### Starting weights, not claimed optima

The following is a concrete initial scale for development. Recalibrate jointly
with null cost after candidate/group construction is stable.

| Term | Initial weight | Calibration intent |
|---|---:|---|
| Log distance in local line heights | 1.0 | Reference scale; no universal two-line hard cutoff |
| Misalignment | 0.5 | Prefer facing/shared bands, tolerate offset/multiline labels |
| Wrong label role / unsupported grouping | 3.0 | Units, leaders, and question-vs-option confusion must not win solely by proximity |
| Unrelated-widget obstruction | 4.0 per clear obstruction | Usually prefer abstention to a link through someone else's field |
| Label transition inconsistent with fixed value order | 3.0 per incompatible transition | Evaluate on the supplied sequence, never on global y-sort |
| Group-level alternation violation | 0.5 | Supporting regularity, not a binding mandate |
| Local style inconsistency | 1.0 | Infer direction per block; retain multiple hypotheses if uncertain |
| Corroborating metadata agreement | at most 0.5 reward | Never override visible evidence or create visible text |
| Abstention | 2.5 per eligible assignment unit | Tune for precision/coverage; account for group cardinality explicitly |

“Wrong role” and “unsupported group” need separately logged feature values even
if initially sharing a coefficient. A group of n widgets must be compared with
n appropriately weighted null outcomes; use per-widget local costs plus a
once-per-group structure prior. Option and common-question stages should have
separate denominators. This avoids making grouping attractive simply because it
pays one distance cost for many widgets.

These values encode priorities, not measured superiority. In particular, the
simple direction-weight experiment above does not validate the local style
feature in this table. A material link should beat abstention and have a margin
over the best conflicting assignment. Calibrate that margin on held-out forms;
do not call raw score differences probabilities.

### Solver choice

Start with a **small exact offline reference formulation** over sparse candidate
groups. A binary MILP expresses single ownership, shared keys, span conflicts,
and pairwise order/crossing costs without pretending this is ordinary 1:1
matching. Linearize a pair penalty with an auxiliary binary variable bounded by
`y >= x_g + x_h - 1`, `y <= x_g`, and `y <= x_h`.

SciPy already exists in `convert-core`. Its
[`milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
interface supplies integer variables, linear constraints, time limits, and solver
status/gap reporting. However the declared SciPy floor is currently **1.6**;
MILP was introduced in [SciPy 1.9](https://docs.scipy.org/doc/scipy/release/1.9.0-notes.html),
so using it in production would require an explicit
compatibility/dependency decision. Do not silently assume all supported installs
have it. Keep the first formulation an experiment until its value is established.

Ordinary
[`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
solves the additive 1:1 baseline, not arbitrary shared-label groups or pairwise
constraints. It was sufficient for the diagnostic ablation, not the target model.

A dynamic program is preferable for components that really reduce to ordered
sequences with local transition costs. But an arbitrary set of shared span
candidates and pairwise intersections does not become Needleman–Wunsch by naming
it that. A narrow diagonal band is also unjustified when keys and values have
different counts and many table widgets are filtered out. Validate candidate
recall before pruning; only derive a simpler production solver after comparing
it with the exact reference on saved components.

Generate sparse candidates and partition disconnected components first. Do not
build a dense page-wide pairwise tensor. Measure component size, solve time,
objective gap, and fallback frequency. On a timeout, retain feasible high-margin
assignments and abstentions; never drop widgets or silently assert optimality.

A grouping must not erase the supplied sequence. If values from different
semantic groups interleave, retain their fixed sequence and represent group
membership by references; do not reorder values to make a convenient tree.
Contiguous groups can use the existing nested field-item representation directly.

## Calibration and validation plan

1. **Freeze semantic annotations before tuning.** Extend the 102 local-link
   annotations to explicit question groups, option captions, composite values,
   and table row/column ownership. Include “label missing from extracted spans”,
   “ambiguous”, and “native widget has no visible counterpart”. Preserve the
   original screenshot-based judgment when the current algorithm disagrees.
2. **Measure candidate recall separately.** For every expected relationship,
   record whether the correct span/group/cell is even feasible. Manitoba page 1,
   RF1084S's distant labels, and split Arabic captions must not be treated as
   weight-tuning failures. Table routing needs its own confusion matrix.
3. **Use form-family splits.** Keep pages and related templates together. Leave
   one form family out during development and report macro scores as well as
   widget-weighted totals. Obtain fresh forms for final validation: these 11
   have already influenced earlier code and the present recommendations.
4. **Ablate each objective.** Compare current greedy, independent nearest,
   additive global assignment, structural grouping, obstacle/order factors,
   alternation, and optional metadata. Vary weights coarsely, e.g. 0.5×/1×/2×;
   reject settings that improve aggregate coverage by creating severe wrong
   links or sacrificing a whole form family. Keep value order fixed in every
   ablation; vary only how strongly candidate label transitions must align with it.
5. **Report the right outcomes.** Key-link precision/recall with abstention;
   exact group membership; option-caption accuracy; table ownership/row/column
   accuracy including spans; missing candidate rate; native value/state/index
   preservation; exported traversal; runtime and solver fallback. Report table
   and free-form metrics separately. The gold sample's 70/102 is not corpus
   accuracy, and a key for every widget is not the objective.
6. **Add meaningful regressions.** Cover F1040LEP's column reset and Chinese
   swap, F14446CN's second signer block, RF1084S's six header fields, GST494's
   shared Business Number, rc7190's two choice groups and line-number distractors,
   RF1125S's partial table detection, and existing F1120SO inline clauses.
   Check serialized DCLX/JSON structure as well as pure assignment results.
7. **Test generalism.** Uniformly scale coordinates, perturb boxes within
   realistic OCR noise, permute candidate enumeration order, split/merge label
   spans, and remove metadata. Separate layout failures from assignment
   instability. A fixture name, language-specific token, or special numeric
   code must never select an algorithm branch.

## Proposed implementation scope

Keep the work centered on `docling/models/stages/form_field/form_field_model.py`.
The unavoidable adjacent changes are:

- `docling/datamodel/base_models.py`: native identity/geometry and explicit
  internal ownership/group information that survives to assembly.
- `docling/pipeline/standard_pdf_pipeline.py`: final association after table
  structure, with native normalization retained where needed.
- `docling/models/stages/page_assemble/page_assemble_model.py` and
  `docling/models/stages/reading_order/readingorder_model.py`: materialize the
  selected groups/cell ownership once and preserve local native value order.
- Focused regression tests and the existing validation harness, including
  successful-status/page-count checks and replayable evidence.
- Potential docling-core coordination for a durable widget-to-table-cell
  reference if existing table/field representations cannot express it losslessly.
  Verify the actual contract before adding a new public schema.

Do not begin by rewriting extraction or the entire reading-order predictor.
Do not add a public configuration knob for each experimental objective.
First implement one inspectable offline formulation and its annotations, prove
which signals earn their complexity, then replace the heuristic with the
smallest production formulation that achieves the validated behavior.

## Validation commands and boundaries

The successful audit and tests used this process-local source override:

```bash
export DOCLING_DEVICE=cpu
export PYTHONPATH=/Users/cau/Documents/Development/docling_release:/Users/cau/Documents/Development/docling-core:/private/tmp/acroform-review-20260908/native
/private/tmp/acroform-review-20260908/venv/bin/python \
  output/acroform-keying-review-20260908/replay.py
/private/tmp/acroform-review-20260908/venv/bin/python \
  output/acroform-keying-review-20260908/ablate.py
/private/tmp/acroform-review-20260908/venv/bin/python \
  output/acroform-keying-review-20260908/check_counterexamples.py
/private/tmp/acroform-review-20260908/venv/bin/python -m pytest \
  tests/test_acroform_label_binding.py tests/test_form_extraction.py -q
```

The `/private/tmp` environment is session-local; the saved data and source
identities are the durable evidence. Recreate a compatible CPU environment if
it is gone. `run_audit.py` reruns conversion and writes into its evidence folder;
use a copy of that folder to preserve this baseline.

Final validation: **`make validate` passed** in an isolated changeset containing
this handoff and the ground-truth data. No hooks modified the deliverables.
Unrelated untracked files in the working checkout were excluded from mutating
hooks. The result is recorded in the evidence directory's `validation.log`.
Ground-truth integrity, exact 19-page replay, and the saved 102-link ablation
were also rerun successfully from their delivered paths.
