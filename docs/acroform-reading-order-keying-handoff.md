# AcroForm reading-order & widget-to-key association — design handoff

Status snapshot: 2026-09-04
Base branch: `feat/acroform-native-fields`, head `505a96be`.
**Phase 1 is now BUILT & VALIDATED on branch `feat/acroform-label-binding`
(stacked on the base). See §5 for the outcome; §3's `_precedes` carries a
validated correction (row-only guard, no left-order gate).**
Working tree adds `scripts/validate_acroform_forms.py` (the validation harness,
see §6) — commit or keep it alongside this work.

**2026-09-04 revision:** §3 was reworked from a scored multi-direction search
(six knobs) into a single-metric **order-preserving alignment** — the objective
is "minimize global ordering deviation", not distance. §4/§5 updated to match,
plus a validation fence (§4) that holds the RO model off form-enclosed elements.

**Fixtures** (15 forms, used for all validation here) live OUTSIDE the repo at
`/Users/cau/Documents/Data/DocLayNet_v3_campaign_data/test_fixtures_15forms_thinned`
(override with env `DOCLING_FORM_FIXTURES` or `--fixtures`). Details in §6.

This is a **design handoff** (ideas + phased plan), not yet an implementation.
It extends `docs/acroform-keyed-field-item-handoff.md`, which solved keying for
the case where a widget **overlaps** its label cluster (inline paragraph,
checkbox cluster). This doc covers the two things that still do not work:

1. **Widget → key binding when the key label does NOT overlap the widget** —
   the common case (label left of / above / below the field). Reading order is
   messy as a result.
2. **Following the AcroForm-indicated field order** as a reading-order signal.

Both are inherently heuristic: there is no reliable link in AcroForm between a
value widget and the layout element that labels it. `/TU` tooltips are empty,
`/T` field names are mostly mangled identifiers. So this is a heuristic design;
it must be calibrated and validated against the fixture set (§6), never merged
on a single example.

## 1. Empirical findings (measured on the fixtures)

Probed `parsed_page.widgets` (native `.index` order) on `gst494-fill-09e`,
`rf-1084s`, `f1120so` via the docling-parse backend:

| Signal | Verdict | Evidence |
|---|---|---|
| **`widget.index`** (AcroForm/tab order) | **Strong. Effectively already reading order.** | Widgets emerge monotone top→bottom; left→right within a y-band. gst494: y=744 row (#3,4,5 @ x=25,389,533), y=721 row (#6,7,8), 6-cell y=672 row (#9–14 ascending x), checkbox row y=624. rf-1084s interleaves its two columns correctly (left field then right field per row). `page.parsed_page.widgets` is already in this order — asserted by `tests/test_form_extraction.py::test_docling_parse_exposes_complete_widget_contract_in_native_order`. |
| **`widget_description` (/TU)** | **Dead. Empty on every widget, every fixture.** | Drop as a signal. No code should depend on /TU. |
| **`widget_field_name` (/T)** | **Unreliable — do not display as key.** | Mix of meaningful (`LegalName`, `ContactPerson`, `Title`) and garbage (`fill_21`, `Number only2222`, all f1120so names truncated to the same `topmostSubform[0].Page1[0].f…` XFA prefix). Weak tiebreaker at most. |

Headline: **the ordering sub-problem (2) has a clean deterministic answer**
(`widget.index`); the heuristic risk lives almost entirely in the **binding**
sub-problem (1).

## 2. Root cause of the messy order (traced)

The mess is **not** wrong ordering — it is that the widget half and the label
half are never joined into one bbox-coherent element:

- Labels flow through the layout reading-order model as ordinary
  `TextElement`s (`docling/models/stages/reading_order/readingorder_model.py:501`).
- A widget with no overlapping FORM/text cluster is dumped into **one
  page-wide unmatched `FieldRegionPrediction`**
  (`docling/models/stages/form_field/form_field_model.py:272`) whose bbox spans
  the page. The RO model then places that giant region by its bbox — nowhere
  near the labels.

**Important correction (from review): the page-wide dump is the extreme case,
not the whole problem.** Even when a widget *does* land in a real `FORM`
cluster (`_match_form`, `form_field_model.py:116`), the order is still messy:
the FORM region is emitted with `FieldItemPrediction(values=[value])` — **no
key** (`form_field_model.py:246-253`) — and the label text stays a separate
body `TextElement` ordered independently. So the widgets inside a FORM region
are value-only and detached from their labels just like the dumped ones. **The
binding fix (§3) must therefore run for FORM-matched widgets too, not only for
the unmatched fallback.**

The `overlapping` case already shows the shape of the fix: bind key+values into
one `field_item` and **drop the consumed label cluster from the body** (inline
path `form_field_model.py:230-239`; checkbox promotion via `promoted_cluster_ids`
+ `_drop_clusters`). The non-overlapping case needs the same, with a geometric
label search instead of a containment test.

## 3. Sub-problem 1 — widget → key binding as order-preserving alignment

**The key can appear left of, above, OR below the widget** — left is the classic
`key : [value]` (dominant in gst494's contact rows), above is a column/section
label over a field, below is the underline-style caption under a rule. An
earlier draft scored all three directions independently, each gated by its own
band-overlap fraction, slack, and a `left > above > below` preference tiebreak.
**Rejected: six hand-tuned knobs that were all faking one principle — respect
reading order.** Encode that principle directly and the knobs disappear.

### Reframing: it is an assignment problem, not N independent decisions

Binding `label A → widget w1` consumes `A`, so `w2` can no longer use it. And a
label that is *close in distance* but *two rows up in reading order* is the
wrong label no matter how small the gap. So this is a coupled assignment, and
the objective is not raw distance — it is **minimize global ordering
deviation**: the good assignment is the one under which the widgets and their
bound labels agree on order.

Two sequences are each individually reliable:

- **widgets in `widget.index` order** — proven effectively reading order (§1);
- **label clusters** (`TEXT_ELEM_LABELS`) — orderable top→bottom, left→right.

Binding is a **monotonic (order-preserving) matching** between them. Cost of
matching `wi ↔ Lj` = the geometric edge-gap; the structural constraint is **no
crossings** (if `w1` binds `A`, then `w2` may only bind a label at or after `A`
in reading order). Minimizing total gap subject to no-crossings *is* "minimize
global ordering deviation" — the two are the same objective. This structurally
removes the "widget grabbed a label from the wrong row" error class that no
distance threshold can catch.

A widget that needs no label falls out for free as a **skip** (cost = the gap
cap): a standalone tabular field, or one already enclosed by a label — the
latter never reaches this stage, the overlapping `_match_text_container` path
(doc 1) consumes it first.

### The one metric — rectangle edge-gap

```python
def _gap(w: BoundingBox, c: BoundingBox) -> float:
    # top-left origin: `t` is the upper edge (smaller y), `b` the lower.
    dx = max(0.0, w.l - c.r, c.l - w.r)   # 0 when they share a vertical band
    dy = max(0.0, w.t - c.b, c.t - w.b)   # 0 when they share a horizontal band
    return dx + dy
```

This single number *is* the direction logic, for free: a label directly left
shares a horizontal band (`dy=0`, gap = horizontal spacing); a label above/below
shares a vertical band (`dx=0`, gap = vertical spacing); a diagonal distractor
has both nonzero and so scores worse and is deprioritized. No per-direction
predicate, no band-overlap fraction, no preference order. Alignment is not a
hard gate — a slightly offset label just scores slightly worse, which is more
robust than a 0.5-band cutoff, not less.

**One quality knob:** the cap. Bind only if `gap ≤ _LABEL_GAP_CAP_LINES ×
median_line_height` — a text-scale bound, *not* a page fraction, so a field with
no nearby label stays keyless instead of grabbing text across the page. Start
`_LABEL_GAP_CAP_LINES ≈ 2` and let the fixtures (§6) calibrate it. This is a
legitimate calibration knob (real-world text scale), not a crude threshold.

### How much machinery — greedy first, DP only if earned

1. **Monotonic forward greedy (start here).** One pass in `widget.index` order.
   Each widget takes the nearest unconsumed label at or after the last binding
   in reading order, within the cap. This is the banded DP without the
   backtracking table: it captures the no-crossing coupling, works on a local
   neighbourhood, needs no cost matrix, and needs only geometry + `widget.index`
   — no dependency on a global label linearization.

   ```python
   def _match_labels(
       widgets: list[tuple[int, BoundingBox]],  # (widget.index, bbox), in index order
       labels: list[Cluster],                   # unconsumed, TEXT_ELEM_LABELS
       line_height: float,
   ) -> dict[int, Cluster]:                      # widget.index -> key cluster
       cap = cls._LABEL_GAP_CAP_LINES * line_height
       bound: dict[int, Cluster] = {}
       used: set[int] = set()
       frontier: BoundingBox | None = None       # last bound label -> no crossing
       for index, w in widgets:
           best: tuple[float, Cluster] | None = None
           for c in labels:
               if c.id in used or (frontier is not None and _precedes(c.bbox, frontier)):
                   continue
               g = _gap(w, c.bbox)
               if g <= cap and (best is None or g < best[0]):
                   best = (g, c)
           if best is not None:
               bound[index], frontier = best[1], best[1].bbox
               used.add(best[1].id)
       return bound
   ```

   `_precedes(c, frontier)` = `c` sits before `frontier` in reading order. This
   crossing guard is the one subtle spot and the first thing to sharpen against
   the fixtures.

   **Validated correction (2026-09-04): the guard is row-only — no left/right
   gate.** The first draft made `_precedes` true for "same row-band and further
   left". On gst494 that backfired: binding a right-side label (a trailing
   `RT` in the Business Number field) moved the frontier to the far right, and
   every left-side label in the *next* row (`Contact person`, `Title`,
   `Telephone number`, all gap 0) was then read as "same row, further left" =
   preceding = blocked. Section A bound nothing. The fix is to gate the guard on
   **vertical order only** — block a candidate solely when its y-center is a
   strictly higher row than the frontier — and make **no left-to-right
   assumption at all**. The two invariants that remain are exactly: (1) no
   crossing (vertical monotonicity), and (2) 1:1 pairing, which the per-label
   `used` set enforces on its own (one label per widget, one widget per label).
   After this correction gst494 binds 11/12 labels correctly.

2. **Banded DP** — only if the fixtures show greedy's local optima mis-bind (an
   early locally-cheapest binding that steals the next widget's real label).
   Then a Needleman–Wunsch alignment over the two sequences, **banded** to a
   local window around the diagonal, recovers the globally-cheaper assignment:
   O(n · band), not O(n·m). Same cost function, same single cap. Earn it; do not
   assume it.

On a bind: emit a keyed `FieldItemPrediction` (`key_text`/`key_bbox` already
exist on the model, materialized at `readingorder_model.py:575`) and add the
label cluster to `promoted_cluster_ids` so `_drop_clusters` removes it from the
body. **No new plumbing** — reuses the overlapping-case machinery.

### This unifies binding and reading order

§4 was framed as "bind keys, *then* order by `widget.index`." The alignment
shows they were never separable: its objective *is* ordering agreement. The
monotonic match binds keys and confirms `widget.index`-as-reading-order in one
pass. A cheap alignment (low total gap, no forced crossings) is itself the
evidence the order is trustworthy; an expensive one flags the page where §4's
explicit `widget.index` ordering should take over.

### Known ceilings (flag, don't solve yet)

- **Shared label ↔ many widgets**: a column header (`Amount`) over N number
  fields, or a section header over a checkbox row. Table-shaped, the genuine
  hard case. Phase-1 policy: monotonic match is 1:1, so the header binds one
  field and the rest stay keyless. Matrix/column association is a later phase,
  gated on a fixture that needs it.
- **Bordered table-forms — column crossing (validated on rf-1084s, ACCEPTED
  Phase-1 ceiling).** A field sandwiched between its own left label (`Namma`)
  and the next column's label (`Organisašunnummar`) binds the geometrically
  *nearer* one — the wrong column's — because in a bordered form the true labels
  are table cells, thin in the free-text label pool, and pure edge-gap has no
  column signal. rf-1084s bound 4/63 with such crossings. This is **not** fixable
  by the guard, and a left-preference tiebreak is exactly the assumption §3
  removed. It needs the table/border cell structure fed into the label pool —
  Phase 4. Decision (2026-09-04): accept and document; do not add a directional
  tiebreak to paper over it. The bindings that *are* made across the corpus are
  mostly correct; the mis-binds concentrate in bordered multi-column blocks.
- **The banded DP was not needed** for the greedy's local optima on the fixtures
  probed; the mis-binds above are a missing-signal problem, not a local-optimum
  problem, so Phase 2 stays unbuilt.

## 4. Sub-problem 2 — reading order from AcroForm order

§3's alignment already carries the order signal (it *is* an order objective), so
this sub-problem shrinks to "when do we let `widget.index` override the RO
model's geometry threading." `widget.index` is the spine. Earn the override, do
not assume it:

- **Lazy first cut**: once §3 produces keyed field_items with tight bboxes
  (label ∪ widget), let the **existing** RO model thread them in by geometry.
  Much of the mess was caused by page-spanning / detached bboxes; fixing the
  binding may fix the order for free. Validate before building more.
- **If that still mis-threads** (dense forms are where the RO model is
  weakest): order field_items by `widget.index` — group widgets into rows by
  index-run + y-band and emit in index order. Gate this on a **per-page
  "form-dominated" flag** (many widgets / high widget-area coverage) so prose
  pages keep the normal RO path. Do **not** replace the RO model globally.

Expose the order authority as a **calibration knob** (`index` / `geometry` /
`auto`): AcroForm tab order is author-controlled and a bad export (XFA) can
scramble it. The fixtures set the default.

### Validation fence: keep the RO model off form-enclosed elements

To judge the binding in isolation, **disable the reading-order model from
re-ordering anything enclosed in a form cluster** while validating §3. Right now
the RO model threads the widgets' labels as ordinary body `TextElement`s and its
geometry ordering muddies the picture — we cannot tell a binding mistake from an
RO-threading mistake. Fence it off: elements consumed into a field_item (labels
dropped via `promoted_cluster_ids`) already leave the body; additionally hold
the field_items themselves at their `widget.index`/alignment order and let the
RO model place only the surrounding prose. This is a temporary validation gate,
not the final §4 policy — but it is what lets the fixtures attribute a wrong
result to the binding rather than to interference.

## 5. Phased build plan

- **Phase 1 — BUILT & VALIDATED (2026-09-04), branch `feat/acroform-label-binding`.**
  `_match_labels` monotonic order-preserving binding (§3), 1:1, greedy forward
  pass, drop bound label from body, applied to **both** FORM-matched and
  unmatched widgets. Pure core + unit tests in `tests/test_acroform_label_binding.py`;
  wired into `form_field_model.py::__call__`.
  - Two bugs caught only by inspecting output-vs-image (tests passed throughout):
    the keyless guard tested `checkbox_label is None` but the field defaults to
    `""`, so the pass was dead; and `_precedes` had a left-order gate that
    poisoned the frontier (see §3 correction). Both fixed.
  - Outcome on the corpus: single-column / label-adjacent forms bind cleanly
    (rc7190 19/19, f1120so 21/21, gst494 11/12 correct). Bordered table-forms
    column-cross (rf-1084s, accepted ceiling above). Coverage varies mostly
    because dense grids are legitimately keyless (Phase 4).
  - The RO fence (§4) was **not** needed to validate — inspecting the keyed
    field_items directly in the `.dclx` against the page raster was enough.
  - Not the earlier multi-direction scored search — that was six knobs faking
    one order principle; the alignment encodes the principle directly.
- **Phase 2** — only if Phase-1 DCLX shows greedy's local optima mis-bind:
  upgrade the greedy to the **banded DP** (§3), same cost function and cap.
- **Phase 3** — only if Phase-1/2 DCLX shows order still wrong: `widget.index`
  ordering behind the form-dominated per-page flag + knob (§4).
- **Phase 4** — only if a fixture needs it: shared-label / column-header /
  matrix association.

Dropped for good: `/TU` tooltips (empty), `/T` field name as displayed key
(garbage).

## 6. Validation harness — `scripts/validate_acroform_forms.py`

**Read this so no future session re-establishes the setup.**

- **Fixtures** (15 forms; multi-country, single/multi-column, checkbox-heavy,
  prefilled) live OUTSIDE the repo at
  `/Users/cau/Documents/Data/DocLayNet_v3_campaign_data/test_fixtures_15forms_thinned`.
  Override with env `DOCLING_FORM_FIXTURES=/path` or `--fixtures`.
- **What it does**: converts each fixture with `extract_form_fields=True` +
  `generate_page_images=True` (`images_scale=2.0`) and writes
  `<out>/<name>.dclx` (+ `.md`). Default out: `<repo>/scratch_acroform_out/`.
- **DCLX = doclang archive**: a zip of `document.xml` (doclang) + `pages/N.png`
  (the page raster). Confirmed contents on gst494:
  `document.xml`, `pages/1.png`, `assets/*.png`. Open a `.dclx` in the doclang
  viewer — each field_item's key/value and prov box overlays the rendered page,
  which is the only reliable way to judge whether a widget bound to the right
  label. `.md` is a quick text-order sanity check on the side.

Run:

    uv run python scripts/validate_acroform_forms.py                 # all 15
    uv run python scripts/validate_acroform_forms.py --only gst494   # subset
    uv run python scripts/validate_acroform_forms.py --out before    # baseline

**Before/after loop** for each phase:

1. On the current head (no new binding), run `--out before/` → baseline DCLX.
2. Implement the phase, run `--out after/`.
3. Open matching `before/<name>.dclx` vs `after/<name>.dclx` in the viewer;
   compare key binding + reading order on the page image. Judge on the images,
   not the markdown.

The harness itself is validated (converts gst494, emits a well-formed archive
with an embedded page raster).
