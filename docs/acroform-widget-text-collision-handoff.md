# AcroForm widget/layout-text collision — experiment handoff

Status snapshot: 2026-09-03

This documents a follow-up experiment run against the branch implementing
`docs/acroform-field-items-replacement-handoff.md` (Docling branch
`feat/acroform-field-items`, head `b54c2d0b` at experiment time). It is a
record of an observed regression, not an implementation plan. No production
code was changed while writing this.

## Question

The replacement design extracts fillable field values from native PDF widget
state (`/V`, `/AS`) rather than from rendered pixels. But a PDF's widget
appearance stream is still painted onto the page raster that the layout model
and OCR run against. What happens when a widget actually has visible filled-in
text, i.e. the two extraction paths — native widget state and visual layout —
both have something to say about the same screen region?

## Inputs (persisted, not in a session temp dir)

All experiment inputs and outputs are at
`/private/tmp/docling-acroform-collision-experiment/`, which is outside any
Claude session scratch directory and will not be swept:

```text
/private/tmp/docling-acroform-collision-experiment/
├── f1040s1_filled.pdf          # filled fixture (see below), sha256 3627057b6c7e58ad4abdd9b0dc9d6886f889de6504162cab53b2b88307b724ed
├── fill_irs_form.py             # script that produced it from the source fixture
└── outputs/
    ├── branch/                  # feat/acroform-field-items @ b54c2d0b results
    │   ├── f1040s1_filled.dclx
    │   ├── f1040s1_filled.json
    │   ├── f1040s1_filled_page1.png
    │   └── f1040s1_filled_page2.png
    └── main/                    # origin/main (no AcroForm support) results
        ├── f1040s1_filled.dclx
        ├── f1040s1_filled.json
        ├── f1040s1_filled_page1.png
        └── f1040s1_filled_page2.png
```

The unfilled source fixture is the same IRS Schedule 1 reproduction used by
the original handoff:
`/private/tmp/docling-acord-analysis/f1040s1.pdf`, sha256
`8dafec719f6a4716c259a2bdaca546d9bb9e262d1eabef885fe116a7327458fa`.

### How the filled fixture was built

`fill_irs_form.py` uses `pypdf` (`uv run --with pypdf`, not a project
dependency) to set values on all 68 `/Tx` fields (`"111.00"`, `"222.00"`, ...
incrementing) and check 2 of the 5 `/Btn` fields
(`topmostSubform[0].Page1[0].c1_1[0]`,
`topmostSubform[0].Page2[0].c2_1[0]`), with
`writer.update_page_form_field_values(page, values, auto_regenerate=True)`.

`auto_regenerate=True` is required: this fixture is an XFA form
(`pypdfium2.PdfDocument.get_formtype()` returns `3`,
`FPDF_FORMTYPE_XFA_FULL`), and without regeneration the written `/V` has no
baked `/AP` appearance stream, so nothing renders. With regeneration,
`pypdfium2` confirmed (after `init_forms()`, ignoring its XFA-unsupported
warning and falling back to plain AcroForm rendering) that the appearance
streams paint correctly. Separately, and more importantly:
**docling-parse's own page rasterization also paints the filled widget text**,
visible directly in `outputs/branch/f1040s1_filled_page1.png` — this is not a
pypdfium2-only artifact, it is what the layout model actually sees.

### Conversion settings

Both runs used the same `PdfPipelineOptions`:

```python
PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    extract_form_fields=True,       # branch only; option does not exist on main
    generate_page_images=True,
    images_scale=2.0,
    layout_options=LayoutOptions(create_orphan_clusters=False),
)
```

`DOCLING_DEVICE=cpu`.

## Results

### Branch (`feat/acroform-field-items`)

| | unfilled `f1040s1.pdf` | filled `f1040s1_filled.pdf` |
|---|---:|---:|
| field regions | 2 (one `FORM` cluster per page) | **7** |
| field items | 73 | 73 (unchanged — correct) |
| plain document text items that duplicate a fillable value's text | 0 | **60+** |

Region detail for the filled run:

- Page 1: 2 regions. One keeps `source_container_id` matching the original
  `FORM` cluster; the other has `source_container_id=129`, i.e. a different
  cluster than the page's single detected `FORM` — the filled visual content
  changed what the layout model segmented.
- Page 2: 5 regions (`source_container_id` = 107, 94, 98, 10, 29), where the
  unfilled run had exactly 1.

Duplicate text: values such as `"111.00"`, `"1554.00"`, `"1998.00"`, etc.
appear both as an ordinary text item in `doc.texts` (because the layout model
detected the rendered blue appearance-stream text as a normal `TEXT`/table
cluster) and as the `FieldValueItem` extracted from the same widget's native
`/V`. `outputs/branch/f1040s1_filled.json` and `.dclx` contain this
duplication as delivered; no suppression was attempted.

### Main (`origin/main`, no AcroForm support)

| | unfilled | filled |
|---|---:|---:|
| document text items | 72 | 196 |

No duplication is possible on main because there is no separate
widget-derived channel to collide with — the rendered appearance text is the
only representation of the field content. This is the expected, uninteresting
baseline; it confirms the collision is specific to running two extraction
paths (native widget + visual layout) over the same filled region, not an
artifact of the fixture itself.

## Diagnosis

Two distinct failure modes, both downstream of the fact that widget
appearance streams are part of the page raster the layout model consumes:

1. **Duplicate text.** The field-mapping stage (per
   `docs/acroform-field-items-replacement-handoff.md`, "Narrow duplicate
   suppression") only suppresses proven-equivalent `CHECKBOX_SELECTED` /
   `CHECKBOX_UNSELECTED` leaves against a native checkable widget. There is no
   corresponding suppression for text, and the handoff explicitly rejected
   attempting one ("Do not attempt text-field duplicate suppression by
   deleting overlapping text ... not enough evidence"). This experiment
   supplies a concrete case where that gap produces visible duplication, but
   does not by itself argue the rejection was wrong — a filled and an
   unfilled text field are visually indistinguishable to the layout model
   from ordinary printed text, so any suppression heuristic risks the same
   over-deletion risk the handoff was avoiding.

2. **Region fragmentation.** The "Region construction" matching (raw `FORM`
   cluster vs. widget bboxes, by strongest coverage) runs on layout output
   that already reflects the filled appearance text. When that text changes
   what the layout model detects as containers on the page, one raw `FORM`
   cluster can be perceived as several smaller clusters, so widgets that
   would otherwise all match one region instead scatter across multiple
   matched regions plus the fallback. Field *items* stay correct (73/73 in
   both runs — native order and count are preserved), but the region
   *grouping* becomes unstable and no longer reflects the form's actual
   layout structure. This is a more clear-cut defect than (1): the design's
   invariant "layout predictions themselves remain unchanged by field
   assembly" is preserved, but the invariant that a single retained `FORM`
   with native controls produces one corresponding `FieldRegionItem` is
   violated purely as a side effect of the widgets being filled in.

## Non-conclusions

- This does not show the branch's field-*item* extraction is wrong — item
  count, order, and values were correct in both the filled and unfilled runs.
- This does not reopen the "no text-field duplicate suppression" decision by
  itself; it is evidence for the region-fragmentation half of the gap, which
  is a narrower and more mechanical problem (cluster matching stability) than
  duplicate-content suppression (which requires provenance/semantic
  judgment).
- Not evaluated: whether disabling page-image generation, or otherwise
  keeping the layout model from seeing filled appearance streams, is a viable
  mitigation — that would presumably also blind normal OCR/layout to
  legitimately handwritten/typed content on scanned filled forms, which are a
  realistic real-world input for this feature.

## Possible follow-up (not scoped or designed here)

- Reproduce region fragmentation with a synthetic single-`FORM` fixture with
  filled vs. unfilled widgets, to isolate it from IRS-specific layout
  quirks, matching the existing "Multiple-region unit fixture" test style in
  the original handoff.
- Investigate whether region matching (`docling/models/stages/form_field/`)
  can key off the *raw* `FORM` cluster bbox pre-postprocessing consistently
  regardless of visual content, rather than depending on the postprocessed
  layout that filled-in text can perturb.

## Solution proposals (2026-09-03 follow-up analysis)

Added after re-examining the branch output in
`outputs/branch/f1040s1_filled.json`. Two independent fixes, one per failure
mode. Neither was implemented here.

### Idea 1 — text duplicate suppression (the safe rule the data supports)

Addresses failure mode (1), "Duplicate text".

The original handoff rejected text-field duplicate suppression for fear of
over-deletion (a filled and an unfilled text field are visually
indistinguishable from ordinary printed text). Re-measuring the filled fixture
shows a suppression rule that carries almost no over-deletion risk, because it
gates string equality on *geometric containment inside the widget rectangle*.

Measured on the 68 filled `/Tx` widgets in `f1040s1_filled.json`:

| Case | Count | Payload identical? | Geometry |
|---|---:|---|---|
| Clean twin | **61 / 68** | yes, byte-for-byte | rendered-text bbox **fully inside** the widget rect (containment > 0.6; zero partials) |
| Merged into label | **6 / 68** | **no** — value is a *suffix* of a larger line, e.g. `"7 Unemployment compensation. … 1221.00"`, `"( ) 1443.00"`, `"z Other income. List type and amount: 3885.00"` | line only partially overlaps the widget |
| Not detected as text | **1 / 68** (`5328.00`) | n/a | no overlapping plain item at all |

So the assumption behind any "identical + overlapping" dedup — identical text
payload *and* spatial overlap — **holds for 61/68 but is not universal**: ~10%
of the time the layout model glues the filled value onto the neighbouring
label cluster, so there is no clean twin to match.

**Rule.** Suppress a plain text item iff both hold:

1. its normalized text `==` a `field_value`'s normalized text, and
2. its bbox is contained in that field_value's **widget rect**
   (`intersection_over_self` > ~0.6).

On this fixture that removes exactly the 61 real duplicates with **zero** false
positives. The containment predicate is what makes it safe: an ordinary
printed label does not simultaneously sit inside a known widget rectangle and
equal that widget's native `/V`. Empty/unfilled widgets have `/V = ""`, match
nothing, and are inert. This is the same shape as the existing checkbox
suppression (string-equivalence gated by geometry), extended to text.

**Deliberately left behind** (worth a `ponytail:`-style note): the ~7
merged-into-label cases. There the value is a substring of a larger line;
excising just the suffix means editing text content mid-string, which is
exactly the risky over-editing the original handoff rightly avoided. Leaving
them cleans 90 %+ of duplicates; the residual is a trailing value on a label
line.

### Idea 2 — don't inflate a `FORM` bbox from loose text children

Addresses failure mode (2), "Region fragmentation".

**Root cause (more precise than the original diagnosis).** The fragmentation
is *not* the layout model emitting many overlapping forms that dedup fails to
collapse. Postprocessing already deduplicates overlapping `FORM` clusters:
`container_clusters` go through
`_remove_overlapping_clusters(..., "wrapper")`
(`docling/utils/layout_postprocessor.py:312`), merging any two forms with
`IoU > 0.8` **or** `containment > 0.8`. That dedup ran in the experiment branch
(`b54c2d0b`; the dedup landed earlier in `aaa4e28e`, 2026-08-25).

The overlap is created **after** dedup, by the child-enclosure bbox rewrite in
`_set_cluster_children` (`layout_postprocessor.py:404-410`): for
`CONTAINER_TYPES` it overwrites the cluster bbox with the enclosing box of its
assigned children. Sequence:

1. Dedup runs on the **compact detector** form bboxes → several distinct forms
   correctly survive (they do not overlap past threshold yet).
2. Regular text clusters are assigned as children (`> 0.8` containment).
3. Each form's bbox is rewritten to enclose those children.

On a filled page the appearance-stream text spawns many scattered `TEXT`
clusters; different surviving forms adopt different scattered children and each
balloons to a near-page-spanning rectangle. *Then* they overlap heavily —
measured on page 2: region #5 ⊂ #6 at containment 1.00, IoU 0.90 — but dedup
already ran on the pre-inflation boxes and had no reason to merge them. On the
unfilled page there is almost no interior text, the enclosure barely grows, and
forms stay compact and distinct. Confirmed the region bboxes are these
post-enclosure boxes, not the widget spans: region #5 encloses
`(39.5, 240.4, 527.1, 740.0)` while its two widgets span only
`(64.8…481.6, 276…300)`.

**Fix.** In `_set_cluster_children`, restrict the container bbox rewrite so it
encloses only **nested tables/pictures**, not loose regular-text children —
which is exactly what the enclosure was introduced for (#4064, "nest tables and
pictures in form regions"). Tables/pictures are stable layout objects, so the
box no longer drifts with rendered field text and the "one retained `FORM` →
one `FieldRegionItem`" invariant is restored at the source.

**Restrict to `FORM`, not all `CONTAINER_TYPES`.** The rewrite branch currently
fires for `CONTAINER_TYPES = {FORM, KEY_VALUE_REGION}`
(`layout_postprocessor.py:404`). For a **key-value region the text enclosure is
the point** — its bbox is meant to cover the key/value pairs, which are regular
text, and KVRs rarely contain tables/pictures. Applying the restriction to KVR
would shrink KVR boxes to nothing and regress key-value extraction. The change
must therefore be scoped to `DocItemLabel.FORM` only; KVR keeps the current
all-children enclosure.

**Downstream consequences (uncovered during analysis):**

- *No text is lost.* Forms never consumed their text children — the
  contained-child removal at `layout_postprocessor.py:187-193` applies only to
  `TABLE_TYPES`/`PICTURE` wrappers, never to `FORM`/`KVR`. Nested text remains
  as ordinary `TEXT` in the output regardless of the form bbox. Idea 1 is
  therefore orthogonal and unaffected.
- *Child assignment is unchanged.* Only the bbox rewrite input changes; nesting
  decisions are computed before the rewrite (against the detector bbox), so the
  same tables/pictures still nest and the parent/child structure is preserved.
- *Region completeness now rides on detector-box quality* — the one real risk.
  `_match_form` admits a widget only when it is `> 0.8` covered by the form
  bbox; the old text-enclosure could *rescue* a too-tight detector box by
  growing it over the field text. Removing that means a widget near a poorly
  drawn form edge could drop out of its form into the fallback region. In
  practice form detector boxes are drawn around their fields so this should be
  rare, but it is the thing to verify on the filled IRS fixture: **all 73
  widgets still match a form and the fallback region does not grow.**

**Suggested validation:** re-run the filled fixture with the FORM-only
enclosure change and compare against `outputs/branch/`: expect page-2 regions
to collapse from 5 toward 1, field-item count to stay 73/73, and no increase in
unmatched/fallback widgets.
