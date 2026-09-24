# Visually reviewed AcroForm fixture ground truth

Version 1.1, 2026-09-09. This dataset saves the judgments made while inspecting
all 19 page images of the 11 PDFs in `test_fixtures_15forms_thinned`.
**AcroForm widget order is fixed and authoritative.** Nothing here asks a future
evaluator to infer or reassess that order.

**Local-label review is now complete for the frozen Docling detections.**
`field_reviews.jsonl` records all 946 retained native widgets, in native order:

- **434** have a visually identified label or option caption.
- **8** have **no visible label**: unnamed multiplier/component boxes in the
  Manitoba calculation sheet. Arithmetic signs are not labels.
- **1** is **ambiguous**: GST111 page 1 widget 7 overlaps the Province control
  area, without a distinct visible role. It is not a negative target.
- **503** are **excluded table content**, not unreviewed pairing examples.

Every eligible widget was inspected on a source image with its native rectangle
and the detected FORM/TABLE boundaries. Text and coordinates from the saved
layout supply addressable anchors after the visual judgment; predictions and
tooltips were not used as the oracle. Manually placed anchors were additionally
checked in enlarged source-image crops.

The original `annotations.jsonl` remains unchanged as historical evidence. Its
group, scope and visual-table annotations are still partial:

- **147** visible label/option-caption links.
- **26** choice groups, composite fields, inline clauses, or form scopes.
- **18** table/structural regions; **5** have explicit cell membership, covering
  **128 native widgets** in **107 cells**.
- **5** historical review notes describing the earlier annotation gaps.
- **970** native widget reference records, including controls the baseline skips.
  These are identity/geometry references, not 970 semantic annotations.

The frozen 102-link sample used in the algorithm review is marked by
`diagnostic_102_sample: true`. Additional annotations were saved afterward and
were not included in that reported ablation score. There is no held-out test set.

## Files

- `manifest.json`: schema version, coordinate convention, fixture SHA-256 hashes,
  thinned-PDF page dimensions, and the supplied widget sequence for each page.
- `widgets.jsonl`: one native reference per line, keyed by
  `(fixture, page, widget_index)`, with original native bbox, type, and name.
- `annotations.jsonl`: one independent visual judgment per line, with `kind`
  determining its record shape. UTF-8 JSON Lines permits streaming, diffs, and
  future additions without rewriting a large nested document.
- `field_reviews.jsonl`: the authoritative complete local-label decisions for
  the pinned detection run. Each record gives `disposition`, `expected_label`
  when present, a reason, and detected form/table/cell membership. This supersedes
  historical individual `label_link` records for the current replay.

## Detection boundaries and review decisions

Detected TABLE and DOCUMENT_INDEX regions take precedence over a surrounding
FORM. A widget intersecting one is excluded unless wholly contained in one
unambiguous detected cell; a pairing then requires the label in that same cell.
The frozen cell boxes admit no such widget-level exceptions in this corpus.
Cell-local behavior is covered by the synthetic prototype regression test.

The cyan report regions show detected forms with detected tables cut out. Form
membership records the production form-stage threshold: more than 80% of the
native widget area lies inside the form. A native widget outside a detected form
remains eligible when it is outside tables; missing FORM detection is not a
reason to discard a native field.

For example, GST111 page 2 has a detected middle table and an undetected bottom
grid. Only the middle table is excluded. The bottom grid receives ordinary
local-label annotations: row captions identify amounts, and the printed
estimated-amount instruction identifies the grey checkboxes. Manitoba's grids
also have no TABLE detection. Their row captions are shared local-label targets;
column headings add context. This does not ask the optimizer to reconstruct
tables, and it does not certify full row/column semantic keys.

| `disposition` | Evaluation |
|---|---|
| `label` | Compare the chosen label with the visually recorded region; distinguish wrong pairing from missed label. |
| `no_visible_label` | No pairing is correct; assigning a label is wrong. The reason explains the absence. |
| `ambiguous` | Display the prediction and reason, but do not score it as correct or wrong. |
| `excluded_table` | Report separately from eligible fields and from annotation completeness. |

The manifest pins each reviewed snapshot and source image by SHA-256 as well as
the original PDF. Changing detections requires reviewing the changed eligibility;
the replay rejects stale hashes, changed scopes, duplicate or missing decisions.
These are detections from the saved baseline run, not a fresh conversion with a
reordered production pipeline.

Coordinates are **PDF points, top-left origin**, bboxes are
`[left, top, right, bottom]`, and pages are one-based positions **in the thinned
PDF**. The printed page number may differ. Before scoring, verify the fixture
hash; do not silently apply these native indices to a different PDF revision.

## Record meanings

| `kind` | Meaning and fields |
|---|---|
| `label_link` | `widget_index` has the visible label region in `expected_label.bbox`; `relation` distinguishes `field_key` and `option_caption`. This establishes local association, not necessarily complete semantic key text. |
| `group` | `widget_indices` share the specified `group_type`. `expected_label` supplies the common question, composite-field label, clause, or scope description. Members remain in their fixed native subsequence. |
| `table_region` | A visually identified table or region containing table/cell-local form structure, independently of TABLE predictions. `rows` and `columns` contain header descriptions. When `cell_mapping_status` is `annotated`, each cell gives zero-based `row`, `column`, and native `widget_indices`; multiple widgets can share a cell. |
| `review_note` | A reusable observation about ambiguity or missing annotation. It is not a positive or negative assignment target. |

A `choice_group` common question coexists with each option's `label_link`;
these are two different relations, not contradictory targets. `form_scope` is
only a container: it does not give all contained values one common key.
`inline_clause` groups a checkbox and its associated embedded text/date fields.
A composite field may comprise multiple native widgets, or one rectangle with
several printed date components.

For table cells, `widget_indices` membership is explicit. Operators, row codes,
and percent signs are context, not substitutes for the annotated row/column.
GST111 cells may contain both an estimated-amount checkbox and an amount field.
The Manitoba grid's last column has only its visible final-result cell annotated;
absence of another cell record does not assert that such a cell is impossible.

## What counts as ground truth

The expected relationships, group memberships, and table semantics originate
from direct inspection of source page images. Native geometry and initial label
anchor coordinates were then taken from the saved parser/layout snapshot to make
those judgments addressable. Table-region bounds and two Manitoba label regions
are approximate visual bounds. These coordinates are anchors, not pixel-perfect
polygon annotations.

`text_hint` helps humans identify a region. Most hints reuse extracted text and
are **not an exact-text oracle**: they can contain OCR errors, omitted fragments,
leaders, or extra formula text. `text_hint_source` states the distinction.
`source_cluster_id_at_review` is diagnostic provenance only; a future layout model
can split, merge, or renumber clusters without changing the correct relationship.
Neither tooltips nor the model's selected labels were automatically accepted as
truth. The reference widgets' native field names are identity aids, not labels.

For the historical file, **no annotation means unreviewed, not wrong, keyless, or negative.** In
particular, coarse table regions with `cell_mapping_status: not_annotated` do not
provide row/column ground truth. Do not report whole-corpus accuracy from these
partial group/table annotations. The present set does not label value correctness, hidden
annotation validity, full question hierarchy, or exact complete text everywhere.

## Reuse for validation

1. Verify SHA-256 and page dimensions, then identify native widgets by their
   recorded page-local index. Confirm their reference geometry/type. Never sort
   them geometrically to reconstruct their order.
2. Resolve predicted keys/option captions to the expected spatial text region,
   allowing text splitting/merging. Use actual source spans when available.
   Require meaningful coverage and semantic role: a large paragraph bbox merely
   touching the expected label is not sufficient. Treat exact text completeness
   as a separate metric until it has its own annotations.
3. Score local label links separately from common-question membership, composite
   group membership, and table-cell membership. A correct option caption with a
   missing common question is not a completely correct choice group.
4. Count wrong assignments, missed labels, correct no-label abstentions and
   incorrect labels on keyless fields separately. Ambiguous cases and table
   exclusions do not enter the scored denominator. A missing record in the
   complete review file is an integrity error.
5. For native indices missing from current serialized output, instrument the
   extraction/association boundary or match reference native rectangles uniquely.
   If enlarged checkbox bboxes prevent unique identification, mark the case
   unscorable; do not infer an identity by arbitrary nearest-neighbor matching.
6. Keep future corrections explicit, with an annotation version and reason.
   Do not overwrite ground truth to agree with a new model prediction.

The corresponding analysis is
`docs/acroform-keying-algorithm-review-handoff.md`. Successful CPU conversion
archives, source images, frozen snapshots, and diagnostic replay scripts live in
`output/acroform-keying-review-20260908/`. These paths are relative to the repo
root. The original PDFs remain external; their default location is a hint in the
manifest, not a required installation path.

Integrity check (stdlib only; accepts an alternate PDF directory):

```bash
python3 output/acroform-keying-review-20260908/validate_ground_truth.py \
  --fixtures /path/to/test_fixtures_15forms_thinned
```

The integrity check validates hashes, identities, coordinate bounds, ordered group
membership, and table references. It does not substitute for semantic scoring of
Docling output.

The complete review file, detection scopes and snapshot/image hashes are checked
by the current replay:

```bash
uv run --no-sync python -m scripts.replay_acroform_keying
uv run --no-sync python -m pytest tests/test_acroform_keying.py -q
```
