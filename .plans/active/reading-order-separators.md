# Separator-aware rule-based reading order

**Status:** implemented as an experimental, opt-in PDF pipeline feature.

## Implementation result

`PdfPipelineOptions.use_reading_order_separators` enables separator-aware
ordering and defaults to `False`. When enabled, page preprocessing retains
visible shape geometry in private, non-serialized `Page` attributes before the
backend is released. No shape-query overhead is added while the option is off.

The CLI exposes the option as
`--reading-order-separators/--no-reading-order-separators`. For controlled
comparisons, `--output-file PATH` can name the exact result when one input and
one output format are selected.

The implementation uses both visible stroked segments from `get_shape_lines()`
and connected visible-shape boxes that are themselves thin enough to be filled
rules. The latter refinement is required by `Elsevier.pdf`, whose important
horizontal rules are filled paths rather than strokes. It does not synthesize
rectangle edges from a region bbox.

Candidates are clipped, length-filtered, merged, rejected when they cross text
or sit inside a graphic, and required to have content on both relevant sides.
Horizontal separators enter the graph as transient barrier nodes; vertical
separators constrain same-band links and horizontal dilation. All separator
nodes are removed before captions, footnotes, merges, or document construction.

Page-1 evaluation produced:

- `Elsevier.pdf`: seven accepted horizontal separators. The abstract now
  precedes `1. Introduction`, correcting the baseline's left-column-first jump.
- `sunday.pdf`: 12 horizontal and 13 vertical separators. A spurious merge
  across neighboring lower-right columns is removed.
- `2603.16056.pdf`: one accepted horizontal separator. Output order is
  unchanged.

The comparison Markdown and accepted-separator overlays are under
`debug/reading_order_separators/`.

## Objective

Use visible horizontal and vertical page rules to improve rule-based reading
order, especially for pages containing column separators, ruled metadata
regions, and rectangular frames.

Separator geometry must remain an internal ordering signal. It must never
appear in the final `DoclingDocument`, caption/footnote relationships, or
merge results.

## Current-state findings

1. `PdfPageBackend.get_shape_lines()` already provides an optional backend
   contract for visible, axis-aligned stroked segments.
2. `ThreadedDoclingParsePageBackend` implements that contract and returns
   top-left-origin degenerate bounding boxes.
3. Docling-parse filters strokes using their paint visibility and clip state.
   This is substantially safer than consuming unfiltered shape bounding boxes.
4. Pypdfium currently exposes connected shape bounding boxes, but cannot
   reliably determine clipping, transparency, or individual line segments.
5. Page backends may be released before document-level reading order runs.
   Separator geometry therefore has to be captured during page preprocessing.
6. Adding line boxes to `ReadingOrderPredictor` unchanged is insufficient:
   - a page-height vertical line is normally an isolated graph head and does
     not interrupt above-to-below relationships;
   - horizontal lines can become common graph nodes and unintentionally join
     otherwise independent columns;
   - assigning an ordinary `DocItemLabel` to a separator risks contaminating
     caption, footnote, and merge processing.

## Scope

The first implementation will use native visible vector lines from
`get_shape_lines()`. Raster/Hough detection is a separate, later phase.

Initially apply separators only while ordering root page elements. Container
children retain their existing local ordering, since whole-page separators
must not be injected into an unrelated nested sibling group.

## Phase 0: establish acceptance cases

1. Add a reduced synthetic reproduction of the supplied journal page:
   full-width title material, side-by-side article-info and abstract regions,
   a horizontal section boundary, and two-column body text.
2. Add cases for:
   - a vertical separator between two columns;
   - a horizontal separator between full-width bands;
   - a visible rectangle whose edges yield line segments;
   - a line crossing a text box, which must be rejected;
   - dense table ruling, which must not influence page-level reading order;
   - decorative header/footer rules.
3. Add the source PDF corresponding to the screenshot as an integration
   fixture if licensing and repository size permit. Otherwise, create a
   minimal generated PDF reproducing its geometry.
4. Record baseline output from the current predictor before introducing
   separator behavior.

## Phase 1: capture separator candidates

1. Add a transient, non-serialized field on `Page` for optional shape lines:
   - `None`: the backend cannot provide trustworthy line geometry;
   - empty list: the backend inspected the page and found no lines;
   - non-empty list: visible axis-aligned segments in top-left page space.
2. Populate it in page preprocessing while the page backend is available.
3. Prefer `get_shape_lines()`, and additionally accept a connected visible
   shape bbox only when the bbox itself is line-like. Do not infer the edges of
   a non-line-like connected region.
4. Preserve existing behavior for unsupported backends and non-PDF inputs.

## Phase 2: normalize and validate candidates

Implement a small separator-normalization component that:

1. clips segments to page bounds and removes zero-length segments;
2. classifies orientation explicitly;
3. merges collinear overlapping or nearly adjacent fragments;
4. deduplicates coincident rectangle edges;
5. removes segments below page-relative minimum lengths;
6. rejects segments crossing the interior of text elements;
7. rejects rules contained in table or picture regions;
8. requires structural support on the relevant sides:
   - text on both left and right for a vertical separator;
   - text both above and below for a horizontal separator.

Thresholds must be page-relative and derived from the acceptance corpus rather
than copied from raster-pixel assumptions.

Add a debug overlay showing accepted and rejected candidates with rejection
reasons.

## Phase 3: integrate explicit separator semantics

Keep content elements and separators distinguishable inside the predictor.
Do not assign separators a dummy document label.

Extend `predict_reading_order()` with an optional separator input and ensure
that it returns content elements only.

Prototype and compare two strategies:

### A. Synthetic-node baseline

Insert tagged separator nodes, run the existing graph construction, and remove
them before returning. This directly tests the original proposal and provides
a measurable baseline.

This strategy is not expected to handle vertical separators correctly without
additional graph semantics.

### B. Separator-aware graph constraints

- Vertical separators prevent same-band links and horizontal dilation from
  crossing the separator. They also define lane boundaries used when ordering
  otherwise independent graph heads.
- Horizontal separators act as ordering barriers: content supported above the
  rule must complete before supported content below it. They must not act as
  ordinary content nodes that join unrelated columns.
- Intersections between accepted horizontal and vertical rules split the
  affected region into local ordering zones.
- Relations outside the spatial span of a separator remain unchanged.

Select the strategy from the acceptance results. If synthetic nodes do not
produce a clear improvement, retain only separator-aware constraints.

## Phase 4: pipeline integration

1. Pass normalized page separators from `ReadingOrderModel` into root-level
   calls to `ReadingOrderPredictor`.
2. Never pass separator objects to:
   - `_find_to_captions`;
   - `_find_to_footnotes`;
   - `predict_merges`;
   - `DoclingDocument` construction.
3. Keep separator support behind a `ReadingOrderOptions` switch during
   evaluation.
4. Decide separately, after corpus evaluation, whether the option should
   become public and whether it is safe to enable by default.

## Phase 5: raster fallback

Defer Hough or morphological line detection until native-vector behavior is
validated.

If added later:

1. run it only when `get_shape_lines()` returns `None`, not when it returns an
   authoritative empty list;
2. make it explicitly configurable;
3. avoid introducing OpenCV as a mandatory core dependency;
4. map detected pixels back into page coordinates;
5. apply the same normalization and text-intersection filters as vector lines;
6. require strong raster support along the complete accepted segment.

## Verification

Add focused tests for:

- visibility and coordinate-origin propagation from the backend;
- candidate merging and rejection;
- vertical lane separation;
- horizontal band ordering;
- rectangle-derived segments;
- table-rule suppression;
- unsupported backend fallback;
- unchanged results when separator support is disabled or no lines exist;
- absence of separators from final output and relationship calculations.

Evaluate on:

1. the supplied journal-page case;
2. a corpus of PDFs with column rules and boxed regions;
3. PDFs with ruled tables and decorative lines;
4. PDFs with no useful rules.

Measure pairwise reading-order accuracy, full-order rank correlation, changed
documents, and runtime. Enabling the feature by default requires improvement
on separator-bearing pages with no material regression on the control corpus.

Run targeted reading-order and backend tests, followed by `make validate`.

## Decision gates

Resolved before implementation:

- native visible-vector geometry is the only source in the first iteration;
- separators have explicit semantics rather than relying solely on ordinary
  `PageElement` behavior;
- the three supplied PDFs provide the initial reproducible evaluation cases.

Before enabling by default:

- review the accepted/rejected-line overlays;
- review corpus-level ordering deltas;
- confirm that table rules and decorative rectangles do not cause regressions.
