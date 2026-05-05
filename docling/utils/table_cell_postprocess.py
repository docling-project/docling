"""Post-process TableFormer's per-cell predictions.

Four issues TableFormer's output occasionally exhibits:

1. **Header marking dropped or wrapped across rows.** When column titles
   wrap onto 2-3 visual lines (e.g. "Net Cash for / Reinvestment# / Amount
   ($)"), TableFormer marks only one of those lines as
   ``column_header=True`` and rolls the others into row 1 of data. On
   simpler grids it occasionally drops header marking entirely.
   ``promote_wrapped_header_rows`` handles both: when no header is marked
   it seeds row 0 if it looks like an alphabetic caption over a numeric
   data row, then walks forward promoting header-like continuations.

2. **Merged-cell duplication.** When a label visually spans multiple grid
   columns (or rows), TableFormer's matcher can emit the same text into
   each of those grid positions instead of one cell with ``col_span > 1``.
   ``consolidate_duplicate_spans`` collapses adjacent same-row cells that
   share text and have touching bboxes into one colspan cell, and the
   mirror for vertical rowspans.

3. **Silent loss of bbox-but-not-grid words.** TableFormer is fed every word
   inside ``table_cluster.bbox`` (pulled via ``get_cells_in_bbox``) but only
   the words it places in its predicted grid come back via ``tf_responses``.
   Words it does not grid are otherwise silently dropped — they appear in
   neither the table nor any other layout element. ``recover_leftover_words``
   restores conservation: every input word ends up in *some* output element,
   either as a new table row (when leftovers form a clear tabular pattern)
   or as a new TEXT cluster on the page layout.

4. **Multiple sub-tables merged into one cluster.** The layout
   postprocessor occasionally merges visually-distinct tables into a single
   TABLE cluster (e.g. a stack of "label, value" summary tables followed
   by a wide totals table). TableFormer is then forced to invent a grid
   covering all of them, with rows of sharply-different structural shape.
   ``split_into_structural_subtables`` detects per-row distinct-cell-count
   shifts (sustained across multiple rows) and emits each piece as its own
   TABLE Cluster + Table. Page assembly handles multiple Tables on one
   page natively.

All four helpers are conservative: they require strong structural evidence
(text equality, bbox adjacency, codebase-standard 0.5 IoS containment,
sustained pattern shifts) and only act on cells that are obviously
redundant or obviously misplaced. Tables where TableFormer already emitted
the correct grid are unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc import BoundingBox, TableCell
    from docling_core.types.doc.page import TextCell

    from docling.datamodel.base_models import Cluster


_DIGIT_OR_CURRENCY = re.compile(r"[\d$€£¥%]")


def _avg_height(cells: Sequence[TableCell]) -> float:
    heights = [
        c.bbox.t - c.bbox.b
        if c.bbox.coord_origin == "BOTTOMLEFT"
        else c.bbox.b - c.bbox.t
        for c in cells
        if c.bbox is not None
    ]
    heights = [h for h in heights if h > 0]
    return sum(heights) / len(heights) if heights else 0.0


def _seed_row_0_as_header(
    table_cells: list[TableCell],
    by_row: dict[int, list[TableCell]],
) -> bool:
    """Promote row 0 to ``column_header=True`` when it looks like an
    alphabetic-caption row over a numeric data row.

    TableFormer occasionally returns ``column_header=False`` on every cell
    despite the row clearly being headers (the grid + cell text are
    correct; only the boolean is wrong). Without this seed step, the
    walk-forward continuation logic in ``promote_wrapped_header_rows``
    has nothing to walk from and the table is rendered headerless.

    Discriminator (must satisfy ALL):

    * Row 0 and row 1 both exist (gate at caller-side typically also
      enforces this via ``num_rows >= 2``).
    * Every non-empty cell in row 0 is non-digit (alphabetic caption).
      Currency symbols / percent signs without surrounding digits are
      tolerated so headers like "Cash per Unit ($)" still qualify.
    * At least one row-1 cell contains digit / currency / percent
      characters — confirms row 1 is a data row, not another header.

    Returns True iff at least one row-0 cell was promoted.
    """
    if 0 not in by_row or 1 not in by_row:
        return False

    row0_text_cells = [c for c in by_row[0] if c.text and c.text.strip()]
    if not row0_text_cells:
        return False

    # Row 0 must be alphabetic — no digits in any non-empty cell. (The
    # `_DIGIT_OR_CURRENCY` regex matches currency too, but caption-style
    # headers like "Cash per Unit ($)" should pass; reject only on
    # digits, since numeric content is the strong "this is data" signal.)
    for c in row0_text_cells:
        if any(ch.isdigit() for ch in c.text):
            return False

    # Row 1 must contain at least one numeric / currency / percent token.
    row1_text_cells = [c for c in by_row[1] if c.text and c.text.strip()]
    if not any(_DIGIT_OR_CURRENCY.search(c.text) for c in row1_text_cells):
        return False

    promoted_any = False
    for c in row0_text_cells:
        c.column_header = True
        promoted_any = True
    return promoted_any


def promote_wrapped_header_rows(
    table_cells: list[TableCell],
    num_rows: int,
    *,
    max_avg_text_len: int = 25,
    line_height_tol: float = 0.20,
) -> None:
    """In-place: promote rows that look like header rows or wrapped header
    continuations.

    Two cases:

    * **Seed case** — when no cell currently has ``column_header=True`` but
      row 0 is an alphabetic caption above a numeric row 1, promote row 0
      first (see ``_seed_row_0_as_header``). This handles tables where
      TableFormer dropped the column-header flag entirely.

    * **Continuation case** — walks forward from the last existing
      column-header row. For each subsequent row, the row is promoted if
      all the following hold:
        - No cell text contains digits, currency symbols, or %.
        - Mean cell-text length is short (<= ``max_avg_text_len``).
        - Average row height matches the header's average within
          ``line_height_tol``.
        - At least one cell in the row is non-empty.
      Stops at the first row that fails any check.

    No-op if there are no rows or ``num_rows < 2``.
    """
    if not table_cells or num_rows < 2:
        return

    by_row: dict[int, list[TableCell]] = {}
    for c in table_cells:
        by_row.setdefault(c.start_row_offset_idx, []).append(c)

    header_idxs = [
        c.start_row_offset_idx
        for c in table_cells
        if getattr(c, "column_header", False)
    ]
    if not header_idxs:
        # Seed: no header at all. Try to promote row 0 if it looks like a
        # caption over a numeric data row.
        if not _seed_row_0_as_header(table_cells, by_row):
            return
        # Re-collect now that row 0 has been promoted.
        header_idxs = [
            c.start_row_offset_idx
            for c in table_cells
            if getattr(c, "column_header", False)
        ]
        if not header_idxs:
            return
    last_header_row = max(header_idxs)

    header_cells = [c for c in table_cells if getattr(c, "column_header", False)]
    hdr_h = _avg_height(header_cells)

    for r in range(last_header_row + 1, num_rows):
        row = by_row.get(r, [])
        if not row:
            return
        texts = [(c.text or "").strip() for c in row]
        if not any(texts):
            return
        if any(_DIGIT_OR_CURRENCY.search(t) for t in texts if t):
            return
        non_empty = [t for t in texts if t]
        if (
            non_empty
            and (sum(len(t) for t in non_empty) / len(non_empty)) > max_avg_text_len
        ):
            return
        row_h = _avg_height(row)
        if hdr_h > 0 and row_h > 0 and abs(row_h - hdr_h) / hdr_h > line_height_tol:
            return
        for c in row:
            c.column_header = True


_WS_COLLAPSE = re.compile(r"\s+")


def _norm_text(s: str | None) -> str:
    """Normalise for text-equality comparisons across cells.

    Collapses internal whitespace runs to a single space so cells whose
    text differs only by stray double-spaces (a common PDF-extraction
    artefact) compare equal. Lower-cased and stripped — same shape as
    before, just whitespace-aware.
    """
    return _WS_COLLAPSE.sub(" ", (s or "").strip()).lower()


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over Union of two bboxes. Coord-origin-agnostic."""
    inter = a.intersection_area_with(b)
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _bboxes_touch_horizontally(
    a: BoundingBox, b: BoundingBox, tol: float = 2.0
) -> bool:
    # Same vertical band (overlap), horizontally adjacent.
    if a.coord_origin == "BOTTOMLEFT":
        v_overlap = min(a.t, b.t) - max(a.b, b.b)
    else:
        v_overlap = min(a.b, b.b) - max(a.t, b.t)
    if v_overlap <= 0:
        return False
    # adjacency: a.r ~ b.l or b.r ~ a.l
    return abs(a.r - b.l) <= tol or abs(b.r - a.l) <= tol


def _bboxes_touch_vertically(a: BoundingBox, b: BoundingBox, tol: float = 2.0) -> bool:
    # Same horizontal band (overlap), vertically adjacent.
    h_overlap = min(a.r, b.r) - max(a.l, b.l)
    if h_overlap <= 0:
        return False
    if a.coord_origin == "BOTTOMLEFT":
        return abs(a.b - b.t) <= tol or abs(b.b - a.t) <= tol
    return abs(a.b - b.t) <= tol or abs(b.b - a.t) <= tol


def _union_bbox(cells: Sequence[TableCell]) -> BoundingBox:
    from docling_core.types.doc import BoundingBox

    boxes = [c.bbox for c in cells if c.bbox is not None]
    return BoundingBox(
        l=min(b.l for b in boxes),
        t=min(b.t for b in boxes),
        r=max(b.r for b in boxes),
        b=max(b.b for b in boxes),
        coord_origin=boxes[0].coord_origin,
    )


def consolidate_duplicate_spans(
    table_cells: list[TableCell],
) -> list[TableCell]:
    """Collapse adjacent cells with identical text into one cell with span > 1.

    Returns a new list. The original list is not modified.

    Detection is conservative: cells must have the same row offset (for
    column merging) or same column offset (for row merging), exactly equal
    normalised text, and bboxes that touch (within 2 page units).
    """
    if not table_cells:
        return list(table_cells)

    # ----- 1) Horizontal merge (collapse colspan duplicates). -----
    # Group by start_row_offset_idx, sort by start_col_offset_idx.
    by_row: dict[int, list[TableCell]] = {}
    others: list[TableCell] = []
    for c in table_cells:
        if c.bbox is None or not (c.text and c.text.strip()):
            others.append(c)
            continue
        by_row.setdefault(c.start_row_offset_idx, []).append(c)

    horizontally_merged: list[TableCell] = list(others)
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: c.start_col_offset_idx)
        i = 0
        while i < len(row_cells):
            anchor = row_cells[i]
            run = [anchor]
            j = i + 1
            while j < len(row_cells):
                cand = row_cells[j]
                last = run[-1]
                if (
                    _norm_text(cand.text) == _norm_text(anchor.text)
                    and _norm_text(anchor.text) != ""
                    and cand.start_col_offset_idx == last.end_col_offset_idx
                    and last.bbox is not None
                    and cand.bbox is not None
                    and _bboxes_touch_horizontally(last.bbox, cand.bbox)
                ):
                    run.append(cand)
                    j += 1
                else:
                    break
            if len(run) > 1:
                merged = anchor.model_copy(
                    update={
                        "bbox": _union_bbox(run),
                        "col_span": sum(c.col_span for c in run),
                        "end_col_offset_idx": run[-1].end_col_offset_idx,
                    }
                )
                horizontally_merged.append(merged)
            else:
                horizontally_merged.append(anchor)
            i = j

    # ----- 2) Vertical merge (collapse rowspan duplicates). -----
    by_col: dict[int, list[TableCell]] = {}
    leftovers: list[TableCell] = []
    for c in horizontally_merged:
        if c.bbox is None or not (c.text and c.text.strip()):
            leftovers.append(c)
            continue
        by_col.setdefault(c.start_col_offset_idx, []).append(c)

    final: list[TableCell] = list(leftovers)
    for col_cells in by_col.values():
        col_cells.sort(key=lambda c: c.start_row_offset_idx)
        i = 0
        while i < len(col_cells):
            anchor = col_cells[i]
            run = [anchor]
            j = i + 1
            while j < len(col_cells):
                cand = col_cells[j]
                last = run[-1]
                if (
                    _norm_text(cand.text) == _norm_text(anchor.text)
                    and _norm_text(anchor.text) != ""
                    and cand.start_row_offset_idx == last.end_row_offset_idx
                    and cand.start_col_offset_idx == anchor.start_col_offset_idx
                    and cand.end_col_offset_idx == anchor.end_col_offset_idx
                    and last.bbox is not None
                    and cand.bbox is not None
                    and _bboxes_touch_vertically(last.bbox, cand.bbox)
                ):
                    run.append(cand)
                    j += 1
                else:
                    break
            if len(run) > 1:
                merged = anchor.model_copy(
                    update={
                        "bbox": _union_bbox(run),
                        "row_span": sum(c.row_span for c in run),
                        "end_row_offset_idx": run[-1].end_row_offset_idx,
                    }
                )
                final.append(merged)
            else:
                final.append(anchor)
            i = j

    # ----- 3) Identical-bbox same-text dedup. -----
    # Catches duplicates that aren't grid-adjacent: the same physical PDF
    # region emitted into two distinct grid positions. Grid-adjacency is
    # not required because the duplicates may sit anywhere in the grid.
    # Conservative: requires normalised-text equality AND IoU >= 0.95
    # (essentially "same bbox") so distinct cells with coincidentally
    # equal values stay separate.
    deduped: list[TableCell] = []
    by_text: dict[str, list[TableCell]] = {}
    for c in final:
        if c.bbox is None or not (c.text and c.text.strip()):
            deduped.append(c)
            continue
        key = _norm_text(c.text)
        bucket = by_text.setdefault(key, [])
        is_dup = any(
            existing.bbox is not None and _bbox_iou(c.bbox, existing.bbox) >= 0.95
            for existing in bucket
        )
        if not is_dup:
            bucket.append(c)
            deduped.append(c)

    return deduped


def _grid_bbox(table_cells: Sequence[TableCell]) -> BoundingBox | None:
    from docling_core.types.doc import BoundingBox

    boxes = [c.bbox for c in table_cells if c.bbox is not None]
    if not boxes:
        return None
    return BoundingBox(
        l=min(b.l for b in boxes),
        t=min(b.t for b in boxes),
        r=max(b.r for b in boxes),
        b=max(b.b for b in boxes),
        coord_origin=boxes[0].coord_origin,
    )


def _column_centroids(table_cells: Sequence[TableCell]) -> dict[int, float]:
    """Per-column X-centroid, computed over all gridded cells in that column."""
    by_col: dict[int, list[BoundingBox]] = {}
    for c in table_cells:
        if c.bbox is None:
            continue
        by_col.setdefault(c.start_col_offset_idx, []).append(c.bbox)
    return {
        col: (min(b.l for b in bs) + max(b.r for b in bs)) / 2
        for col, bs in by_col.items()
    }


def _median_row_pitch(table_cells: Sequence[TableCell]) -> float | None:
    """Median Y-distance between consecutive predicted rows.

    Returns None for tables with fewer than 2 distinct rows — there is no
    structural signal to derive a row-pitch from in that case, so callers
    should fall back to conservative behaviour rather than guessing.
    """
    by_row: dict[int, list[BoundingBox]] = {}
    for c in table_cells:
        if c.bbox is None:
            continue
        by_row.setdefault(c.start_row_offset_idx, []).append(c.bbox)
    if len(by_row) < 2:
        return None
    centroids = sorted(
        (min(b.t for b in bs) + max(b.b for b in bs)) / 2 for bs in by_row.values()
    )
    deltas = [abs(b - a) for a, b in pairwise(centroids)]
    if not deltas:
        return None
    return sorted(deltas)[len(deltas) // 2]


def _attach_as_table_rows(
    leftover_cells: list[TextCell],
    table_cells: list[TableCell],
) -> tuple[list[TableCell], list[TextCell]]:
    """Try to place leftover words as new table rows.

    Discriminator (purely structural, no content / template hints): cluster
    the leftovers by Y-centroid using the table's own median row pitch as
    the proximity threshold. A cluster qualifies as a tabular row only if
    its members hit ≥2 distinct nearest-columns (column inferred by snapping
    each cell's X-centroid to the closest gridded-column centroid). Single-
    column clusters look like prose / captions and do not qualify.

    Each qualifying cluster becomes a new table row; cells nearest the same
    column are merged (text joined, bbox unioned) so the row keeps the
    column count it earned. Disqualified cells are returned for fallback
    handling (TEXT cluster).

    Returns (placed_table_cells, unplaced_text_cells). ``placed_table_cells``
    can be appended directly to the existing table_cells list.
    """
    if not table_cells or not leftover_cells:
        return [], list(leftover_cells)

    col_centroids = _column_centroids(table_cells)
    if len(col_centroids) < 2:
        # Without ≥2 columns there is no "distinct columns" signal — bail.
        return [], list(leftover_cells)

    pitch = _median_row_pitch(table_cells)
    if pitch is None or pitch <= 0:
        return [], list(leftover_cells)

    template = next((c for c in table_cells if c.bbox is not None), None)
    if template is None:
        return [], list(leftover_cells)

    # Annotate each leftover with its Y-centroid, X-centroid, and nearest
    # column id, then sort by Y so consecutive cells can be Y-clustered.
    enriched = []
    for c in leftover_cells:
        bb = c.rect.to_bounding_box()
        y_cent = (bb.t + bb.b) / 2
        x_cent = (bb.l + bb.r) / 2
        nearest_col = min(
            col_centroids, key=lambda col: abs(col_centroids[col] - x_cent)
        )
        enriched.append((y_cent, x_cent, nearest_col, c, bb))
    enriched.sort(key=lambda t: t[0])

    # Y-cluster: consecutive cells within `pitch` of each other belong to
    # the same logical row. `pitch` is a property of *this* table, not a
    # magic constant, so the threshold adapts to dense vs sparse layouts.
    clusters: list[list[tuple]] = [[enriched[0]]]
    for prev, curr in pairwise(enriched):
        if abs(curr[0] - prev[0]) < pitch:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    existing_max_row = max(c.start_row_offset_idx for c in table_cells)

    placed: list[TableCell] = []
    unplaced: list[TextCell] = []
    next_row = existing_max_row + 1

    for cluster in clusters:
        cols_hit = {item[2] for item in cluster}
        if len(cols_hit) < 2:
            # One column: looks like prose / single-column caption — not a row.
            for item in cluster:
                unplaced.append(item[3])
            continue

        # Group this cluster's cells by their nearest column. Multiple
        # leftovers nearest to the same column get merged into one cell so
        # the row keeps a clean column-aligned shape.
        by_col: dict[int, list[tuple]] = {}
        for item in cluster:
            by_col.setdefault(item[2], []).append(item)

        for col_id, items in by_col.items():
            text = " ".join(item[3].text.strip() for item in items if item[3].text)
            bbs = [item[4] for item in items]
            from docling_core.types.doc import BoundingBox

            merged_bbox = BoundingBox(
                l=min(b.l for b in bbs),
                t=min(b.t for b in bbs),
                r=max(b.r for b in bbs),
                b=max(b.b for b in bbs),
                coord_origin=bbs[0].coord_origin,
            )
            new_cell = template.model_copy(
                update={
                    "bbox": merged_bbox,
                    "text": text,
                    "start_row_offset_idx": next_row,
                    "end_row_offset_idx": next_row + 1,
                    "start_col_offset_idx": col_id,
                    "end_col_offset_idx": col_id + 1,
                    "row_span": 1,
                    "col_span": 1,
                    "column_header": False,
                    "row_header": False,
                    "row_section": False,
                }
            )
            placed.append(new_cell)
        next_row += 1

    return placed, unplaced


def recover_leftover_words(
    *,
    tcells: Sequence[TextCell],
    table_cells: list[TableCell],
    table_cluster: Cluster,
    page_clusters: list[Cluster],
    threshold: float = 0.5,
) -> tuple[int, Cluster | None]:
    """Recover input PDF words TableFormer did not place in its predicted grid.

    The TABLE cluster bbox is grown by ``LayoutPostprocessor`` (TABLE-only
    union-with-cells branch) to swallow nearby content. Every word inside
    that bbox is shipped to TableFormer as input; only words it places in
    its predicted grid come back via ``tf_responses``. Without this helper,
    everything in between is silently dropped — invisible in both the table
    and the surrounding text. TABLE is the only cluster type with this
    non-conservative dataflow; every other label preserves the invariant
    "every input word lands in some output element".

    Containment uses the codebase's existing ``intersection_over_self``
    convention with the ``0.5`` threshold (see ``layout_postprocessor.py``
    lines 345, 404, 444): a word is considered "outside the grid" when more
    than half its bbox lies outside the union of predicted grid cell bboxes.

    Recovery is two-pass:

    1. **Structural pass** (``_attach_as_table_rows``): leftovers that form
       ≥2 cells in distinct columns at similar Y are placed as new rows in
       ``table_cells``. The proximity threshold is the table's own median
       row pitch — no magic constants. This restores both the data AND its
       column-header context.
    2. **Conservation pass**: any leftovers that fail the structural test
       (single-column clusters, isolated cells, captions) are emitted as a
       TEXT cluster appended to ``page_clusters`` so they at least appear
       in the JSON. The TABLE cluster's bbox is shrunk to the union of all
       gridded cells (existing + newly placed) so the residual region is no
       longer claimed by the table.

    Returns ``(num_new_rows, leftover_text_cluster)``. The caller must add
    ``num_new_rows`` to its ``num_rows`` value before constructing the
    ``Table`` so downstream code knows the table grew.

    No-op (returns ``(0, None)``) when:

    * ``tcells`` is empty (nothing to recover);
    * ``table_cells`` produced no grid (no signal for what to keep);
    * no ``tcells`` entry falls outside the grid by more than ``threshold``.
    """
    from docling_core.types.doc import BoundingBox, DocItemLabel

    from docling.datamodel.base_models import Cluster

    grid_bbox = _grid_bbox(table_cells)
    if grid_bbox is None or not tcells:
        return 0, None

    leftover_cells: list[TextCell] = []
    for c in tcells:
        if not c.text or not c.text.strip():
            continue
        cb = c.rect.to_bounding_box()
        if cb.intersection_over_self(grid_bbox) < threshold:
            leftover_cells.append(c)

    if not leftover_cells:
        return 0, None

    # Pass 1: structural recovery — place qualifying leftovers as new rows.
    placed, unplaced = _attach_as_table_rows(leftover_cells, table_cells)
    table_cells.extend(placed)
    new_row_ids = {c.start_row_offset_idx for c in placed}
    new_row_count = len(new_row_ids)

    # Pass 2: cells that didn't qualify → emit as TEXT cluster (conservation).
    leftover_cluster: Cluster | None = None
    if unplaced:
        unplaced_bboxes = [c.rect.to_bounding_box() for c in unplaced]
        leftover_bbox = BoundingBox(
            l=min(b.l for b in unplaced_bboxes),
            t=min(b.t for b in unplaced_bboxes),
            r=max(b.r for b in unplaced_bboxes),
            b=max(b.b for b in unplaced_bboxes),
            coord_origin=unplaced_bboxes[0].coord_origin,
        )
        next_id = max((cl.id for cl in page_clusters), default=-1) + 1
        leftover_cluster = Cluster(
            id=next_id,
            label=DocItemLabel.TEXT,
            bbox=leftover_bbox,
            confidence=table_cluster.confidence,
            cells=list(unplaced),
        )
        page_clusters.append(leftover_cluster)

    # Shrink table_cluster.bbox to the union of gridded cells (existing +
    # newly placed). This both releases the leftover-TEXT region and, when
    # rows were structurally added, expands to encompass them.
    new_grid_bbox = _grid_bbox(table_cells)
    if new_grid_bbox is not None:
        table_cluster.bbox = new_grid_bbox

    return new_row_count, leftover_cluster


def detect_subtable_boundaries(
    table_cells: list[TableCell],
    *,
    min_count_ratio: float = 1.5,
    min_consecutive: int = 2,
) -> list[int]:
    """Detect rows where a TABLE cluster transitions between sub-tables.

    Layout postprocessing occasionally merges several visually-distinct
    tables on a page into one TABLE cluster (e.g. a stack of "label, value"
    summary tables followed by a wide totals table). TableFormer then has
    to invent a single grid covering all of them, producing rows whose
    structural shape differs sharply: some rows have just two cells (label
    spanning N-1 cols + value), others have N cells across distinct
    columns. This function detects those structural breaks so the caller
    can split the cluster into separate ``Table`` objects.

    The discriminator is the per-row distinct-cell count. A boundary is
    declared between row r-1 and r when:

    * ``max(count_prev, count_curr) / max(min(count_prev, count_curr), 1)``
      is at least ``min_count_ratio`` — the change is structurally large,
      not a one-cell wobble; AND
    * the new shape is **sustained** for ``min_consecutive`` rows — i.e.
      rows ``r..r+min_consecutive-1`` all stay within ``min_count_ratio``
      of each other. A single odd row (e.g. an isolated subtotal in the
      middle of a wide table) does not trigger a split.

    Returns the list of row indices where each sub-table starts. Always
    includes the first row index. Returns an empty list when
    ``table_cells`` is empty.

    Conservative by design: no template-specific or text-content
    heuristics; the discriminator is purely the structural shape of each
    row, and the sustain rule prevents single-row outliers from
    fragmenting otherwise-coherent tables.
    """
    if not table_cells:
        return []

    counts: dict[int, int] = {}
    for c in table_cells:
        counts[c.start_row_offset_idx] = counts.get(c.start_row_offset_idx, 0) + 1
    sorted_rows = sorted(counts)

    boundaries = [sorted_rows[0]]
    for i in range(1, len(sorted_rows)):
        prev_r, curr_r = sorted_rows[i - 1], sorted_rows[i]
        prev_c, curr_c = counts[prev_r], counts[curr_r]
        ratio = max(prev_c, curr_c) / max(min(prev_c, curr_c), 1)
        if ratio < min_count_ratio:
            continue
        # Sustain check: do at least `min_consecutive - 1` more rows after
        # `curr_r` stay within ratio of curr_c?
        sustained = True
        for j in range(1, min_consecutive):
            if i + j >= len(sorted_rows):
                # Ran off the end before sustaining — treat as not sustained
                # (a single trailing odd row is more likely noise than a
                # genuine sub-table).
                sustained = False
                break
            next_c = counts[sorted_rows[i + j]]
            r2 = max(curr_c, next_c) / max(min(curr_c, next_c), 1)
            if r2 > min_count_ratio:
                sustained = False
                break
        if sustained:
            boundaries.append(curr_r)
    return boundaries


def _renumber_rows(cells: list[TableCell], offset: int) -> list[TableCell]:
    """Return cells with start/end_row_offset_idx shifted down by ``offset``.

    Each cell is replaced via ``model_copy`` so the originals are not
    mutated (callers may still hold references to them).
    """
    out = []
    for c in cells:
        out.append(
            c.model_copy(
                update={
                    "start_row_offset_idx": c.start_row_offset_idx - offset,
                    "end_row_offset_idx": c.end_row_offset_idx - offset,
                }
            )
        )
    return out


def split_into_structural_subtables(
    *,
    table_cells: list[TableCell],
    table_cluster: Cluster,
    page_clusters: list[Cluster],
    table_map: dict,
    page_no: int,
) -> tuple[list[TableCell], int, int, bool]:
    """Split ``table_cells`` into multiple Tables on row-pattern boundaries.

    When ``detect_subtable_boundaries`` finds the cluster contains multiple
    structurally-distinct sub-tables, this orchestrator:

    * partitions ``table_cells`` by row range,
    * keeps the first sub-table on the original ``table_cluster`` (with
      bbox shrunk to its extent),
    * for each subsequent sub-table, mints a new cluster id (mirroring the
      idiom in ``recover_leftover_words``), appends a TABLE ``Cluster`` to
      ``page_clusters``, constructs a ``Table`` from the renumbered cells,
      and writes ``table_map[new_id] = table``.

    Returns ``(first_cells, first_num_rows, first_num_cols, did_split)``.
    The caller uses ``first_cells`` / ``first_num_rows`` / ``first_num_cols``
    to construct the Table for the original cluster id; ``did_split=True``
    means caller should also clear ``otsl_seq`` because the predicted
    sequence describes the un-split whole.

    No-op (returns the input unchanged with ``did_split=False``) when the
    discriminator finds at most one sub-table.
    """
    from docling_core.types.doc import DocItemLabel

    from docling.datamodel.base_models import Cluster as ClusterModel, Table

    boundaries = detect_subtable_boundaries(table_cells)
    if len(boundaries) <= 1 or not table_cells:
        nr = max((c.end_row_offset_idx for c in table_cells), default=0)
        nc = max((c.end_col_offset_idx for c in table_cells), default=0)
        return list(table_cells), nr, nc, False

    # Partition cells by row range (boundary[i] inclusive .. boundary[i+1] exclusive)
    sub_groups: list[tuple[int, list[TableCell]]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
        cells = [c for c in table_cells if start <= c.start_row_offset_idx < end]
        sub_groups.append((start, cells))

    # First sub-group keeps the original cluster id; renumber its rows so
    # the kept sub-table starts at row 0.
    first_start, first_raw = sub_groups[0]
    first_cells = _renumber_rows(first_raw, first_start)
    first_nr = max((c.end_row_offset_idx for c in first_cells), default=0)
    first_nc = max((c.end_col_offset_idx for c in first_cells), default=0)
    first_bbox = _grid_bbox(first_cells)
    if first_bbox is not None:
        table_cluster.bbox = first_bbox

    # Subsequent sub-groups become new clusters + table_map entries.
    for start, raw_cells in sub_groups[1:]:
        sub_cells = _renumber_rows(raw_cells, start)
        sub_nr = max((c.end_row_offset_idx for c in sub_cells), default=0)
        sub_nc = max((c.end_col_offset_idx for c in sub_cells), default=0)
        sub_bbox = _grid_bbox(sub_cells)
        if sub_bbox is None:
            continue
        next_id = max((cl.id for cl in page_clusters), default=-1) + 1
        new_cluster = ClusterModel(
            id=next_id,
            label=DocItemLabel.TABLE,
            bbox=sub_bbox,
            confidence=table_cluster.confidence,
            cells=[],
        )
        page_clusters.append(new_cluster)
        sub_table = Table(
            otsl_seq=[],
            table_cells=sub_cells,
            num_rows=sub_nr,
            num_cols=sub_nc,
            id=next_id,
            page_no=page_no,
            cluster=new_cluster,
            label=new_cluster.label,
        )
        table_map[next_id] = sub_table

    return first_cells, first_nr, first_nc, True
