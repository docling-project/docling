"""
Monkey patch for docling_ibm_models.reading_order.reading_order_rb module.

This module patches the _init_ud_maps method to add defensive checks
and prevent KeyError when accessing dn_map and up_map dictionaries.

The issue occurs when following the l2r_map chain results in an index
that doesn't exist in the dn_map dictionary. This can happen when
maps are reinitialized with different element lists.

This patch will be applied when the readingorder_model is imported.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from docling_ibm_models.reading_order.reading_order_rb import (
        PageElement,
        ReadingOrderPredictor,
        _ReadingOrderPredictorState,
    )


def _patched_init_ud_maps(
    self: "ReadingOrderPredictor",
    page_elems: List["PageElement"],
    state: "_ReadingOrderPredictorState",
) -> None:
    """
    Patched version of _init_ud_maps with defensive checks.

    Initialize up/down maps for reading order prediction using R-tree spatial indexing.

    Uses R-tree for spatial queries.
    Determines linear reading sequence by finding preceding/following elements.
    """
    from rtree import index as rtree_index

    state.up_map = {}
    state.dn_map = {}

    for i, pelem_i in enumerate(page_elems):
        state.up_map[i] = []
        state.dn_map[i] = []

    # Build R-tree spatial index
    spatial_idx = rtree_index.Index()
    for i, pelem in enumerate(page_elems):
        spatial_idx.insert(i, (pelem.l, pelem.b, pelem.r, pelem.t))

    for j, pelem_j in enumerate(page_elems):
        if j in state.r2l_map:
            i = state.r2l_map[j]
            # Defensive check: ensure i exists in dn_map
            if i in state.dn_map and j in state.up_map:
                state.dn_map[i] = [j]
                state.up_map[j] = [i]
            continue

        # Find elements above current that might precede it in reading order
        query_bbox = (pelem_j.l - 0.1, pelem_j.t, pelem_j.r + 0.1, float("inf"))
        candidates = list(spatial_idx.intersection(query_bbox))

        for i in candidates:
            if i == j:
                continue

            pelem_i = page_elems[i]

            # Check spatial relationship
            if not (
                pelem_i.is_strictly_above(pelem_j)
                and pelem_i.overlaps_horizontally(pelem_j)
            ):
                continue

            # Check for interrupting elements
            if not self._has_sequence_interruption(
                spatial_idx, page_elems, i, j, pelem_i, pelem_j
            ):
                # Follow left-to-right mapping
                original_i = i
                while i in state.l2r_map:
                    i = state.l2r_map[i]
                    # Defensive check: prevent infinite loops
                    if i == original_i:
                        break

                # Defensive check: ensure i and j exist in the maps before accessing
                if i in state.dn_map and j in state.up_map:
                    state.dn_map[i].append(j)
                    state.up_map[j].append(i)


def apply_patch() -> None:
    """Apply the monkey patch to ReadingOrderPredictor._init_ud_maps."""
    try:
        from docling_ibm_models.reading_order.reading_order_rb import (
            ReadingOrderPredictor,
        )

        # Store original method for reference
        if not hasattr(ReadingOrderPredictor, "_original_init_ud_maps"):
            ReadingOrderPredictor._original_init_ud_maps = (  # type: ignore
                ReadingOrderPredictor._init_ud_maps
            )

        # Apply the patch
        ReadingOrderPredictor._init_ud_maps = _patched_init_ud_maps  # type: ignore
    except ImportError:
        # If docling_ibm_models is not installed, silently skip the patch
        pass
