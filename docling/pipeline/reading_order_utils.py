"""Utilities for correcting document reading order issues.

This module provides helpers for fixing reading order problems where items
are emitted out of their physical page order, particularly for single-column PDFs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


def _get_first_prov(
    ref_item, doc: DoclingDocument, _visited: Optional[set] = None
) -> Optional[Tuple[int, float]]:
    """Extract the first provenance of an item as (page_no, -bbox.t) sort key.

    Returns None if the item has no provenance or cannot be resolved.
    Recursively searches children if the item itself has no direct provenance.
    """
    if _visited is None:
        _visited = set()

    # Avoid infinite recursion
    if id(ref_item) in _visited:
        return None
    _visited.add(id(ref_item))

    try:
        node = ref_item.resolve(doc=doc)
    except Exception:
        return None

    # Check if this node has provenance
    result = None
    prov = getattr(node, "prov", None)
    if prov:
        try:
            result = (prov[0].page_no, -prov[0].bbox.t)
        except (AttributeError, IndexError, TypeError):
            result = None

    # If no provenance, try children recursively
    if result is None:
        for child in getattr(node, "children", None) or []:
            sub = _get_first_prov(child, doc, _visited)
            if sub is not None:
                result = sub
                break

    return result


def correct_reading_order_on_page(doc: DoclingDocument) -> int:
    """Re-order doc.body.children by (page_no, -bbox.t) within each page.

    This corrects cases where docling's reading-order linearization places
    items out of their physical order on the page. Uses stable insertion sort
    to preserve correctly-ordered sequences and multi-column layouts.

    Items lacking provenance (empty groups, etc.) are anchored to their
    original position; cross-page order is never touched.

    Args:
        doc: The DoclingDocument to correct

    Returns:
        The number of items repositioned (for logging/debugging)
    """
    body = doc.body
    children = list(body.children)
    if len(children) < 2:
        return 0

    # Insertion sort, page-scoped: each item floats up past same-page
    # siblings whose key is greater (= bbox.t is higher on page).
    result: list = []
    moved = 0

    for cur in children:
        cur_key = _get_first_prov(cur, doc)

        if cur_key is None:
            # No provenance - anchor to original position
            result.append(cur)
            continue

        # Find insertion point: scan backward to find where cur should go
        insert_at = len(result)
        for j in range(len(result) - 1, -1, -1):
            prev_key = _get_first_prov(result[j], doc)

            if prev_key is None:
                # Previous item has no provenance - don't pass it
                break

            if prev_key[0] != cur_key[0]:
                # Different page - stop scanning
                break

            if prev_key[1] <= cur_key[1]:
                # Previous item already sorts before current item
                break

            # Current item should go before previous item
            insert_at = j

        if insert_at != len(result):
            moved += 1

        result.insert(insert_at, cur)

    body.children = result
    return moved
