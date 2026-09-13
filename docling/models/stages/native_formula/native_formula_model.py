# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Attach the MathML a tagged PDF already carries to the formulas detected on the page.

The layout model finds *where* the equations are; a tagged PDF's structure tree says what
they *are*. When both are available the author's own MathML is the better answer, so this
stage matches the two by position and records the MathML on the matching ``FormulaItem``.

It runs right after the reading-order model, on the assembled document, and only ever
writes to ``FormulaItem``\\ s -- it never adds, removes, relabels or reorders items.
Structure elements that match no detected formula are dropped: recovering equations the
layout model missed entirely is a separate problem.

Once an item carries MathML the formula understanding model skips it (see
:func:`docling.utils.formula_meta.has_native_mathml`), which leaves ``text`` empty, so the
structure element's ``/ActualText`` or ``/Alt`` is used to fill it. Neither is guaranteed to
be present; ``orig`` (the raw extracted text) is the last resort.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from docling_core.types.doc import BoundingBox, DoclingDocument
from docling_core.types.doc.common.meta import FormulaMeta, FormulaMetaField
from docling_core.types.doc.document import FormulaItem

from docling.datamodel.document import ConversionResult
from docling.utils.pdf_struct_tree import _PdfFormulaStruct

_log = logging.getLogger(__name__)

# Recorded on the meta field so consumers can tell an authored representation from a
# predicted one.
NATIVE_FORMULA_SOURCE = "pdf_struct_tree"

# A confident overlap between the detected formula and the structure element.
_MIN_IOU = 0.5

# Structure ``BBox`` attributes are frequently padded relative to the detected layout box
# (and marked-content bounds are frequently tighter), which drags IoU down even on an
# obvious match. Fall back to how much of the smaller box the larger one swallows.
_MIN_CONTAINMENT = 0.8


def _overlap_score(a: BoundingBox, b: BoundingBox) -> float:
    """Return how confidently *a* and *b* describe the same region, 0.0 when they do not."""
    iou = a.intersection_over_union(b)
    if iou >= _MIN_IOU:
        return iou
    containment = max(a.intersection_over_self(b), b.intersection_over_self(a))
    # Scaled below the IoU band so a containment match never outranks a genuine IoU match.
    return containment * _MIN_IOU if containment >= _MIN_CONTAINMENT else 0.0


class NativeFormulaModel:
    """Records embedded MathML on the formulas of an already-assembled ``DoclingDocument``."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        structs = conv_res._pdf_formula_structs
        if not self.enabled or not structs:
            return document

        try:
            return self.apply_native_formulas(document, structs)
        finally:
            # Release the transient records once consumed.
            conv_res._pdf_formula_structs = None

    def apply_native_formulas(
        self, document: DoclingDocument, structs: list[_PdfFormulaStruct]
    ) -> DoclingDocument:
        """Match *structs* to the document's formulas and record their MathML.

        Works on a bare ``DoclingDocument`` so it can be reused outside the pipeline.
        """
        by_page: dict[int, list[_PdfFormulaStruct]] = defaultdict(list)
        for struct in structs:
            # Records without MathML carry nothing this stage can add.
            if struct.mathml and struct.bbox is not None:
                by_page[struct.page_no].append(struct)
        if not by_page:
            return document

        items_by_page: dict[int, list[tuple[FormulaItem, BoundingBox]]] = defaultdict(
            list
        )
        for item, _level in document.iterate_items(with_groups=False):
            if not isinstance(item, FormulaItem) or not item.prov:
                continue
            prov = item.prov[0]
            page = document.pages.get(prov.page_no)
            if page is None or page.size is None or prov.page_no not in by_page:
                continue
            items_by_page[prov.page_no].append(
                (item, prov.bbox.to_top_left_origin(page_height=page.size.height))
            )

        matched = 0
        for page_no, page_items in items_by_page.items():
            matched += self._apply_page(page_items, by_page[page_no])

        if matched:
            _log.debug("Recorded embedded MathML on %d formula(s)", matched)
        return document

    def _apply_page(
        self,
        page_items: list[tuple[FormulaItem, BoundingBox]],
        page_structs: list[_PdfFormulaStruct],
    ) -> int:
        """Greedily pair the formulas and structure elements on one page, best match first."""
        candidates = [
            (_overlap_score(item_bbox, struct.bbox), item_idx, struct_idx)
            for item_idx, (_item, item_bbox) in enumerate(page_items)
            for struct_idx, struct in enumerate(page_structs)
            if struct.bbox is not None
        ]
        # Ties are broken by document order so the pairing does not depend on dict ordering.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        used_items: set[int] = set()
        used_structs: set[int] = set()
        matched = 0
        for score, item_idx, struct_idx in candidates:
            if score <= 0.0:
                break
            if item_idx in used_items or struct_idx in used_structs:
                continue
            used_items.add(item_idx)
            used_structs.add(struct_idx)

            item, _bbox = page_items[item_idx]
            struct = page_structs[struct_idx]
            item.meta = FormulaMeta(
                formula=FormulaMetaField(
                    mathml=struct.mathml, created_by=NATIVE_FORMULA_SOURCE
                )
            )
            if not item.text:
                item.text = (
                    struct.actual_text or struct.alt_text or item.orig or ""
                ).strip()
            matched += 1

        return matched
