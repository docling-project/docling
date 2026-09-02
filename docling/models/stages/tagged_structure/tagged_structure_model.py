# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Layout from the tagged-PDF structure tree.

A tagged PDF (ISO 32000-2, 14.7; Well-Tagged PDF 1.0) carries the author's own
logical structure: which glyphs form a paragraph, a heading and its level, a
list item, a caption, a figure with its alternate text, and which content is a
pagination artifact. When that structure is present this stage turns it into
layout clusters with confidence 1.0, placed where the linked marked content
sits on the page, so layout postprocessing, assembly and reading order treat
the authored structure as authoritative instead of the layout model's guess.

The stage runs after OCR and before layout postprocessing, like the form-field
stage: raw clusters are still available for the ``prefer`` merge, and nothing
downstream has to learn a new element kind. Untagged pages, and every page when
the option is ``off``, pass through unchanged.
"""

import logging
from collections.abc import Iterable
from typing import Literal

from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_parse.pdf_parser import (
    PdfMarkedContentRef,
    PdfStructure,
    PdfStructureElement,
)

from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import (
    Cluster,
    LayoutPrediction,
    Page,
    TaggedStructurePrediction,
    TaggedTextCell,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

TaggedStructureMode = Literal["off", "prefer", "require"]

# Structure types whose content is one layout element. Values are the layout
# labels the postprocessor knows how to score and assemble.
_LEAF_LABELS: dict[str, DocItemLabel] = {
    "/P": DocItemLabel.TEXT,
    "/Title": DocItemLabel.TITLE,
    "/H": DocItemLabel.SECTION_HEADER,
    "/FENote": DocItemLabel.FOOTNOTE,
    "/Note": DocItemLabel.FOOTNOTE,
    "/Caption": DocItemLabel.CAPTION,
    "/Code": DocItemLabel.CODE,
    "/Formula": DocItemLabel.FORMULA,
    "/Figure": DocItemLabel.PICTURE,
    "/Table": DocItemLabel.TABLE,
    "/LI": DocItemLabel.LIST_ITEM,
    "/TOC": DocItemLabel.DOCUMENT_INDEX,
    "/BibEntry": DocItemLabel.TEXT,
    "/Index": DocItemLabel.TEXT,
}
for _level in range(1, 7):
    _LEAF_LABELS[f"/H{_level}"] = DocItemLabel.SECTION_HEADER

# Grouping types: no element of their own, their children are walked.
_CONTAINERS = frozenset(
    {
        "/Document",
        "/DocumentFragment",
        "/Part",
        "/Art",
        "/Sect",
        "/Div",
        "/BlockQuote",
        "/Aside",
        "/NonStruct",
        "/Private",
        "/L",
        "/Form",
        "/Annot",
    }
)

# Fraction of a model cluster that must lie inside tagged clusters for it to
# count as already covered in ``prefer`` mode.
_COVERED_THRESHOLD = 0.5


def _heading_level(resolved_type: str) -> int | None:
    if resolved_type == "/H":
        return 1
    if resolved_type.startswith("/H") and resolved_type[2:].isdigit():
        return int(resolved_type[2:])
    return None


def _mcids_on_page(element: PdfStructureElement, page_index: int) -> set[int]:
    """Every /MCID this element and its inline descendants own on the page."""
    found: set[int] = set()
    stack = [element]
    while stack:
        current = stack.pop()
        for kid in current.kids:
            if isinstance(kid, PdfMarkedContentRef):
                if kid.page == page_index:
                    found.add(kid.mcid)
            elif isinstance(kid, PdfStructureElement):
                stack.append(kid)
    return found


def _layout_bbox(
    element: PdfStructureElement, page_height: float
) -> BoundingBox | None:
    """The Layout /BBox attribute (bottom-left user space) as a top-left box."""
    layout = element.attributes.get("/Layout") or {}
    raw = layout.get("/BBox")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return BoundingBox(
        l=min(x0, x1),
        r=max(x0, x1),
        t=page_height - max(y0, y1),
        b=page_height - min(y0, y1),
    )


class TaggedStructureModel(BasePageModel):
    """Turn structure elements with page content into pre-labeled layout clusters."""

    def __init__(self, *, mode: TaggedStructureMode) -> None:
        self.mode = mode
        self._structures: dict[int, PdfStructure | None] = {}

    def _structure_for(self, conv_res: ConversionResult) -> PdfStructure | None:
        key = id(conv_res)
        if key not in self._structures:
            backend = conv_res.input._backend
            # Only PDF backends expose the structure tree; the pipeline's input
            # type is not narrowed, so check the backend family explicitly.
            structure = (
                backend.get_structure()
                if isinstance(backend, PdfDocumentBackend)
                else None
            )
            self._structures[key] = structure
        return self._structures[key]

    def _clusters_from_structure(
        self,
        structure: PdfStructure,
        page: Page,
        cells: list[TaggedTextCell],
    ) -> tuple[list[Cluster], TaggedStructurePrediction]:
        assert page.size is not None
        page_index = page.page_no - 1
        page_height = page.size.height
        by_mcid: dict[int, list[TaggedTextCell]] = {}
        for cell in cells:
            if cell.mcid >= 0:
                by_mcid.setdefault(cell.mcid, []).append(cell)

        clusters: list[Cluster] = []
        prediction = TaggedStructurePrediction(used=True)

        def emit(
            label: DocItemLabel, bbox: BoundingBox, element: PdfStructureElement
        ) -> None:
            cluster = Cluster(id=len(clusters), label=label, bbox=bbox, confidence=1.0)
            clusters.append(cluster)
            level = _heading_level(element.resolved_type(structure.role_map))
            if level is not None:
                prediction.heading_levels[cluster.id] = level
            if element.alt:
                prediction.alt_texts[cluster.id] = element.alt

        def visit(element: PdfStructureElement) -> None:
            kind = element.resolved_type(structure.role_map)
            label = _LEAF_LABELS.get(kind)
            if label is None and kind not in _CONTAINERS:
                # Unknown block type: treat as a container so nothing is lost.
                label = None
            if label is None:
                for kid in element.kids:
                    if isinstance(kid, PdfStructureElement):
                        visit(kid)
                return
            boxes = [
                cell.bbox
                for mcid in _mcids_on_page(element, page_index)
                for cell in by_mcid.get(mcid, [])
            ]
            if boxes:
                emit(label, BoundingBox.enclosing_bbox(boxes), element)
                return
            # Figures and formulas are often pure graphics: no text cells, so
            # fall back to the authored Layout /BBox when the element is on this page.
            if element.page == page_index:
                bbox = _layout_bbox(element, page_height)
                if bbox is not None:
                    emit(label, bbox, element)

        for root in structure.elements:
            visit(root)

        # Pagination artifacts (WTPDF 8.3) become furniture; other artifacts carry
        # no label docling can assemble and are left to orphan handling.
        furniture: dict[DocItemLabel, list[BoundingBox]] = {}
        for cell in cells:
            if cell.mcid >= 0 or cell.artifact_type != "/Pagination":
                continue
            subtype = cell.artifact_subtype or ""
            if subtype == "/Header":
                label = DocItemLabel.PAGE_HEADER
            elif subtype == "/Footer":
                label = DocItemLabel.PAGE_FOOTER
            else:
                label = (
                    DocItemLabel.PAGE_HEADER
                    if cell.bbox.t < page_height / 2
                    else DocItemLabel.PAGE_FOOTER
                )
            furniture.setdefault(label, []).append(cell.bbox)
        for label, boxes in furniture.items():
            clusters.append(
                Cluster(
                    id=len(clusters),
                    label=label,
                    bbox=BoundingBox.enclosing_bbox(boxes),
                    confidence=1.0,
                )
            )
        return clusters, prediction

    @staticmethod
    def _merge_uncovered(
        tagged: list[Cluster], predicted: list[Cluster]
    ) -> list[Cluster]:
        """``prefer``: keep model clusters the tags did not account for."""
        merged = list(tagged)
        for cluster in predicted:
            covered = sum(
                cluster.bbox.intersection_over_self(tag.bbox) for tag in tagged
            )
            if covered >= _COVERED_THRESHOLD:
                continue
            merged.append(cluster.model_copy(update={"id": len(merged)}))
        return merged

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            if self.mode == "off" or page._backend is None or page.size is None:
                yield page
                continue

            with TimeRecorder(conv_res, "tagged_structure"):
                structure = self._structure_for(conv_res)
                cells = page._backend.get_marked_content() if structure else []
                if structure is None or not cells:
                    if self.mode == "require":
                        page.predictions.layout = LayoutPrediction(clusters=[])
                        page.predictions.tagged_structure = TaggedStructurePrediction(
                            used=True
                        )
                    yield page
                    continue

                tagged, prediction = self._clusters_from_structure(
                    structure, page, cells
                )
                predicted = (
                    page.predictions.layout.clusters
                    if page.predictions.layout is not None
                    else []
                )
                clusters = (
                    self._merge_uncovered(tagged, predicted)
                    if self.mode == "prefer"
                    else tagged
                )
                page.predictions.layout = LayoutPrediction(clusters=clusters)
                page.predictions.tagged_structure = prediction
                _log.debug(
                    "page %d: %d tagged clusters, %d kept from the layout model",
                    page.page_no,
                    len(tagged),
                    len(clusters) - len(tagged),
                )

            yield page
