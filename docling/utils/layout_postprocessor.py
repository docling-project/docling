# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import sys
from collections import defaultdict
from itertools import pairwise
from statistics import median

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import TextCell

from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.pipeline_options import BaseLayoutPostprocessorOptions
from docling.datamodel.spatial import (
    BoundingBoxSpatialIndex,
    has_positive_area,
    ordered_bounding_box,
)

_log = logging.getLogger(__name__)


class UnionFind:
    """Efficient Union-Find data structure for grouping elements."""

    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = dict.fromkeys(elements, 0)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self) -> dict[int, list[int]]:
        """Returns groups as {root: [elements]}."""
        groups = defaultdict(list)
        for elem in self.parent:
            groups[self.find(elem)].append(elem)
        return groups


class SpatialClusterIndex:
    """Efficient spatial indexing for clusters using an R-tree."""

    def __init__(self, clusters: list[Cluster]):
        self.spatial_index = BoundingBoxSpatialIndex()
        self.clusters_by_id: dict[int, Cluster] = {}

        for cluster in clusters:
            self.add_cluster(cluster)

    def add_cluster(self, cluster: Cluster):
        self.spatial_index.insert(cluster.id, cluster.bbox)
        self.clusters_by_id[cluster.id] = cluster

    def remove_cluster(self, cluster: Cluster):
        self.spatial_index.delete(cluster.id, cluster.bbox)
        del self.clusters_by_id[cluster.id]

    def find_candidates(self, bbox: BoundingBox) -> set[int]:
        """Find cluster IDs whose bbox intersects ``bbox``.

        Every relation ``check_overlap`` accepts -- IoU or either containment
        ratio above a positive threshold -- requires a positive intersection
        area, so the R-tree query is already a superset of the pairs that can
        survive. Widening it further only adds candidates that are then
        rejected.
        """
        return set(self.spatial_index.intersection(bbox))

    def check_overlap(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        overlap_threshold: float,
        containment_threshold: float,
    ) -> bool:
        """Check if two bboxes overlap sufficiently."""
        bbox1 = ordered_bounding_box(bbox1)
        bbox2 = ordered_bounding_box(bbox2)
        if not has_positive_area(bbox1) or not has_positive_area(bbox2):
            return False

        iou = bbox1.intersection_over_union(bbox2)
        containment1 = bbox1.intersection_over_self(bbox2)
        containment2 = bbox2.intersection_over_self(bbox1)

        return (
            iou > overlap_threshold
            or containment1 > containment_threshold
            or containment2 > containment_threshold
        )


class LayoutPostprocessor:
    """Postprocesses layout predictions by cleaning up clusters and mapping cells."""

    # Cluster type-specific parameters for overlap resolution
    OVERLAP_PARAMS = {
        "regular": {"area_threshold": 1.3, "conf_threshold": 0.05},
        "picture": {"area_threshold": 2.0, "conf_threshold": 0.3},
        "wrapper": {"area_threshold": 2.0, "conf_threshold": 0.2},
    }

    CONTAINER_TYPES = {
        DocItemLabel.FORM,
        DocItemLabel.KEY_VALUE_REGION,
    }
    TABLE_TYPES = {
        DocItemLabel.TABLE,
        DocItemLabel.DOCUMENT_INDEX,
    }
    WRAPPER_TYPES = CONTAINER_TYPES.union(TABLE_TYPES)
    SPECIAL_TYPES = WRAPPER_TYPES.union({DocItemLabel.PICTURE})

    CONFIDENCE_THRESHOLDS = {
        DocItemLabel.CAPTION: 0.5,
        DocItemLabel.FOOTNOTE: 0.5,
        DocItemLabel.FORMULA: 0.5,
        DocItemLabel.LIST_ITEM: 0.5,
        DocItemLabel.PAGE_FOOTER: 0.5,
        DocItemLabel.PAGE_HEADER: 0.5,
        DocItemLabel.PICTURE: 0.5,
        DocItemLabel.SECTION_HEADER: 0.45,
        DocItemLabel.TABLE: 0.5,
        DocItemLabel.TEXT: 0.5,  # 0.45,
        DocItemLabel.TITLE: 0.45,
        DocItemLabel.CODE: 0.45,
        DocItemLabel.CHECKBOX_SELECTED: 0.45,
        DocItemLabel.CHECKBOX_UNSELECTED: 0.45,
        DocItemLabel.FORM: 0.45,
        DocItemLabel.KEY_VALUE_REGION: 0.45,
        DocItemLabel.DOCUMENT_INDEX: 0.45,
    }

    LABEL_REMAPPING = {
        # DocItemLabel.DOCUMENT_INDEX: DocItemLabel.TABLE,
        DocItemLabel.TITLE: DocItemLabel.SECTION_HEADER,
    }

    def __init__(
        self,
        page: Page,
        clusters: list[Cluster],
        options: BaseLayoutPostprocessorOptions,
    ) -> None:
        """Initialize processor with page and clusters."""

        self.cells = page.cells
        self.page = page
        self.page_size = page.size
        self.all_clusters = clusters
        self.options = options
        self.regular_clusters = [
            c for c in clusters if c.label not in self.SPECIAL_TYPES
        ]
        self.special_clusters = [c for c in clusters if c.label in self.SPECIAL_TYPES]

        # Build spatial indices once
        self.regular_index = SpatialClusterIndex(self.regular_clusters)
        self.picture_index = SpatialClusterIndex(
            [c for c in self.special_clusters if c.label == DocItemLabel.PICTURE]
        )
        self.wrapper_index = SpatialClusterIndex(
            [c for c in self.special_clusters if c.label in self.WRAPPER_TYPES]
        )

    def postprocess(self) -> list[Cluster]:
        """Main processing pipeline."""
        self.regular_clusters = self._process_regular_clusters()
        self.special_clusters = self._process_special_clusters()

        # Remove regular clusters that are included in wrappers
        contained_ids = {
            child.id
            for wrapper in self.special_clusters
            if wrapper.label in self.TABLE_TYPES
            or wrapper.label == DocItemLabel.PICTURE
            for child in wrapper.children
        }
        self.regular_clusters = [
            c for c in self.regular_clusters if c.id not in contained_ids
        ]

        # Keep a deterministic assembly order. Semantic reading order is predicted later.
        final_clusters = self._sort_clusters(
            self.regular_clusters + self.special_clusters, mode="id"
        )

        # Conditionally process cells if not skipping cell assignment
        if not self.options.skip_cell_assignment:
            for cluster in final_clusters:
                cluster.cells = self._sort_cells(cluster.cells)
                # Also sort cells in children if any
                for child in cluster.children:
                    child.cells = self._sort_cells(child.cells)

            # parsed_page is absent when native cell extraction was skipped
            # (PagePreprocessingOptions.skip_cell_extraction); there are no
            # textline cells to write back in that case.
            if self.page.parsed_page is not None:
                self.page.parsed_page.textline_cells = self.cells
                self.page.parsed_page.has_lines = len(self.cells) > 0

        return final_clusters

    def _process_regular_clusters(self) -> list[Cluster]:
        """Process regular clusters with iterative refinement."""
        clusters = [
            c
            for c in self.regular_clusters
            if c.confidence >= self.CONFIDENCE_THRESHOLDS[c.label]
        ]

        # Apply label remapping
        for cluster in clusters:
            if cluster.label in self.LABEL_REMAPPING:
                cluster.label = self.LABEL_REMAPPING[cluster.label]

        # Conditionally assign cells to clusters
        if not self.options.skip_cell_assignment:
            clusters = self._remove_redundant_text_clusters(clusters)
            # Initial cell assignment
            clusters = self._assign_cells_to_clusters(clusters)

            # Remove clusters with no cells (if keep_empty_clusters is False),
            # but always keep clusters with label DocItemLabel.FORMULA
            if not self.options.keep_empty_clusters:
                clusters = [
                    cluster
                    for cluster in clusters
                    if cluster.cells or cluster.label == DocItemLabel.FORMULA
                ]

            # Preserve orphan cells as ordinary text clusters. Their source-cell order is
            # only an assembly tie-break; the reading-order stage still orders them.
            unassigned = self._find_unassigned_cells(clusters)
            if unassigned and self.options.create_orphan_clusters:
                next_id = max((c.id for c in self.all_clusters), default=0) + 1
                orphan_clusters = []
                for i, cell in enumerate(unassigned):
                    conf = cell.confidence

                    orphan_clusters.append(
                        Cluster(
                            id=next_id + i,
                            label=DocItemLabel.TEXT,
                            bbox=cell.to_bounding_box(),
                            confidence=conf,
                            cells=[cell],
                        )
                    )
                clusters.extend(orphan_clusters)

        # Iterative refinement
        prev_count = len(clusters) + 1
        for _ in range(3):  # Maximum 3 iterations
            if prev_count == len(clusters):
                break
            prev_count = len(clusters)
            clusters = self._adjust_cluster_bboxes(clusters)
            clusters = self._remove_overlapping_clusters(clusters, "regular")

        if not self.options.skip_cell_assignment:
            clusters = self._split_text_clusters_on_paragraph_gaps(clusters)

        return clusters

    def _process_special_clusters(self) -> list[Cluster]:
        special_clusters = [
            c
            for c in self.special_clusters
            if c.confidence >= self.CONFIDENCE_THRESHOLDS[c.label]
        ]

        # Calculate page area from known page size
        assert self.page_size is not None
        page_area = self.page_size.width * self.page_size.height
        if page_area > 0:
            # Filter out full-page pictures
            special_clusters = [
                cluster
                for cluster in special_clusters
                if not (
                    cluster.label == DocItemLabel.PICTURE
                    and cluster.bbox.area() / page_area > 0.90
                )
            ]

        picture_clusters = [
            c for c in special_clusters if c.label == DocItemLabel.PICTURE
        ]
        picture_clusters = self._remove_overlapping_clusters(
            picture_clusters, "picture"
        )

        table_clusters = [c for c in special_clusters if c.label in self.TABLE_TYPES]
        table_clusters = self._remove_overlapping_clusters(table_clusters, "wrapper")

        container_clusters = [
            c for c in special_clusters if c.label in self.CONTAINER_TYPES
        ]
        container_clusters = self._remove_overlapping_clusters(
            container_clusters, "wrapper"
        )

        special_clusters = self._handle_cross_type_overlaps(
            picture_clusters + table_clusters + container_clusters
        )
        picture_clusters = [
            cluster
            for cluster in special_clusters
            if cluster.label == DocItemLabel.PICTURE
        ]
        table_clusters = [
            cluster for cluster in special_clusters if cluster.label in self.TABLE_TYPES
        ]
        container_clusters = [
            cluster
            for cluster in special_clusters
            if cluster.label in self.CONTAINER_TYPES
        ]

        nested_clusters = table_clusters + picture_clusters
        for cluster in nested_clusters:
            children = [
                regular
                for regular in self.regular_clusters
                if regular.bbox.intersection_over_self(cluster.bbox) > 0.8
            ]
            self._set_cluster_children(cluster, children)

        parent_by_child_id = {}
        for child in nested_clusters:
            parents = [
                container
                for container in container_clusters
                if child.bbox.intersection_over_self(container.bbox) > 0.8
            ]
            if parents:
                parent = min(
                    parents,
                    key=lambda container: (
                        container.bbox.area(),
                        -container.confidence,
                        container.id,
                    ),
                )
                parent_by_child_id[child.id] = parent.id

        nested_regular_ids = {
            regular.id for child in nested_clusters for regular in child.children
        }
        parent_by_regular_id = {}
        for child in self.regular_clusters:
            if child.id in nested_regular_ids:
                continue
            parents = [
                container
                for container in container_clusters
                if child.bbox.intersection_over_self(container.bbox) > 0.8
            ]
            if parents:
                parent = min(
                    parents,
                    key=lambda container: (
                        container.bbox.area(),
                        -container.confidence,
                        container.id,
                    ),
                )
                parent_by_regular_id[child.id] = parent.id

        for container in container_clusters:
            nested_children = [
                child
                for child in nested_clusters
                if parent_by_child_id.get(child.id) == container.id
            ]
            direct_children = [
                regular
                for regular in self.regular_clusters
                if parent_by_regular_id.get(regular.id) == container.id
            ]
            self._set_cluster_children(container, direct_children + nested_children)

        return picture_clusters + table_clusters + container_clusters

    def _set_cluster_children(self, cluster: Cluster, children: list[Cluster]) -> None:
        if not children:
            return

        cluster.children = self._sort_clusters(children, mode="id")

        if cluster.label in self.CONTAINER_TYPES:
            cluster.bbox = BoundingBox(
                l=min(child.bbox.l for child in cluster.children),
                t=min(child.bbox.t for child in cluster.children),
                r=max(child.bbox.r for child in cluster.children),
                b=max(child.bbox.b for child in cluster.children),
            )

        if not self.options.skip_cell_assignment:
            cluster.cells = self._deduplicate_cells(
                [cell for child in cluster.children for cell in child.cells]
            )
            cluster.cells = self._sort_cells(cluster.cells)
        else:
            cluster.cells = []

    @staticmethod
    def _resolve_coincident_pairs(
        losers: list[Cluster],
        winners: list[Cluster],
        iou_threshold: float = 0.8,
        conf_tolerance: float = 0.1,
    ) -> set[int]:
        """Elect a winner for near-identical (loser, winner) label pairs.

        For each pair whose bboxes are near-identical (``IoU > iou_threshold``)
        AND whose confidences are within ``conf_tolerance`` of each other, the
        loser label is dropped so the winner label survives. Nothing else --
        containment, area, downstream behaviour -- is considered.

        Pairs outside this envelope (low IoU, or a clearly more confident
        loser) are left alone for other passes to handle.
        """
        to_drop: set[int] = set()
        for loser in losers:
            for winner in winners:
                if loser.bbox.intersection_over_union(winner.bbox) <= iou_threshold:
                    continue
                if (loser.confidence - winner.confidence) < conf_tolerance:
                    to_drop.add(loser.id)
                    break
        return to_drop

    def _handle_cross_type_overlaps(self, special_clusters) -> list[Cluster]:
        """Elect a winner for cross-type label pairs at near-identical bboxes.

        The layout model can emit the same grounded region under several
        labels. When two labels sit at near-identical bboxes with similar
        confidence, this step picks the label carrying the richer downstream
        semantic. Anything outside that envelope is out of scope here.

        | pair                                | loser     | winner                 |
        |-------------------------------------|-----------|------------------------|
        | TABLE vs DOCUMENT_INDEX             | TABLE     | DOCUMENT_INDEX         |
        | PICTURE vs TABLE / DOC_INDEX        | PICTURE   | TABLE / DOCUMENT_INDEX |
        | FORM / KVR vs TABLE / DOC / PICTURE | container | structured element     |
        """
        tables = [c for c in special_clusters if c.label == DocItemLabel.TABLE]
        doc_indices = [
            c for c in special_clusters if c.label == DocItemLabel.DOCUMENT_INDEX
        ]
        pictures = [c for c in special_clusters if c.label == DocItemLabel.PICTURE]
        containers = [c for c in special_clusters if c.label in self.CONTAINER_TYPES]

        clusters_to_remove: set[int] = set()
        clusters_to_remove |= self._resolve_coincident_pairs(tables, doc_indices)
        clusters_to_remove |= self._resolve_coincident_pairs(
            pictures, tables + doc_indices
        )
        surviving_structured = [
            c for c in tables + doc_indices + pictures if c.id not in clusters_to_remove
        ]
        clusters_to_remove |= self._resolve_coincident_pairs(
            containers, surviving_structured
        )

        return [c for c in special_clusters if c.id not in clusters_to_remove]

    def _should_prefer_cluster(
        self, candidate: Cluster, other: Cluster, params: dict
    ) -> bool:
        """Determine if candidate cluster should be preferred over other cluster based on rules.
        Returns True if candidate should be preferred, False if not."""

        # Rule 1: LIST_ITEM vs TEXT
        if (
            candidate.label == DocItemLabel.LIST_ITEM
            and other.label == DocItemLabel.TEXT
        ):
            # Check if areas are similar (within 20% of each other)
            area_ratio = candidate.bbox.area() / other.bbox.area()
            area_similarity = abs(1 - area_ratio) < 0.2
            if area_similarity:
                return True

        # Rule 2: CODE vs others
        if candidate.label == DocItemLabel.CODE:
            # Calculate how much of the other cluster is contained within the CODE cluster
            containment = other.bbox.intersection_over_self(candidate.bbox)
            if containment > 0.8:  # other is 80% contained within CODE
                return True

        # If no label-based rules matched, fall back to area/confidence thresholds
        area_ratio = candidate.bbox.area() / other.bbox.area()
        conf_diff = other.confidence - candidate.confidence

        if (
            area_ratio <= params["area_threshold"]
            and conf_diff > params["conf_threshold"]
        ):
            return False

        return True  # Default to keeping candidate if no rules triggered rejection

    def _select_best_cluster_from_group(
        self,
        group_clusters: list[Cluster],
        params: dict,
    ) -> Cluster:
        """Select best cluster from a group of overlapping clusters based on all rules."""
        current_best = None

        for candidate in group_clusters:
            should_select = True

            for other in group_clusters:
                if other == candidate:
                    continue

                if not self._should_prefer_cluster(candidate, other, params):
                    should_select = False
                    break

            if should_select:
                if current_best is None:
                    current_best = candidate
                else:
                    # If both clusters pass rules, prefer the larger one unless confidence differs significantly
                    if (
                        candidate.bbox.area() > current_best.bbox.area()
                        and current_best.confidence - candidate.confidence
                        <= params["conf_threshold"]
                    ):
                        current_best = candidate

        return current_best if current_best else group_clusters[0]

    def _remove_overlapping_clusters(
        self,
        clusters: list[Cluster],
        cluster_type: str,
        overlap_threshold: float = 0.8,
        containment_threshold: float = 0.8,
    ) -> list[Cluster]:
        if not clusters:
            return []

        spatial_index = (
            self.regular_index
            if cluster_type == "regular"
            else self.picture_index
            if cluster_type == "picture"
            else self.wrapper_index
        )

        # Map of currently valid clusters
        valid_clusters = {c.id: c for c in clusters}
        uf = UnionFind(valid_clusters.keys())
        params = self.OVERLAP_PARAMS[cluster_type]

        for cluster in clusters:
            candidates = spatial_index.find_candidates(cluster.bbox)
            candidates &= valid_clusters.keys()  # Only keep existing candidates
            candidates.discard(cluster.id)

            for other_id in candidates:
                if spatial_index.check_overlap(
                    cluster.bbox,
                    valid_clusters[other_id].bbox,
                    overlap_threshold,
                    containment_threshold,
                ):
                    uf.union(cluster.id, other_id)

        result = []
        for group in uf.get_groups().values():
            if len(group) == 1:
                result.append(valid_clusters[group[0]])
                continue

            group_clusters = [valid_clusters[cid] for cid in group]
            best = self._select_best_cluster_from_group(group_clusters, params)

            # Simple cell merging - no special cases
            for cluster in group_clusters:
                if cluster != best:
                    best.cells.extend(cluster.cells)

            best.cells = self._deduplicate_cells(best.cells)
            best.cells = self._sort_cells(best.cells)
            result.append(best)

        return result

    def _select_best_cluster(
        self,
        clusters: list[Cluster],
        area_threshold: float,
        conf_threshold: float,
    ) -> Cluster:
        """Iteratively select best cluster based on area and confidence thresholds."""
        current_best = None
        for candidate in clusters:
            should_select = True
            for other in clusters:
                if other == candidate:
                    continue

                area_ratio = candidate.bbox.area() / other.bbox.area()
                conf_diff = other.confidence - candidate.confidence

                if area_ratio <= area_threshold and conf_diff > conf_threshold:
                    should_select = False
                    break

            if should_select:
                if current_best is None or (
                    candidate.bbox.area() > current_best.bbox.area()
                    and current_best.confidence - candidate.confidence <= conf_threshold
                ):
                    current_best = candidate

        return current_best if current_best else clusters[0]

    def _deduplicate_cells(self, cells: list[TextCell]) -> list[TextCell]:
        """Ensure each cell appears only once, maintaining order of first appearance."""
        seen_ids = set()
        unique_cells = []
        for cell in cells:
            if cell.index not in seen_ids:
                seen_ids.add(cell.index)
                unique_cells.append(cell)
        return unique_cells

    def _remove_redundant_text_clusters(
        self,
        clusters: list[Cluster],
        min_overlap: float = 0.2,
        containment_threshold: float = 0.8,
    ) -> list[Cluster]:
        """Resolve redundant TEXT predictions before they compete for individual lines.

        A slightly wider, weaker prediction can otherwise take isolated lines
        from several paragraphs. Once boxes are fitted to those cells, overlap
        resolution can no longer recover the original paragraph boundaries.
        Only suppress a prediction when retained, more confident TEXT regions
        cover all of its cells comparably; unique continuations must survive.
        """
        text_clusters = [c for c in clusters if c.label == DocItemLabel.TEXT]
        if len(text_clusters) < 2:
            return clusters

        spatial_index = SpatialClusterIndex(clusters)
        by_id = {c.id: c for c in clusters}
        overlaps: dict[int, dict[int, float]] = {c.id: {} for c in text_clusters}
        competing_overlaps: dict[int, float] = {}
        for cell in self.cells:
            cell_bbox = cell.rect.to_bounding_box()
            if not cell.text.strip() or not has_positive_area(cell_bbox):
                continue
            for cid in spatial_index.find_candidates(cell_bbox):
                ratio = cell_bbox.intersection_over_self(
                    ordered_bounding_box(by_id[cid].bbox)
                )
                if ratio > min_overlap:
                    if by_id[cid].label == DocItemLabel.TEXT:
                        overlaps[cid][cell.index] = ratio
                    else:
                        competing_overlaps[cell.index] = max(
                            competing_overlaps.get(cell.index, 0.0), ratio
                        )

        redundant: set[int] = set()
        retained: set[int] = set()
        for cluster in sorted(text_clusters, key=lambda c: c.confidence, reverse=True):
            alternatives = {
                cid
                for cid in spatial_index.find_candidates(cluster.bbox) & retained
                if spatial_index.check_overlap(
                    cluster.bbox,
                    by_id[cid].bbox,
                    containment_threshold,
                    containment_threshold,
                )
            }
            # Use the overlap-resolution tolerance for cell coverage too. A
            # narrow fragment must not replace a substantially better fit.
            if overlaps[cluster.id] and all(
                any(
                    overlaps[cid].get(cell_id, 0.0) >= ratio * containment_threshold
                    # Do not expose a competing heading, list item, etc. to
                    # cells previously protected by this TEXT prediction.
                    and overlaps[cid].get(cell_id, 0.0)
                    > competing_overlaps.get(cell_id, 0.0)
                    for cid in alternatives
                )
                for cell_id, ratio in overlaps[cluster.id].items()
            ):
                redundant.add(cluster.id)
            else:
                retained.add(cluster.id)

        return [c for c in clusters if c.id not in redundant]

    def _split_text_clusters_on_paragraph_gaps(
        self, clusters: list[Cluster], min_confidence: float = 0.3
    ) -> list[Cluster]:
        """Recover paragraphs corroborated by whitespace and nested predictions.

        A confident parent can survive while a paragraph proposal falls below
        the confidence threshold. Use that proposal only as boundary evidence:
        it cannot take cells or change their labels. Require a full line-height
        of whitespace, and multiple lines per paragraph when not every group
        has its own prediction. Do not split overlapping lines or columns.
        """
        result: list[Cluster] = []
        next_id = max((c.id for c in self.all_clusters + clusters), default=0) + 1
        proposals = [
            c
            for c in self.regular_clusters
            if c.label == DocItemLabel.TEXT and c.confidence >= min_confidence
        ]
        for cluster in clusters:
            if (
                cluster.label != DocItemLabel.TEXT
                or len(cluster.cells) < 2
                or cluster.children
            ):
                result.append(cluster)
                continue
            cells = sorted(cluster.cells, key=lambda c: c.rect.to_bounding_box().t)
            boxes = [cell.rect.to_bounding_box() for cell in cells]
            line_height = median(box.height for box in boxes)
            if line_height <= 0 or any(
                after.t < before.b for before, after in pairwise(boxes)
            ):
                result.append(cluster)
                continue
            groups: list[list[TextCell]] = [[cells[0]]]
            for before, box, cell in zip(boxes, boxes[1:], cells[1:]):
                if box.t - before.b >= line_height:
                    groups.append([])
                groups[-1].append(cell)
            if len(groups) < 2:
                result.append(cluster)
                continue

            group_lefts = [
                median(c.rect.to_bounding_box().l for c in g) for g in groups
            ]
            if max(group_lefts) - min(group_lefts) > line_height:
                result.append(cluster)
                continue
            corroborated = 0
            for group in groups:
                group_ids = {cell.index for cell in group}
                if any(
                    proposal.id != cluster.id
                    and ordered_bounding_box(proposal.bbox).intersection_over_self(
                        cluster.bbox
                    )
                    >= 0.8
                    and all(
                        cell.rect.to_bounding_box().intersection_over_self(
                            ordered_bounding_box(proposal.bbox)
                        )
                        >= 0.8
                        for cell in group
                    )
                    and all(
                        cell.rect.to_bounding_box().intersection_over_self(
                            ordered_bounding_box(proposal.bbox)
                        )
                        <= 0.2
                        for cell in cells
                        if cell.index not in group_ids
                    )
                    for proposal in proposals
                ):
                    corroborated += 1
            if not corroborated or (
                corroborated < len(groups) and any(len(g) < 2 for g in groups)
            ):
                result.append(cluster)
                continue

            for index, group in enumerate(groups):
                paragraph = cluster.model_copy(
                    update={"id": cluster.id if index == 0 else next_id, "cells": group}
                )
                if index > 0:
                    next_id += 1
                result.extend(self._adjust_cluster_bboxes([paragraph]))
        return result

    def _assign_cells_to_clusters(
        self, clusters: list[Cluster], min_overlap: float = 0.2
    ) -> list[Cluster]:
        """Assign cells to best overlapping cluster."""
        for cluster in clusters:
            cluster.cells = []

        cluster_by_id = {cluster.id: cluster for cluster in clusters}
        cluster_ids = set(cluster_by_id)
        cluster_order = {cluster.id: order for order, cluster in enumerate(clusters)}
        cluster_index = SpatialClusterIndex(clusters)

        for cell in self.cells:
            if not cell.text.strip():
                continue

            cell_bbox = cell.rect.to_bounding_box()
            if cell_bbox.area() <= 0:
                continue

            best_overlap = min_overlap
            best_cluster = None
            candidate_ids = cluster_index.find_candidates(cell_bbox) & cluster_ids

            for cluster_id in sorted(candidate_ids, key=cluster_order.__getitem__):
                cluster = cluster_by_id[cluster_id]

                overlap_ratio = cell_bbox.intersection_over_self(
                    ordered_bounding_box(cluster.bbox)
                )
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cluster = cluster

            if best_cluster is not None:
                best_cluster.cells.append(cell)

        # Deduplicate cells in each cluster after assignment
        for cluster in clusters:
            cluster.cells = self._deduplicate_cells(cluster.cells)

        return clusters

    def _find_unassigned_cells(self, clusters: list[Cluster]) -> list[TextCell]:
        """Find cells not assigned to any cluster."""
        assigned = {cell.index for cluster in clusters for cell in cluster.cells}
        return [
            cell
            for cell in self.cells
            if cell.index not in assigned and cell.text.strip()
        ]

    def _adjust_cluster_bboxes(self, clusters: list[Cluster]) -> list[Cluster]:
        """Adjust cluster bounding boxes to contain their cells."""
        for cluster in clusters:
            if not cluster.cells:
                continue

            cells_bbox = BoundingBox(
                l=min(cell.rect.to_bounding_box().l for cell in cluster.cells),
                t=min(cell.rect.to_bounding_box().t for cell in cluster.cells),
                r=max(cell.rect.to_bounding_box().r for cell in cluster.cells),
                b=max(cell.rect.to_bounding_box().b for cell in cluster.cells),
            )

            if cluster.label == DocItemLabel.TABLE:
                # For tables, take union of current bbox and cells bbox
                cluster.bbox = BoundingBox(
                    l=min(cluster.bbox.l, cells_bbox.l),
                    t=min(cluster.bbox.t, cells_bbox.t),
                    r=max(cluster.bbox.r, cells_bbox.r),
                    b=max(cluster.bbox.b, cells_bbox.b),
                )
            else:
                cluster.bbox = cells_bbox

        return clusters

    def _sort_cells(self, cells: list[TextCell]) -> list[TextCell]:
        """Sort cells by their source/parser index."""
        return sorted(cells, key=lambda c: c.index)

    def _sort_clusters(
        self, clusters: list[Cluster], mode: str = "id"
    ) -> list[Cluster]:
        """Sort clusters for deterministic layout-stage storage."""
        if mode == "id":  # Source-cell order, with geometry for empty/tied clusters.
            return sorted(
                clusters,
                key=lambda cluster: (
                    (
                        min(cell.index for cell in cluster.cells)
                        if cluster.cells
                        else sys.maxsize
                    ),
                    cluster.bbox.t,
                    cluster.bbox.l,
                ),
            )
        elif mode == "tblr":  # Sort top-to-bottom, then left-to-right ("row first")
            return sorted(
                clusters, key=lambda cluster: (cluster.bbox.t, cluster.bbox.l)
            )
        elif mode == "lrtb":  # Sort left-to-right, then top-to-bottom ("column first")
            return sorted(
                clusters, key=lambda cluster: (cluster.bbox.l, cluster.bbox.t)
            )
        else:
            return clusters
