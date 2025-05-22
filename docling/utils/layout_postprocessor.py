import bisect
import logging
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import TextCell
from rtree import index

from docling.datamodel.base_models import BoundingBox, Cluster

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

    def get_groups(self) -> Dict[int, List[int]]:
        """Returns groups as {root: [elements]}."""
        groups = defaultdict(list)
        for elem in self.parent:
            groups[self.find(elem)].append(elem)
        return groups


class SpatialClusterIndex:
    """Efficient spatial indexing for clusters using R-tree and interval trees."""

    def __init__(self, clusters: List[Cluster]):
        p = index.Property()
        p.dimension = 2
        self.spatial_index = index.Index(properties=p)
        self.x_intervals = IntervalTree()
        self.y_intervals = IntervalTree()
        self.clusters_by_id: Dict[int, Cluster] = {}

        for cluster in clusters:
            self.add_cluster(cluster)

    def add_cluster(self, cluster: Cluster):
        bbox = cluster.bbox
        self.spatial_index.insert(cluster.id, bbox.as_tuple())
        self.x_intervals.insert(bbox.l, bbox.r, cluster.id)
        self.y_intervals.insert(bbox.t, bbox.b, cluster.id)
        self.clusters_by_id[cluster.id] = cluster

    def remove_cluster(self, cluster: Cluster):
        self.spatial_index.delete(cluster.id, cluster.bbox.as_tuple())
        del self.clusters_by_id[cluster.id]

    def find_candidates(self, bbox: BoundingBox) -> Set[int]:
        """Find potential overlapping cluster IDs using all indexes."""
        spatial = set(self.spatial_index.intersection(bbox.as_tuple()))
        x_candidates = self.x_intervals.find_containing(
            bbox.l
        ) | self.x_intervals.find_containing(bbox.r)
        y_candidates = self.y_intervals.find_containing(
            bbox.t
        ) | self.y_intervals.find_containing(bbox.b)
        return spatial.union(x_candidates).union(y_candidates)

    def check_overlap(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        overlap_threshold: float,
        containment_threshold: float,
    ) -> bool:
        """Check if two bboxes overlap sufficiently."""
        if bbox1.area() <= 0 or bbox2.area() <= 0:
            return False

        iou = bbox1.intersection_over_union(bbox2)
        containment1 = bbox1.intersection_over_self(bbox2)
        containment2 = bbox2.intersection_over_self(bbox1)

        return (
            iou > overlap_threshold
            or containment1 > containment_threshold
            or containment2 > containment_threshold
        )


class Interval:
    """Helper class for sortable intervals."""

    def __init__(self, min_val: float, max_val: float, id: int):
        self.min_val = min_val
        self.max_val = max_val
        self.id = id

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.min_val < other.min_val
        return self.min_val < other


class IntervalTree:
    """Memory-efficient interval tree for 1D overlap queries."""

    def __init__(self):
        self.intervals: List[Interval] = []  # Sorted by min_val

    def insert(self, min_val: float, max_val: float, id: int):
        interval = Interval(min_val, max_val, id)
        bisect.insort(self.intervals, interval)

    def find_containing(self, point: float) -> Set[int]:
        """Find all intervals containing the point."""
        pos = bisect.bisect_left(self.intervals, point)
        result = set()

        # Check intervals starting before point
        for interval in reversed(self.intervals[:pos]):
            if interval.min_val <= point <= interval.max_val:
                result.add(interval.id)
            else:
                break

        # Check intervals starting at/after point
        for interval in self.intervals[pos:]:
            if point <= interval.max_val:
                if interval.min_val <= point:
                    result.add(interval.id)
            else:
                break

        return result


class LayoutPostprocessor:
    """Postprocesses layout predictions by cleaning up clusters and mapping cells."""

    # Cluster type-specific parameters for overlap resolution
    OVERLAP_PARAMS = {
        "regular": {"area_threshold": 1.3, "conf_threshold": 0.05},
        "picture": {"area_threshold": 2.0, "conf_threshold": 0.3},
        "wrapper": {"area_threshold": 2.0, "conf_threshold": 0.2},
    }

    WRAPPER_TYPES = {
        DocItemLabel.FORM,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.TABLE,
        DocItemLabel.DOCUMENT_INDEX,
    }
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

    def __init__(self, cells: List[TextCell], clusters: List[Cluster], page_size: Size):
        """Initialize processor with cells and clusters."""
        """Initialize processor with cells and spatial indices."""
        self.cells = cells
        self.page_size = page_size
        self.all_clusters = clusters
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

    def postprocess(self) -> Tuple[List[Cluster], List[TextCell]]:
        """Main processing pipeline."""
        self.regular_clusters = self._process_regular_clusters()
        self.special_clusters = self._process_special_clusters()

        # Remove regular clusters that are included in wrappers
        contained_ids = {
            child.id
            for wrapper in self.special_clusters
            if wrapper.label in self.SPECIAL_TYPES
            for child in wrapper.children
        }
        self.regular_clusters = [
            c for c in self.regular_clusters if c.id not in contained_ids
        ]

        # Combine and sort final clusters
        final_clusters = self._sort_clusters(
            self.regular_clusters + self.special_clusters, mode="id"
        )
        for cluster in final_clusters:
            cluster.cells = self._sort_cells(cluster.cells)
            # Also sort cells in children if any
            for child in cluster.children:
                child.cells = self._sort_cells(child.cells)

        return final_clusters, self.cells

    def _process_regular_clusters(self) -> List[Cluster]:
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

        # Initial cell assignment
        clusters = self._assign_cells_to_clusters(clusters)

        # Remove clusters with no cells
        clusters = [cluster for cluster in clusters if cluster.cells]

        # Handle orphaned cells
        unassigned = self._find_unassigned_cells(clusters)
        if unassigned:
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

        return clusters

    def _process_special_clusters(self) -> List[Cluster]:
        special_clusters = [
            c
            for c in self.special_clusters
            if c.confidence >= self.CONFIDENCE_THRESHOLDS[c.label]
        ]

        special_clusters = self._handle_cross_type_overlaps(special_clusters)

        # Calculate page area from known page size
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

        for special in special_clusters:
            contained = []
            for cluster in self.regular_clusters:
                containment = cluster.bbox.intersection_over_self(special.bbox)
                if containment > 0.8:
                    contained.append(cluster)

            if contained:
                # Sort contained clusters by minimum cell ID:
                contained = self._sort_clusters(contained, mode="id")
                special.children = contained

                # Adjust bbox only for Form and Key-Value-Region, not Table or Picture
                if special.label in [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]:
                    special.bbox = BoundingBox(
                        l=min(c.bbox.l for c in contained),
                        t=min(c.bbox.t for c in contained),
                        r=max(c.bbox.r for c in contained),
                        b=max(c.bbox.b for c in contained),
                    )

                # Collect all cells from children
                all_cells = []
                for child in contained:
                    all_cells.extend(child.cells)
                special.cells = self._deduplicate_cells(all_cells)
                special.cells = self._sort_cells(special.cells)

        picture_clusters = [
            c for c in special_clusters if c.label == DocItemLabel.PICTURE
        ]
        picture_clusters = self._remove_overlapping_clusters(
            picture_clusters, "picture"
        )

        wrapper_clusters = [
            c for c in special_clusters if c.label in self.WRAPPER_TYPES
        ]
        wrapper_clusters = self._remove_overlapping_clusters(
            wrapper_clusters, "wrapper"
        )

        return picture_clusters + wrapper_clusters

    def _handle_cross_type_overlaps(self, special_clusters) -> List[Cluster]:
        """Handle overlaps between regular and wrapper clusters before child assignment.

        In particular, KEY_VALUE_REGION proposals that are almost identical to a TABLE
        should be removed.
        """
        wrappers_to_remove = set()

        for wrapper in special_clusters:
            if wrapper.label not in self.WRAPPER_TYPES:
                continue  # only treat KEY_VALUE_REGION for now.

            for regular in self.regular_clusters:
                if regular.label == DocItemLabel.TABLE:
                    # Calculate overlap
                    overlap_ratio = wrapper.bbox.intersection_over_self(regular.bbox)

                    conf_diff = wrapper.confidence - regular.confidence

                    # If wrapper is mostly overlapping with a TABLE, remove the wrapper
                    if (
                        overlap_ratio > 0.9 and conf_diff < 0.1
                    ):  # self.OVERLAP_PARAMS["wrapper"]["conf_threshold"]):  # 80% overlap threshold
                        wrappers_to_remove.add(wrapper.id)
                        break

        # Filter out the identified wrappers
        special_clusters = [
            cluster
            for cluster in special_clusters
            if cluster.id not in wrappers_to_remove
        ]

        return special_clusters

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
        group_clusters: List[Cluster],
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
        clusters: List[Cluster],
        cluster_type: str,
        overlap_threshold: float = 0.8,
        containment_threshold: float = 0.8,
    ) -> List[Cluster]:
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
        clusters: List[Cluster],
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

    def _deduplicate_cells(self, cells: List[TextCell]) -> List[TextCell]:
        """Ensure each cell appears only once, maintaining order of first appearance."""
        seen_ids = set()
        unique_cells = []
        for cell in cells:
            if cell.index not in seen_ids:
                seen_ids.add(cell.index)
                unique_cells.append(cell)
        return unique_cells

    def _assign_cells_to_clusters(
        self, clusters: List[Cluster], min_overlap: float = 0.2
    ) -> List[Cluster]:
        """Assign cells to best overlapping cluster."""
        for cluster in clusters:
            cluster.cells = []

        for cell in self.cells:
            if not cell.text.strip():
                continue

            best_overlap = min_overlap
            best_cluster = None

            for cluster in clusters:
                if cell.rect.to_bounding_box().area() <= 0:
                    continue

                overlap_ratio = cell.rect.to_bounding_box().intersection_over_self(
                    cluster.bbox
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

    def _find_unassigned_cells(self, clusters: List[Cluster]) -> List[TextCell]:
        """Find cells not assigned to any cluster."""
        assigned = {cell.index for cluster in clusters for cell in cluster.cells}
        return [
            cell
            for cell in self.cells
            if cell.index not in assigned and cell.text.strip()
        ]

    def _adjust_cluster_bboxes(self, clusters: List[Cluster]) -> List[Cluster]:
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

    def _sort_cells(self, cells: List[TextCell]) -> List[TextCell]:
        """Sort cells in native reading order."""
        return sorted(cells, key=lambda c: (c.index))

    def _sort_clusters(
        self, clusters: List[Cluster], mode: str = "id"
    ) -> List[Cluster]:
        """Sort clusters in reading order (top-to-bottom, left-to-right)."""
        if mode == "id":  # sort in the order the cells are printed in the PDF.
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
