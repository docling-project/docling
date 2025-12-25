from collections import Counter
from enum import Enum
from functools import cached_property
from io import BytesIO
from pathlib import PurePath
from typing import Any, Optional, Union

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel, ListItem, TextItem
from docling_core.types.doc.document import (
    NodeItem,
    SectionHeaderItem,
)
from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement as ReadingOrderPageElement,
)
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from docling.datamodel.base_models import DocumentStream, PageElement, TextElement
from docling.datamodel.document import ConversionResult
from docling.models.header_hierarchy.parsers import (
    infer_header_level_letter,
    infer_header_level_numerical,
    infer_header_level_roman,
)
from docling.models.header_hierarchy.types.enums import NumberingLevel, StyleAttributes
from docling.models.header_hierarchy.types.hierarchical_header import HierarchicalHeader


class InconsistentNumberingException(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Inconsistent numbering - either reading order is messed up or header numbering is not intended. Extract hierarchy from style only."
        )


class ImplausibleHeadingStructureException(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Hierarchy demands equal level heading, but no common parent was found!"
        )


class DoclingResultNotReadyException(Exception):
    def __init__(self) -> None:
        super().__init__(
            "It seems that the docling result has not been filled / is not ready for postprocessing."
        )


class ItemNotRegisteredAsChildException(Exception):
    def __init__(self, item: NodeItem):
        super().__init__(
            f"The item {item} does not seem to be registered as a child of its parent node!"
        )


class ItemInconsitencyException(Exception):
    pass


class DocumentHierarchyBuilder:
    def __init__(
        self,
        headings: list[dict],
        min_prop_numbered: float = 0.3,
        infer_hierarchy_from_style: bool = True,
        infer_hierarchy_from_numbering: bool = True,
    ):
        self.headings: list[dict] = headings
        self._features: Optional[list[dict[Enum, Any]]] = None
        self._style_features: Optional[list[dict[StyleAttributes, float]]] = None
        self.min_prop_numbered: float = min_prop_numbered
        self.infer_hierarchy_from_style: bool = infer_hierarchy_from_style
        self.infer_hierarchy_from_numbering: bool = infer_hierarchy_from_numbering

    @property
    def features(self) -> list[dict[Enum, Any]]:
        """
        Extract features from heading properties for number-based hierarchy inference.
        """
        if self._features is None:
            features = []
            for heading in self.headings:
                feature_dict = {
                    StyleAttributes.font_size: heading["font_size"],
                    StyleAttributes.bold: 1.0 if heading["is_bold"] else 0.0,
                    StyleAttributes.italic: 1.0 if heading["is_italic"] else 0.0,
                    NumberingLevel.level_latin: infer_header_level_roman(
                        heading["text"]
                    ),
                    NumberingLevel.level_alpha: infer_header_level_letter(
                        heading["text"]
                    ),
                    NumberingLevel.level_numerical: infer_header_level_numerical(
                        heading["text"]
                    ),
                }
                features.append(feature_dict)
            self._features = features

        return self._features

    @property
    def style_features(self) -> list[dict[StyleAttributes, float]]:
        """
        Extract features from heading properties for clustering.
        """
        if self._style_features is None:
            features = []
            for heading in self.headings:
                feature_dict = {
                    StyleAttributes.font_size: heading["font_size"],
                    StyleAttributes.bold: 1.0 if heading["is_bold"] else 0.0,
                    StyleAttributes.italic: 1.0 if heading["is_italic"] else 0.0,
                }
                features.append(feature_dict)
            self._style_features = features

        return self._style_features

    @staticmethod
    def _gt_hierarchical(left: list[int], right: list[int]) -> bool:
        for i, j in zip(left, right):
            if i < j:
                return True
            if i > j:
                return False
        return len(right) > len(left)

    def _infer_from_numbering(
        self,
        start_numbering_at_root: bool = True,
        ignore_numbering_of_title: bool = False,
    ) -> Union[HierarchicalHeader, None]:
        numbering_types = NumberingLevel.__members__.values()

        # If the title starts with an "I " or an "A " or something like that then this can upset the whole logic - catch that...
        features = self.features
        if ignore_numbering_of_title:
            # this logic is too simple - it would be necessary to actually try and judge whether there is a heading
            font_size_clusters = self._cluster_headings_dbscan()
            # assume biggest font size is title:
            title_indices = [i for i, v in font_size_clusters.items() if v == 1]
            if len(title_indices) == 1:
                for k in numbering_types:
                    features[title_indices[0]][k] = []

        # some sort of heuristic on whether numbered headings exist:
        prop_numbered = len(
            [f for f in features if any(len(f[k]) > 0 for k in numbering_types)]
        ) / len(self.headings)
        if prop_numbered < self.min_prop_numbered:
            return None
        # assert that there are never multiple numbering types active at once:
        for f in features:
            if len([k for k in numbering_types if f[k]]) > 1:
                raise InconsistentNumberingException()
        root = HierarchicalHeader()
        current = root
        first_number = True

        for i, f in enumerate(features):
            new_parent = None
            kwargs = {}
            for k in numbering_types:
                kval = k.value
                if new_level := f[k]:
                    kwargs = {k.value: new_level}
                    if first_number and start_numbering_at_root:
                        new_parent = root
                    elif current_level := getattr(current, kval):
                        if self._gt_hierarchical(current_level, new_level):
                            if len(new_level) > len(current_level):
                                if current_level == new_level[: len(current_level)]:
                                    new_parent = current
                                    break
                                else:
                                    new_parent = current
                                    while (
                                        new_parent.parent is not None
                                        and hasattr(new_parent, kval)
                                        and (
                                            getattr(new_parent, kval)
                                            != new_level[
                                                : len(getattr(new_parent, kval))
                                            ]
                                        )
                                    ):
                                        new_parent = new_parent.parent
                                    break
                            else:
                                new_parent = current
                                while new_parent.parent is not None and (
                                    len(getattr(new_parent, kval)) >= len(new_level)
                                    or (
                                        hasattr(new_parent, kval)
                                        and (
                                            getattr(new_parent, kval)
                                            != new_level[
                                                : len(getattr(new_parent, kval))
                                            ]
                                        )
                                    )
                                ):
                                    new_parent = new_parent.parent
                                break
                        else:
                            raise InconsistentNumberingException()
                    else:
                        # switch of numbering style!
                        base_obj = current
                        # alternative: collect all possible parents and their numberings then select the one that has the minimum distance.
                        while True:
                            # serach upwards in a tree if this numbering type is continued at that point.
                            last_level, last_level_obj = base_obj.last_level_of_kind(k)
                            if last_level_obj is None:
                                # didn't find anything
                                break
                            if last_level and self._gt_hierarchical(
                                last_level, new_level
                            ):
                                # a higher order hierarchy is continued
                                if len(new_level) > len(last_level):
                                    # new_parent = last_level_obj
                                    # break
                                    if last_level == new_level[: len(last_level)]:
                                        new_parent = last_level_obj
                                        break
                                    else:
                                        new_parent = last_level_obj
                                        while (
                                            new_parent.parent is not None
                                            and hasattr(new_parent, kval)
                                            and (
                                                getattr(new_parent, kval)
                                                != new_level[
                                                    : len(getattr(new_parent, kval))
                                                ]
                                            )
                                        ):
                                            new_parent = new_parent.parent
                                        break
                                elif len(new_level) < len(last_level):
                                    pass
                                else:
                                    if (
                                        new_level[: max(len(last_level) - 1, 0)]
                                        != last_level[: max(len(last_level) - 1, 0)]
                                    ):
                                        raise InconsistentNumberingException()
                                    new_parent = last_level_obj.parent
                                    break
                            base_obj = last_level_obj
                        if new_parent is None:
                            # a new level is appended
                            new_parent = current
                            break
                    first_number = False
            if not new_parent:
                new_parent = (
                    current
                    if current.any_level() or current.parent is None
                    else current.parent
                )
            new_obj = HierarchicalHeader(
                index=i,
                text=self.headings[i]["text"],
                parent=new_parent,
                doc_ref=self.headings[i].get("reference"),
                **kwargs,
            )
            new_parent.children.append(new_obj)
            current = new_obj
        return root

    def _infer_by_clustering(self) -> HierarchicalHeader:
        heading_to_level = self._cluster_headings_dbscan()
        root = HierarchicalHeader()
        current = root

        def gt_stylistic(
            level_fontsize: int,
            style_attrs: list[StyleAttributes],
            ref: HierarchicalHeader,
        ) -> bool:
            if ref.level_fontsize is None:
                return True
            if level_fontsize > ref.level_fontsize:
                return True
            if level_fontsize < ref.level_fontsize:
                return False
            return len(style_attrs) > len(ref.style_attrs)

        def eq_stylistic(
            level_fontsize: int,
            style_attrs: list[StyleAttributes],
            ref: HierarchicalHeader,
        ) -> bool:
            if ref.level_fontsize is None:
                return False
            return level_fontsize == ref.level_fontsize and len(style_attrs) == len(
                ref.style_attrs
            )

        for i, this_style in enumerate(self.style_features):
            new_parent = None
            this_fs_level = heading_to_level[i]
            this_style_attr = [
                k
                for k in [StyleAttributes.bold, StyleAttributes.italic]
                if this_style[k]
            ]
            if gt_stylistic(this_fs_level, this_style_attr, current):
                # print(f"gt: {this_fs_level, this_style_attr} VS: {current.level_fontsize, current.style_attrs}")
                new_parent = current
            elif eq_stylistic(this_fs_level, this_style_attr, current):
                # print(f"eq: {this_fs_level, this_style_attr} VS: {current.level_fontsize, current.style_attrs}")
                if current.parent is not None:
                    new_parent = current.parent
                else:
                    raise ImplausibleHeadingStructureException()
            else:
                # go back up in hierarchy and try to find a new parent
                new_parent = current
                while new_parent.parent is not None and (
                    not gt_stylistic(this_fs_level, this_style_attr, new_parent)
                    or eq_stylistic(this_fs_level, this_style_attr, new_parent)
                ):
                    new_parent = new_parent.parent
                # print(f"fit parent for : {this_fs_level, this_style_attr} parent: {new_parent.level_fontsize, new_parent.style_attrs}")
            new_obj = HierarchicalHeader(
                index=i,
                text=self.headings[i]["text"],
                parent=new_parent,
                level_fontsize=this_fs_level,
                style_attrs=this_style_attr,
                doc_ref=self.headings[i].get("reference"),
            )
            new_parent.children.append(new_obj)
            current = new_obj

        return root

    def _cluster_headings_dbscan(self) -> dict[int, int]:
        style_features = self.style_features

        if len(self.headings) < 2:
            return dict.fromkeys(range(len(self.headings)), 1)

        style_features_numpy = np.array(
            [[el[StyleAttributes.font_size]] for el in style_features]
        )

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(style_features_numpy)

        min_samples_grid = list(range(1, min(len(self.headings), 4)))
        eps_grid = np.arange(0.05, 0.21, 0.01)

        best_score = None
        best_params = (eps_grid[0], min_samples_grid[0])
        for eps in eps_grid:
            for min_samples in min_samples_grid:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features_scaled)
                if len(set(labels)) > 1 and len(set(labels)) != len(
                    labels
                ):  # At least two clusters required for silhouette
                    score = silhouette_score(features_scaled, labels)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
        # print("best parameters", best_params)
        dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1])
        cluster_labels = dbscan.fit_predict(features_scaled)

        # Map clusters to hierarchy levels based on average font size
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_headings = [
                self.headings[i] for i in range(len(self.headings)) if mask[i]
            ]
            avg_font_size = np.mean([h["font_size"] for h in cluster_headings])
            # print("cluster mean: ", np.mean([h['font_size'] for h in cluster_headings]), "std:", np.std([h['font_size'] for h in cluster_headings]))
            cluster_stats[cluster_id] = avg_font_size

        # Sort clusters by font size (largest = level 1)
        sorted_clusters = sorted(
            cluster_stats.items(), key=lambda x: x[1], reverse=True
        )
        cluster_to_level = {
            cluster_id: level + 1
            for level, (cluster_id, _) in enumerate(sorted_clusters)
        }

        # Create mapping from heading index to hierarchy level
        heading_to_level = {}
        for i, cluster_id in enumerate(cluster_labels):
            heading_to_level[i] = cluster_to_level[cluster_id]

        return heading_to_level

    def infer(self) -> HierarchicalHeader:
        if len(self.headings) == 0:
            return HierarchicalHeader()
        try:
            if (
                self.infer_hierarchy_from_numbering
                and (root_node := self._infer_from_numbering()) is not None
            ):
                return root_node
        except InconsistentNumberingException as e:
            print(f"WARNING: Inferring heading hierarchy from numbering failed. {e}")
            print("falling back to cluster-based")
        if self.infer_hierarchy_from_style:
            root_node = self._infer_by_clustering()
            return root_node
        return HierarchicalHeader()


def cleanup_non_headings(headings: list[dict]) -> list[dict]:
    counter = Counter([el["text"] for el in headings])
    duplicate_headings = [k for k, count in counter.items() if count > 1]
    # if the location is always the same then it's either a header or a footer
    location = [el for el in headings if el["text"] in duplicate_headings]
    # otherwise they might still be sub-headings...

    main_text_elements = [
        el
        for el in headings
        if el["text"].startswith("•") or el["text"].startswith("o")
    ]

    ignore_texts = [el["text"] for el in location + main_text_elements]
    headings = [el for el in headings if el["text"] not in ignore_texts]

    return headings


class StyleBasedHierarchy:
    def __init__(
        self,
        conv_res: ConversionResult,
        sorted_elements: list[ReadingOrderPageElement],
        raise_on_error: bool = False,
        remove_duplicate_headers: bool = True,
        infer_hierarchy_from_style: bool = True,
        infer_hierarchy_from_numbering: bool = True,
        min_prop_numbered: float = 0.3,
    ):
        self.result: ConversionResult = conv_res
        self.raise_on_error: bool = raise_on_error
        self.remove_duplicate_headers: bool = remove_duplicate_headers
        self.infer_hierarchy_from_style: bool = infer_hierarchy_from_style
        self.infer_hierarchy_from_numbering: bool = infer_hierarchy_from_numbering
        self.min_prop_numbered: float = min_prop_numbered

        self.all_elements: list[PageElement] = conv_res.assembled.elements
        self.all_cids: list[str] = [
            f"#/{element.page_no}/{element.cluster.id}" for element in self.all_elements
        ]
        self.sorted_ro_elements: list[ReadingOrderPageElement] = sorted_elements
        self.cid_to_page_element: dict[str, PageElement] = dict(
            zip(self.all_cids, self.all_elements)
        )

    def _get_headers_sorted(self):
        headers = []
        # identify the text item in the document
        for _i, ro_element in enumerate(self.sorted_ro_elements):
            element = self.cid_to_page_element[ro_element.ref.cref]
            if (
                not isinstance(element, TextElement)
                or element.label != DocItemLabel.SECTION_HEADER
            ):
                continue
            cluster = element.cluster
            first_cell = cluster.cells[0]
            font_split = (
                first_cell.font_name.split("-")
                if hasattr(first_cell, "font_name")
                else [""]
            )
            headers.append(
                {
                    "text": " ".join([cell.text for cell in cluster.cells]),
                    "font_size": first_cell.rect.height,
                    "is_bold": "Bold" in font_split[1]
                    if len(font_split) > 1
                    else False,
                    "is_italic": "Italic" in font_split[1]
                    if len(font_split) > 1
                    else False,
                    "top_left": first_cell.rect.r_y0,
                    "text_direction:": first_cell.text_direction,
                    "font": font_split[0],
                    "reference": ro_element.ref.cref,
                }
            )
        return headers

    def get_headers(self) -> list[dict]:
        if not (items := self._get_headers_sorted()):
            return []
        return items

    def process(self) -> HierarchicalHeader:
        headings = self.get_headers()
        if self.remove_duplicate_headers:
            headings = cleanup_non_headings(headings)
        builder = DocumentHierarchyBuilder(
            headings,
            min_prop_numbered=self.min_prop_numbered,
            infer_hierarchy_from_style=self.infer_hierarchy_from_style,
            infer_hierarchy_from_numbering=self.infer_hierarchy_from_numbering,
        )
        return builder.infer()
