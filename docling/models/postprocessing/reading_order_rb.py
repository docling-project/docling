# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import copy
import logging
import math
import re
from dataclasses import dataclass, field
from itertools import islice, takewhile
from typing import Dict, Iterable, List, Set, Tuple

from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel
from rtree import index as rtree_index

_log = logging.getLogger(__name__)


class PageElement(BoundingBox):
    eps: float = 1.0e-3

    cid: int
    ref: RefItem = RefItem(cref="#")  # type: ignore

    text: str = ""

    page_no: int
    page_size: Size

    label: DocItemLabel

    def __str__(self):
        return f"{self.cid:6.2f}\t{self.label!s:<10}\t{self.l:6.2f}, {self.b:6.2f}, {self.r:6.2f}, {self.t:6.2f}"

    def __lt__(self, other):
        if self.page_no == other.page_no:
            if self.overlaps_horizontally(other):
                return self.b > other.b
            else:
                return self.l < other.l
        else:
            return self.page_no < other.page_no

    def follows_maintext_order(self, rhs) -> bool:
        return self.cid + 1 == rhs.cid


@dataclass
class _ReadingOrderPredictorState:
    """
    State container of the reading order of a single page
    """

    h2i_map: Dict[int, int] = field(default_factory=dict)
    i2h_map: Dict[int, int] = field(default_factory=dict)
    l2r_map: Dict[int, int] = field(default_factory=dict)
    r2l_map: Dict[int, int] = field(default_factory=dict)
    up_map: Dict[int, List[int]] = field(default_factory=dict)
    dn_map: Dict[int, List[int]] = field(default_factory=dict)
    heads: List[int] = field(default_factory=list)


GRAPHIC_LABELS = {DocItemLabel.TABLE, DocItemLabel.PICTURE, DocItemLabel.CODE}


def _is_graphic(element: PageElement) -> bool:
    return element.label in GRAPHIC_LABELS


def _graphic_run(elements: Iterable[PageElement]) -> List[PageElement]:
    """The unbroken run of graphics `elements` opens with."""
    return list(takewhile(_is_graphic, elements))


def _shortest_box_gap(lhs: PageElement, rhs: PageElement) -> float:
    """
    Shortest distance between two boxes, 0 once they touch or overlap.

    Along either axis the boxes span their union, so whatever the union has
    left over once both are laid down is the gap between them. The union
    helpers keep that right for either coordinate origin.
    """
    dx = max(0.0, lhs.x_union_with(rhs) - lhs.width - rhs.width)
    dy = max(0.0, lhs.y_union_with(rhs) - lhs.height - rhs.height)
    return math.hypot(dx, dy)


class ReadingOrderPredictor:
    r"""
    Rule based reading order for DoclingDocument
    """

    def __init__(self):
        self.dilated_page_element = True

        # Apply horizontal dilation only if it is less than this page-width normalized threshold
        self._horizontal_dilation_threshold_norm = 0.15

    def predict_reading_order(
        self, page_elements: List[PageElement]
    ) -> List[PageElement]:

        page_nos: Set[int] = set()

        for i, elem in enumerate(page_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        page_to_headers: Dict[int, List[PageElement]] = {}
        page_to_footers: Dict[int, List[PageElement]] = {}

        for page_no in page_nos:
            page_to_elems[page_no] = []
            page_to_footers[page_no] = []
            page_to_headers[page_no] = []

        for i, elem in enumerate(page_elements):
            if elem.label == DocItemLabel.PAGE_HEADER:
                page_to_headers[elem.page_no].append(elem)
            elif elem.label == DocItemLabel.PAGE_FOOTER:
                page_to_footers[elem.page_no].append(elem)
            else:
                page_to_elems[elem.page_no].append(elem)

        # print("headers ....")
        for page_no, elems in page_to_headers.items():
            page_to_headers[page_no] = self._predict_page(elems)

        # print("elems ....")
        for page_no, elems in page_to_elems.items():
            page_to_elems[page_no] = self._predict_page(elems)

        # print("footers ....")
        for page_no, elems in page_to_footers.items():
            page_to_footers[page_no] = self._predict_page(elems)

        sorted_elements = []
        for page_no in sorted(page_nos):
            sorted_elements.extend(page_to_headers[page_no])
            sorted_elements.extend(page_to_elems[page_no])
            sorted_elements.extend(page_to_footers[page_no])

        return sorted_elements

    def predict_to_captions(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_captions: Dict[int, List[int]] = {}

        page_nos: Set[int] = set()
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):
            page_to_elems[elem.page_no].append(elem)

        for page_no, elems in page_to_elems.items():
            page_to_captions = self._find_to_captions(
                page_elements=page_to_elems[page_no]
            )
            for key, val in page_to_captions.items():
                to_captions[key] = val

        return to_captions

    def predict_to_footnotes(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_footnotes: Dict[int, List[int]] = {}

        page_nos: Set[int] = set()
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        page_to_elems: Dict[int, List[PageElement]] = {}
        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):
            page_to_elems[elem.page_no].append(elem)

        for page_no, elems in page_to_elems.items():
            page_to_footnotes = self._find_to_footnotes(
                page_elements=page_to_elems[page_no]
            )
            for key, val in page_to_footnotes.items():
                to_footnotes[key] = val

        return to_footnotes

    def predict_merges(
        self, sorted_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        merges: Dict[int, List[int]] = {}

        skip_labels = [
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.TABLE,
            DocItemLabel.PICTURE,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
        ]

        curr_ind = -1
        for ind, elem in enumerate(sorted_elements):
            if ind <= curr_ind:
                continue

            if elem.label in [DocItemLabel.TEXT]:
                merge_list: List[int] = []
                check_ind = ind

                while True:
                    ind_p1 = check_ind + 1
                    while (
                        ind_p1 < len(sorted_elements)
                        and sorted_elements[ind_p1].label in skip_labels
                    ):
                        ind_p1 += 1

                    if (
                        ind_p1 < len(sorted_elements)
                        and sorted_elements[ind_p1].label == elem.label
                        and (
                            elem.page_no != sorted_elements[ind_p1].page_no
                            or elem.is_strictly_left_of(sorted_elements[ind_p1])
                            or (
                                elem.is_strictly_above(sorted_elements[ind_p1])
                                and elem.overlaps_horizontally(sorted_elements[ind_p1])
                                and _shortest_box_gap(elem, sorted_elements[ind_p1])
                                <= max(elem.height, sorted_elements[ind_p1].height)
                            )
                        )
                    ):
                        m1 = re.fullmatch(
                            r".+([a-z,\-\u00AD])(\s*)", sorted_elements[check_ind].text
                        )
                        m2 = re.fullmatch(
                            r"(\s*[a-zA-Z\u00C0-\u024F])(.+)",
                            sorted_elements[ind_p1].text,
                        )

                        if m1 and m2:
                            merge_list.append(sorted_elements[ind_p1].cid)
                            curr_ind = ind_p1
                            check_ind = ind_p1
                        else:
                            break
                    else:
                        break

                if merge_list:
                    merges[elem.cid] = merge_list

        return merges

    def _predict_page(self, page_elements: List[PageElement]) -> List[PageElement]:
        r"""
        Reorder the output of the page elements into a single-page reading order.
        """

        state = _ReadingOrderPredictorState()

        """
        for i, elem in enumerate(page_elements):
            print(f"{i:6.2f}\t{str(elem)}")
        """

        for i, elem in enumerate(page_elements):
            page_elements[i] = elem.to_bottom_left_origin(  # type: ignore
                page_height=page_elements[i].page_size.height
            )
        self._init_h2i_map(page_elements, state)

        self._init_l2r_map(page_elements, state)

        self._init_ud_maps(page_elements, state)

        if self.dilated_page_element:
            dilated_page_elements: List[PageElement] = copy.deepcopy(
                page_elements
            )  # deep-copy

            dilated_page_elements = self._do_horizontal_dilation(
                page_elements, dilated_page_elements, state
            )

            # redo with dilated provs
            self._init_ud_maps(dilated_page_elements, state)

        self._find_heads(page_elements, state)

        self._sort_ud_maps(page_elements, state)

        """
        print(f"heads: {state.heads}")

        print("l2r: ")
        for k,v in state.l2r_map.items():
            print(f" -> {k}: {v}")

        print("r2l: ")
        for k,v in state.r2l_map.items():
            print(f" -> {k}: {v}")

        print("up: ")
        for k,v in state.up_map.items():
            print(f" -> {k}: {v}")

        print("dn: ")
        for k,v in state.dn_map.items():
            print(f" -> {k}: {v}")
        """

        order: List[int] = self._find_order(page_elements, state)
        # print(f"order: {order}")

        sorted_elements: List[PageElement] = []
        for ind in order:
            sorted_elements.append(page_elements[ind])

        """
        for i, elem in enumerate(sorted_elements):
            print(f"{i:6.2f}\t{str(elem)}")
        """

        return sorted_elements

    def _init_h2i_map(
        self, page_elems: List[PageElement], state: _ReadingOrderPredictorState
    ) -> None:
        state.h2i_map = {}
        state.i2h_map = {}

        for i, pelem in enumerate(page_elems):
            state.h2i_map[pelem.cid] = i
            state.i2h_map[i] = pelem.cid

    def _init_l2r_map(
        self, page_elems: List[PageElement], state: _ReadingOrderPredictorState
    ) -> None:
        state.l2r_map = {}
        state.r2l_map = {}

        for i, pelem_i in enumerate(page_elems):
            for j, pelem_j in enumerate(page_elems):
                if (
                    pelem_i.follows_maintext_order(pelem_j)
                    and pelem_i.is_strictly_left_of(pelem_j)
                    and pelem_i.overlaps_vertically_with_iou(pelem_j, 0.8)
                ):
                    state.l2r_map[i] = j
                    state.r2l_map[j] = i

    def _init_ud_maps(
        self, page_elems: List[PageElement], state: _ReadingOrderPredictorState
    ) -> None:
        """
        Initialize up/down maps for reading order prediction using R-tree spatial indexing.

        Uses R-tree for spatial queries.
        Determines linear reading sequence by finding preceding/following elements.
        """
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
                left_partner = state.r2l_map[j]
                # Link the same-row left partner, then keep searching for vertical parents
                if j not in state.dn_map[left_partner]:
                    state.dn_map[left_partner].append(j)
                if left_partner not in state.up_map[j]:
                    state.up_map[j].append(left_partner)
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
                    while i in state.l2r_map:
                        i = state.l2r_map[i]

                    state.dn_map[i].append(j)
                    state.up_map[j].append(i)

    def _has_sequence_interruption(
        self,
        spatial_idx: rtree_index.Index,
        page_elems: List[PageElement],
        i: int,
        j: int,
        pelem_i: PageElement,
        pelem_j: PageElement,
    ) -> bool:
        """Check if elements interrupt the reading sequence between i and j."""
        # Query R-tree for elements between i and j
        x_min = min(pelem_i.l, pelem_j.l) - 1.0
        x_max = max(pelem_i.r, pelem_j.r) + 1.0
        y_min = pelem_j.t
        y_max = pelem_i.b

        # pelem_i is only guaranteed to sit above pelem_j within is_strictly_above's
        # epsilon, so pelem_i.b can slightly exceed pelem_j.t and leave y_min > y_max.
        # Keep the query rectangle well-formed (min <= max on every axis); otherwise
        # rtree raises "Coordinates must not have minimums more than maximums".
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)

        candidates = list(spatial_idx.intersection((x_min, y_min, x_max, y_max)))

        for w in candidates:
            if w in (i, j):
                continue

            pelem_w = page_elems[w]

            # Check if w interrupts the i->j sequence
            if (
                (
                    pelem_i.overlaps_horizontally(pelem_w)
                    or pelem_j.overlaps_horizontally(pelem_w)
                )
                and pelem_i.is_strictly_above(pelem_w)
                and pelem_w.is_strictly_above(pelem_j)
            ):
                return True

        return False

    def _do_horizontal_dilation(
        self,
        page_elems: List[PageElement],
        dilated_page_elems: List[PageElement],
        state: _ReadingOrderPredictorState,
    ) -> List[PageElement]:
        # Compute the dilation threshold
        th = 0.0
        if page_elems:
            page_size = page_elems[0].page_size
            th = self._horizontal_dilation_threshold_norm * page_size.width

        for i, pelem_i in enumerate(dilated_page_elems):
            x0 = pelem_i.l
            y0 = pelem_i.b

            x1 = pelem_i.r
            y1 = pelem_i.t

            if i in state.up_map and len(state.up_map[i]) > 0:
                pelem_up = page_elems[state.up_map[i][0]]

                # Apply threshold for horizontal dilation
                x0_dil = min(x0, pelem_up.l)
                x1_dil = max(x1, pelem_up.r)
                if (x0 - x0_dil) > th or (x1_dil - x1) > th:
                    continue
                x0 = x0_dil
                x1 = x1_dil

            if i in state.dn_map and len(state.dn_map[i]) > 0:
                pelem_dn = page_elems[state.dn_map[i][0]]

                # Apply threshold for horizontal dilation
                x0_dil = min(x0, pelem_dn.l)
                x1_dil = max(x1, pelem_dn.r)
                if (x0 - x0_dil) > th or (x1_dil - x1) > th:
                    continue
                x0 = x0_dil
                x1 = x1_dil

            pelem_i.l = x0
            pelem_i.r = x1

            overlaps_with_rest: bool = False
            for j, pelem_j in enumerate(page_elems):
                if i == j:
                    continue

                if not overlaps_with_rest:
                    overlaps_with_rest = pelem_j.overlaps(pelem_i)

            # update
            if not overlaps_with_rest:
                dilated_page_elems[i].l = x0
                dilated_page_elems[i].b = y0
                dilated_page_elems[i].r = x1
                dilated_page_elems[i].t = y1

        return dilated_page_elems

    def _find_heads(
        self, page_elems: List[PageElement], state: _ReadingOrderPredictorState
    ) -> None:
        head_page_elems = []
        for key, vals in state.up_map.items():
            if len(vals) == 0:
                head_page_elems.append(page_elems[key])

        """
        print("before sorting the heads: ")
        for l, elem in enumerate(head_page_elems):
            print(f"{l}\t{str(elem)}")
        """

        # this will invoke __lt__ from PageElements
        head_page_elems = sorted(head_page_elems)

        """
        print("after sorting the heads: ")
        for l, elem in enumerate(head_page_elems):
            print(f"{l}\t{str(elem)}")
        """

        state.heads = []
        for item in head_page_elems:
            state.heads.append(state.h2i_map[item.cid])

    def _sort_ud_maps(
        self, provs: List[PageElement], state: _ReadingOrderPredictorState
    ) -> None:
        for ind_i, vals in state.dn_map.items():
            child_provs: List[PageElement] = []
            for ind_j in vals:
                child_provs.append(provs[ind_j])

            # this will invoke __lt__ from PageElements
            child_provs = sorted(child_provs)

            state.dn_map[ind_i] = []
            for child in child_provs:
                state.dn_map[ind_i].append(state.h2i_map[child.cid])

    def _find_order(
        self, provs: List[PageElement], state: _ReadingOrderPredictorState
    ) -> List[int]:
        order: List[int] = []

        visited: List[bool] = [False for _ in provs]

        for j in state.heads:
            if not visited[j]:
                order.append(j)
                visited[j] = True
                self._depth_first_search_downwards(j, order, visited, state)

        if len(order) != len(provs):
            _log.error("something went wrong")

        return order

    def _depth_first_search_upwards(
        self, j: int, visited: List[bool], state: _ReadingOrderPredictorState
    ) -> int:
        """depth_first_search_upwards without recursion"""
        k = j
        while True:
            inds: List[int] = state.up_map[k]
            found_not_visited = False
            for ind in inds:
                if not visited[ind]:
                    k = ind
                    found_not_visited = True
                    break

            # If a not-visited is found repeat the while loop
            if not found_not_visited:
                return k

    def _depth_first_search_downwards(
        self,
        j: int,
        order: List[int],
        visited: List[bool],
        state: _ReadingOrderPredictorState,
    ) -> None:
        """depth_first_search_downwards without recursion"""
        # The outermost list is the main stack.
        # Each list element is a tuple containint the list of the indices to be checked and an offset
        stack: List[Tuple[List[int], int]] = [(state.dn_map[j], 0)]

        while stack:
            inds, offset = stack[-1]

            found_non_visited = False
            if offset < len(inds):
                for new_offset, i in enumerate(inds[offset:]):
                    k: int = self._depth_first_search_upwards(i, visited, state)

                    if not visited[k]:
                        order.append(k)
                        visited[k] = True
                        stack[-1] = (inds, new_offset + 1)
                        stack.append((state.dn_map[k], 0))
                        found_non_visited = True
                        break

            if not found_non_visited:
                stack.pop()

    @staticmethod
    def _rank_caption_candidates(
        page_elements: List[PageElement],
    ) -> Dict[int, List[int]]:
        """
        Map each caption cid to the cids of the graphics it could belong to,
        nearest first.

        A caption only reaches the graphics in an unbroken run on either side
        of it, ranked by the gap each leaves on the page rather than by
        position in the run: the nearer graphic is not always the one above.
        Equal gaps go to the preceding graphic.
        """
        # Reversed once for the whole page, so that each caption can scan back
        # from its own index without a slice of its own.
        backwards = page_elements[::-1]
        size = len(page_elements)

        preferred: Dict[int, List[int]] = {}
        for ind, caption in enumerate(page_elements):
            if caption.label != DocItemLabel.CAPTION:
                continue
            preceding = _graphic_run(islice(backwards, size - ind, None))
            following = _graphic_run(islice(page_elements, ind + 1, None))
            # The candidates, in run order, tagged 0 preceding / 1 following so
            # that equal gaps go to the graphic above.
            ranked = {
                graphic.cid: (_shortest_box_gap(caption, graphic), side)
                for side, run in ((0, preceding), (1, following))
                for graphic in run
            }
            preferred[caption.cid] = sorted(ranked, key=lambda cid: ranked[cid])
        return preferred

    @staticmethod
    def _match_caption(
        caption_cid: int,
        preferred: Dict[int, List[int]],
        matched: Dict[int, int],
        seen: Set[int],
    ) -> Dict[int, int]:
        """
        Given the current matching `matched` (graphic cid -> caption cid),
        return a new matching in which the caption owns one of its graphics,
        displacing an earlier caption when that one can rehouse itself. Empty
        if no graphic can be freed up.

        The displacement chain is walked on an explicit stack, since it can
        grow as long as the page. Each frame is a caption and the graphics it
        has left to try, and the moves in `chain` take effect only once the
        chain reaches a graphic nobody holds. `seen` keeps it from revisiting
        a graphic; pass a fresh set per caption.
        """
        stack = [(caption_cid, iter(preferred[caption_cid]))]
        chain: List[Tuple[int, int]] = []
        while stack:
            claimant, candidates = stack[-1]
            graphic_cid = next((cid for cid in candidates if cid not in seen), None)
            if graphic_cid is None:
                # Out of options: undo the move that got here, and let the
                # caption below resume its own search.
                stack.pop()
                if chain:
                    chain.pop()
                continue
            seen.add(graphic_cid)
            chain.append((graphic_cid, claimant))
            held_by = matched.get(graphic_cid)
            if held_by is None:
                return matched | dict(chain)
            # The graphic is taken; let its caption look for another one.
            stack.append((held_by, iter(preferred[held_by])))
        return {}

    def _find_to_captions(
        self, page_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        # page_elements arrives in reading order, which already places each caption
        # next to its graphic; cids are parse order and would scatter them.
        preferred = self._rank_caption_candidates(page_elements)

        # Match by augmenting paths, not best-first: a caption with a second
        # choice must give way to one that has none, or both end up orphaned.
        matched: Dict[int, int] = {}
        for cid in preferred:
            matched = self._match_caption(cid, preferred, matched, set()) or matched

        return {graphic: [caption] for graphic, caption in matched.items()}

    def _find_to_footnotes(
        self, page_elements: List[PageElement]
    ) -> Dict[int, List[int]]:

        to_footnotes: Dict[int, List[int]] = {}

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):
            if page_element.label in [DocItemLabel.TABLE, DocItemLabel.PICTURE]:
                ind_p1 = ind + 1
                while (
                    ind_p1 < len(page_elements)
                    and page_elements[ind_p1].label == DocItemLabel.FOOTNOTE
                ):
                    if page_element.cid in to_footnotes:
                        to_footnotes[page_element.cid].append(page_elements[ind_p1].cid)
                    else:
                        to_footnotes[page_element.cid] = [page_elements[ind_p1].cid]

                    ind_p1 += 1

        return to_footnotes
