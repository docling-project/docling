# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import copy
import logging
import os
import random
import sys
import time
from typing import Dict, List

import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

from docling.models.postprocessing.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
    SeparatorElement,
    build_page_separators,
)

IS_CI = bool(os.getenv("CI"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def rank_array(arr):
    """Compute ranks, resolving ties by averaging."""
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])  # Sort indices
    ranks = [0] * len(arr)  # Initialize ranks

    i = 0
    while i < len(arr):
        start = i
        while i + 1 < len(arr) and arr[sorted_indices[i]] == arr[sorted_indices[i + 1]]:
            i += 1  # Handle ties
        avg_rank = sum(range(start + 1, i + 2)) / (
            i - start + 1
        )  # Average rank for ties
        for j in range(start, i + 1):
            ranks[sorted_indices[j]] = avg_rank
        i += 1
    return ranks


def spearman_rank_correlation(arr1, arr2):
    assert len(arr1) == len(arr2), "Arrays must have the same length"

    # Compute ranks
    rank1 = rank_array(arr1)
    rank2 = rank_array(arr2)

    # Compute rank differences and apply formula
    d = [rank1[i] - rank2[i] for i in range(len(arr1))]
    d_squared_sum = sum(d_i**2 for d_i in d)

    n = len(arr1)
    if n > 1:
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    else:
        rho = 0
    return rho


@pytest.mark.skipif(IS_CI, reason="dpbench dataset too heavy for CI")
def test_readingorder():
    if sys.version_info >= (3, 14):
        pytest.skip(
            "Pyarrow is not yet available for Python 3.14, hence we cannot load the dataset."
        )

    datasets = pytest.importorskip("datasets")

    ro_scores, caption_scores, footnote_scores = [], [], []

    # Init the reading-order model
    romodel = ReadingOrderPredictor()

    ds = datasets.load_dataset("ds4sd/docling-dpbench")
    for row in ds["test"]:
        true_doc = DoclingDocument.model_validate_json(row["GroundTruthDocument"])

        true_elements: List[PageElement] = []
        pred_elements: List[PageElement] = []

        to_ref: Dict[int, str] = {}
        from_ref: Dict[str, int] = {}

        for item, level in true_doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            if isinstance(item, DocItem):
                for prov in item.prov:
                    page_height = true_doc.pages[prov.page_no].size.height
                    bbox = prov.bbox.to_bottom_left_origin(page_height=page_height)

                    text = ""
                    if isinstance(item, TextItem):
                        text = item.text

                    true_elements.append(
                        PageElement(
                            cid=len(true_elements),
                            ref=item.get_ref(),
                            text=text,
                            page_no=prov.page_no,
                            page_size=true_doc.pages[prov.page_no].size,
                            label=item.label,
                            l=bbox.l,
                            r=bbox.r,
                            b=bbox.b,
                            t=bbox.t,
                            coord_origin=bbox.coord_origin,
                        )
                    )

                    to_ref[true_elements[-1].cid] = item.get_ref().cref
                    from_ref[item.get_ref().cref] = true_elements[-1].cid

        rand_elements = copy.deepcopy(true_elements)
        # Seeded, so that a score close to its threshold cannot flake from one
        # run to the next on a different shuffle.
        random.Random(0).shuffle(rand_elements)

        """
        print(f"reading {os.path.basename(filename)}")
        for true_elem, rand_elem in zip(true_elements, rand_elements):
            print("true: ", str(true_elem), ", rand: ", str(rand_elem))
        """

        pred_elements = romodel.predict_reading_order(page_elements=rand_elements)
        # pred_elements = romodel.predict_page(page_elements=rand_elements)

        assert len(pred_elements) == len(true_elements), (
            f"{len(pred_elements)}!={len(true_elements)}"
        )

        true_cids, pred_cids = [], []
        for true_elem, pred_elem, rand_elem in zip(
            true_elements, pred_elements, rand_elements
        ):
            true_cids.append(true_elem.cid)
            pred_cids.append(pred_elem.cid)

        score = spearman_rank_correlation(true_cids, pred_cids)
        ro_scores.append(score)

        filename = row["document_id"]

        if score == 0:
            continue
        # Identify special cases ...
        if (
            filename
            in [
                "doc_906d54a21ef3c7bfac03f4bb613b0c79ef32fdf81b362450c79e98a96f88708a_page_000001.png",  # 0.720588
                "doc_2cd17a32ee330a239e19c915738df0c27e8ec3635a60a7e16e2a0cf3868d4af3_page_000001.png",  # 0.64920
                "doc_bcb3dafc35b5e7476fd1b9cd6eccf5eeef936cd5b13ad846a4943f1e7797f4e9_page_000001.png",  # 0.65
                "doc_a0edae1fa147c7bb78ebc493743a68ba4372b5ead31f2a2b146c35119462379e_page_000001.png",  # 0.82857
                "doc_94ba5468fcb6277721947697048846dc0d0551296be3b45f5918ab857d21dcc7_page_000001.png",  # 0.857142
                #  "doc_cbb4a13ffd01d9f777fdb939451d6a21cea1b869ee50d79581451e3601df9ec8_page_000001.png",
                "doc_e2b604a3fb1541b82b6af8caca05682dff0c7735e0a3a4fa7c6a68246fb60e57_page_000001.png",  # 0.657142
                "doc_827d21de372a2c26237ee1db526460851ae71c1867761776583535f532432e32_page_000001.png",  # 0.8922077
                "doc_b862cd0d6f06c06ee5ab7729ed4e8ce58e6964eb0f1ab98b3865b57a4808216f_page_000001.png",
            ]
        ):  # 0.642857
            # print(f"{os.path.basename(filename)}: {score}")
            assert score >= 0.60, f"reading-order score={score}>0.60"
        else:
            assert score >= 0.90, f"reading-order score={score}>0.90 for {filename}"

        true_to_captions: Dict[int, List[int]] = {}
        true_to_footnotes: Dict[int, List[int]] = {}

        total_caption_links = 0
        total_footnote_links = 0

        for table in true_doc.tables:
            table_cid = from_ref[table.get_ref().cref]

            true_to_captions[table_cid] = []
            for caption in table.captions:
                caption_cid = from_ref[caption.get_ref().cref]
                true_to_captions[table_cid].append(caption_cid)

                total_caption_links += 1

            true_to_footnotes[table_cid] = []
            for footnote in table.footnotes:
                footnote_cid = from_ref[footnote.get_ref().cref]
                true_to_footnotes[table_cid].append(footnote_cid)

                total_footnote_links += 1

        for picture in true_doc.pictures:
            picture_cid = from_ref[picture.get_ref().cref]

            true_to_captions[picture_cid] = []
            for caption in picture.captions:
                caption_cid = from_ref[caption.get_ref().cref]
                true_to_captions[picture_cid].append(caption_cid)

                total_caption_links += 1

            true_to_footnotes[picture_cid] = []
            for footnote in picture.footnotes:
                footnote_cid = from_ref[footnote.get_ref().cref]
                true_to_footnotes[picture_cid].append(footnote_cid)

                total_footnote_links += 1

        if total_caption_links > 0:
            # print(" *********** ")

            pred_to_captions = romodel.predict_to_captions(
                sorted_elements=pred_elements
            )

            """
            for key,val in pred_to_captions.items():
                print(f"pred {key}: {val}")
            """

            score, total = 0.0, 0.0
            for key, val in true_to_captions.items():
                # print(f"true {key}: {val}")

                total += 1.0
                if pred_to_captions.get(key, []) == val:
                    score += 1.0

            # print(f"to_captions: {score/total}")
            caption_scores.append(score / total)

        if total_footnote_links > 0:
            # print(" *********** ")

            pred_to_footnotes = romodel.predict_to_footnotes(
                sorted_elements=pred_elements
            )

            """
            for key,val in pred_to_footnotes.items():
                print(f"pred {key}: {val}")
            """

            score, total = 0.0, 0.0
            for key, val in true_to_footnotes.items():
                # print(f"true {key}: {val}")

                total += 1.0
                if pred_to_footnotes.get(key, []) == val:
                    score += 1.0

            # print(f"to_footnotes: {score/total}")
            footnote_scores.append(score / total)

        romodel.predict_merges(sorted_elements=pred_elements)
        # print("merges: ", pred_merges)

    mean_ro_score = sum(ro_scores) / len(ro_scores)
    mean_cp_score = sum(caption_scores) / len(caption_scores)
    mean_ft_score = sum(footnote_scores) / len(footnote_scores)

    assert mean_ro_score > 0.95, "mean_ro_score>0.95"
    assert mean_cp_score > 0.85, "mean_cp_score>0.85"
    assert mean_ft_score > 0.90, "mean_ft_score>0.90"

    print("\n  score(reading): ", mean_ro_score)
    print("  score(caption): ", mean_cp_score)
    print("score(footnotes): ", mean_ft_score)


def test_caption_not_orphaned_in_two_column_figure():
    """A caption above the right column of a same-row pair is read in place, not last."""
    from docling_core.types.doc.base import CoordOrigin, Size
    from docling_core.types.doc.labels import DocItemLabel

    page_size = Size(width=600, height=800)

    def elem(cid, label, l, r, b, t, text=""):  # noqa: E741
        return PageElement(
            cid=cid,
            text=text,
            page_no=1,
            page_size=page_size,
            label=label,
            l=l,
            r=r,
            b=b,
            t=t,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    # picture (left) + caption (above the right column only); two same-row body columns below
    elements = [
        elem(0, DocItemLabel.PICTURE, 60, 270, 400, 690),
        elem(1, DocItemLabel.CAPTION, 340, 480, 300, 360, "Figure 1. Example"),
        elem(2, DocItemLabel.TEXT, 60, 270, 200, 290, "left column body"),
        elem(3, DocItemLabel.TEXT, 280, 494, 190, 290, "right column body text"),
    ]

    order = [
        e.cid
        for e in ReadingOrderPredictor().predict_reading_order(page_elements=elements)
    ]

    assert order.index(1) < order.index(3), (
        f"caption (cid 1) should be read before the body columns, got {order}"
    )


# ---------------------------------------------------------------------------
# Regression tests for caption <-> graphic pairing in
# ReadingOrderPredictor._find_to_captions.
#
# Pairing is label-agnostic: pictures and tables are treated the same way, and
# the graphic nearest to a caption wins, with ties going to the preceding
# graphic. Every caption that has an adjacent graphic gets paired, and each
# graphic and caption is used at most once.
#
# The shapes below come from the ground-truth of tests/data/pdf/2203.01017v2.pdf
# in the docling repo (its dense figure appendix, pp. 1/13/14). The pre-fix code
# mis-paired them: a nearer table could steal a picture's caption, and the first
# caption of a cluster could get dropped.
# ---------------------------------------------------------------------------

# A letter page, so that the coordinates below and any page-relative threshold
# mean the same thing they would in a real document.
_DUMMY_PAGE_SIZE = Size(width=612.0, height=792.0)


def _el(cid: int, label: DocItemLabel) -> PageElement:
    # Every element shares one box, so the pairing rests on the run alone.
    return PageElement(
        cid=cid,
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        label=label,
        l=0.0,
        r=10.0,
        t=10.0,
        b=0.0,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


def _graphic(cid: int) -> PageElement:
    return _el(cid, DocItemLabel.PICTURE)


def _table(cid: int) -> PageElement:
    return _el(cid, DocItemLabel.TABLE)


def _caption(cid: int) -> PageElement:
    return _el(cid, DocItemLabel.CAPTION)


def test_single_ambiguous_caption_is_assigned():
    # [G0, C1, G2]: the only caption has graphics on both sides. It must still be
    # paired with a graphic (tie broken towards the preceding graphic G0).
    elements = [_graphic(0), _caption(1), _graphic(2)]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1]}


def test_all_ambiguous_captions_are_assigned():
    # [G0, C1, G2, G3, C4, G5]: both captions are ambiguous; each must still be
    # paired with its nearest graphic (ties broken towards the preceding one).
    elements = [
        _graphic(0),
        _caption(1),
        _graphic(2),
        _graphic(3),
        _caption(4),
        _graphic(5),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1], 3: [4]}


def test_one_sided_caption_does_not_orphan_middle_caption():
    # [C0, G1, G2, C3, G4, C5]: C0 must take only its nearest graphic (G1), so
    # the middle caption C3 keeps G2 instead of being orphaned.
    elements = [
        _caption(0),
        _graphic(1),
        _graphic(2),
        _caption(3),
        _graphic(4),
        _caption(5),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {1: [0], 2: [3], 4: [5]}


def test_caption_above_each_graphic_pairs_both():
    # [C0, G1, C2, G3]: a caption above each graphic. Every candidate sits at
    # distance 1, so the preceding-side tie-break alone would hand G1 to C2 and
    # orphan both C0 (which has no other candidate) and G3. C2 must give way.
    elements = [_caption(0), _graphic(1), _caption(2), _graphic(3)]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {1: [0], 3: [2]}


def test_caption_binds_to_nearest_graphic():
    # [T0, P1, C2]: page 1 of 2203.01017v2 ("Figure 1: Picture of a table"). A
    # table precedes the picture that owns the caption. The caption binds to the
    # nearer picture P1. The pre-fix code gave it to the table (lower cid), which
    # left the picture without a caption.
    elements = [_table(0), _graphic(1), _caption(2)]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {1: [2]}


def test_dense_cluster_pairs_every_picture_with_its_caption():
    # Page 13 of 2203.01017v2: picture/caption pairs bunched with a table in the
    # run:  P0 C1 | P2 C3 | T4 | P5 C6  (Figures 8, 9, 10). Every picture keeps
    # its own caption and the table takes none. The ground truth for this PDF
    # still carries the old bug here, dropping the first caption (Fig 8); the
    # fixed behaviour pairs all three.
    elements = [
        _graphic(0),
        _caption(1),
        _graphic(2),
        _caption(3),
        _table(4),
        _graphic(5),
        _caption(6),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {0: [1], 2: [3], 5: [6]}


# ---------------------------------------------------------------------------
# Reproductions captured from a real conversion of 2203.01017v2.pdf. The lists
# below are the exact page_elements _find_to_captions receives (in reading
# order, with real cids/labels). In reading order the predictor already places
# each picture next to its caption, but cids are assigned in parse order, so a
# caption and its graphic can be far apart in cid order with unrelated elements
# (a table, a page number, a page_footer) sitting between them.
# ---------------------------------------------------------------------------

SH = DocItemLabel.SECTION_HEADER
TX = DocItemLabel.TEXT
PF = DocItemLabel.PAGE_FOOTER


def test_page1_caption_pairs_with_adjacent_picture_not_earlier_table():
    # 2203.01017v2 page 1: the picture (cid 8) sits right before its caption
    # (cid 12) in reading order, but in cid order two tables and a page-number
    # text land between them. The caption belongs to the picture (Figure 1).
    elements = [
        _el(7, SH),
        _table(11),
        _el(9, TX),
        _table(10),
        _graphic(8),
        _caption(12),
        _el(13, TX),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {8: [12]}


def test_page13_cluster_pairs_each_picture_with_its_caption():
    # 2203.01017v2 page 13: three figures, each picture immediately followed by
    # its caption in reading order (Fig 8/9/10), with a table and a page_footer
    # mixed into the run. In cid order all captions precede all pictures with a
    # page_footer between the two blocks, so cid-order pairing finds nothing.
    elements = [
        _el(205, TX),
        _el(206, TX),
        _el(207, TX),
        _graphic(213),
        _caption(208),  # Figure 8
        _graphic(212),
        _caption(209),  # Figure 9
        _table(215),
        _graphic(214),
        _caption(210),  # Figure 10
        _el(211, PF),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {213: [208], 212: [209], 214: [210]}


def _box(
    cid: int,
    label: DocItemLabel,
    b: float,
    t: float,
    l: float = 100.0,  # noqa: E741
    r: float = 500.0,
) -> PageElement:
    # Full-width by default, so only the vertical gap separates it from its
    # neighbours; pass l/r to place graphics side by side.
    return _el(cid, label).model_copy(update={"l": l, "r": r, "t": t, "b": b})


def test_caption_binds_below_when_the_graphic_below_is_nearer():
    # Doc 01030000000128, "Figure 13.3. Graph of Projection Estimates": a table
    # sits 49.7 above the caption, the figure it names 24.9 below. Ranking by
    # position in the run alone hands it to the table; it belongs to the picture.
    elements = [
        _box(0, DocItemLabel.TABLE, 427.2, 738.7),
        _box(1, DocItemLabel.CAPTION, 353.2, 377.5),
        _box(2, DocItemLabel.PICTURE, 161.5, 328.3),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {2: [1]}


def test_caption_between_two_pictures_binds_to_the_nearer_one():
    # Doc 01030000000131, "Figure 17.2. Year-to-year changes in housing prices":
    # 54.7 below the picture above, 47.9 above the picture below.
    elements = [
        _box(0, DocItemLabel.PICTURE, 520.3, 736.9),
        _box(1, DocItemLabel.CAPTION, 456.8, 465.6),
        _box(2, DocItemLabel.PICTURE, 197.9, 408.9),
    ]

    result = ReadingOrderPredictor()._find_to_captions(elements)

    assert result == {2: [1]}


"""
def test_readingorder_multipage():

    filename = Path("<json with page-elements>")

    # Init the reading-order model
    romodel = ReadingOrderPredictor()

    true_elements: List[PageElement] = []
    pred_elements: List[PageElement] = []

    with open(filename, "r") as fr:
        data = json.load(fr)
        true_elements = [PageElement.model_validate(item) for item in data]

    pred_elements = romodel.predict_reading_order(page_elements=true_elements)
    for true_elem, pred_elem in zip(true_elements, pred_elements):
        print("true: ", str(true_elem), ", pred: ", str(pred_elem))
"""


def test_reading_order_near_boundary_clusters(monkeypatch):
    """Regression for #3940: two clusters that almost share a horizontal
    boundary must not build a malformed rtree query rectangle.

    ``_has_sequence_interruption`` sets ``y_min = pelem_j.t`` and
    ``y_max = pelem_i.b``. The caller only guarantees ``pelem_i`` sits above
    ``pelem_j`` within ``is_strictly_above``'s epsilon (1e-3), so ``pelem_i.b``
    can slightly exceed ``pelem_j.t`` and leave ``y_min > y_max``. Depending on
    the installed rtree version that either raises
    ``RTreeError("Coordinates must not have minimums more than maximums")`` or
    is silently masked, so we enforce rtree's documented contract here to keep
    the regression deterministic across versions.
    """
    from docling_core.types.doc.base import CoordOrigin, Size
    from docling_core.types.doc.document import DocItemLabel
    from rtree import index as rtree_index

    from docling.models.postprocessing.reading_order_rb import (
        _ReadingOrderPredictorState,
    )

    _orig_intersection = rtree_index.Index.intersection

    def _strict_intersection(self, coordinates, *args, **kwargs):
        x_min, y_min, x_max, y_max = coordinates
        assert x_min <= x_max and y_min <= y_max, (
            f"malformed rtree query rectangle (min > max): {coordinates}"
        )
        return _orig_intersection(self, coordinates, *args, **kwargs)

    monkeypatch.setattr(rtree_index.Index, "intersection", _strict_intersection)

    def _mk(cid, l, b, r, t):  # noqa: E741
        return PageElement(
            cid=cid,
            text="x",
            page_no=1,
            page_size=Size(width=600, height=800),
            label=DocItemLabel.TEXT,
            l=l,
            r=r,
            b=b,
            t=t,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    # pelem_i.b (302.7814...) is "above" pelem_j.t (302.782...) only within eps.
    page_elems = [
        _mk(0, l=100.0, b=302.7814025878906, r=200.0, t=350.0),
        _mk(1, l=100.0, b=250.0, r=200.0, t=302.78204345703125),
    ]

    state = _ReadingOrderPredictorState()
    # Must not raise; before the fix this built y_min > y_max.
    ReadingOrderPredictor()._init_ud_maps(page_elems, state)


def test_horizontal_separator_finishes_upper_band_before_lower_columns() -> None:
    page_size = Size(width=600, height=800)

    def element(
        cid: int, left: float, bottom: float, right: float, top: float
    ) -> PageElement:
        return PageElement(
            cid=cid,
            text=str(cid),
            page_no=1,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=left,
            r=right,
            b=bottom,
            t=top,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    elements = [
        element(0, 40, 500, 250, 600),
        element(1, 330, 470, 560, 600),
        element(2, 40, 200, 250, 400),
        element(3, 330, 200, 560, 400),
    ]
    separator = SeparatorElement(
        cid=-1,
        page_no=1,
        page_size=page_size,
        orientation="horizontal",
        l=40,
        r=560,
        b=440,
        t=440,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    baseline = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements)
    )
    separated = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements), page_separators=[separator]
    )

    assert [element.cid for element in baseline] == [0, 2, 1, 3]
    assert [element.cid for element in separated] == [0, 1, 2, 3]
    assert all(isinstance(element, PageElement) for element in separated)


def test_horizontal_separator_finishes_columns_in_spatial_order() -> None:
    page_size = Size(width=600, height=800)

    def element(
        cid: int, left: float, bottom: float, right: float, top: float
    ) -> PageElement:
        return PageElement(
            cid=cid,
            text=str(cid),
            page_no=1,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=left,
            r=right,
            b=bottom,
            t=top,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    # Bottom-left origin: the three columns share the same vertical span.
    # Their spatial order (1, 3, 2) differs from their CID order (1, 2, 3).
    #
    #          x=40   180 220   360 400   540 560
    # y=740    +--------------- 0 ---------------+
    # y=700    +-----------------------------------+
    # y=650    +-- 1 --+  +-- 3 --+  +-- 2 --+
    # y=500    +-------+  +-------+  +-------+
    # y=400    ========== separator =============
    # y=350    +--------------- 4 ---------------+
    # y=200    +-----------------------------------+
    elements = [
        element(0, 40, 700, 560, 740),  # cid=0, l=40, r=560, b=700, t=740
        element(1, 40, 500, 180, 650),  # cid=1, l=40, r=180, b=500, t=650
        element(2, 400, 500, 540, 650),  # cid=2, l=400, r=540, b=500, t=650
        element(3, 220, 500, 360, 650),  # cid=3, l=220, r=360, b=500, t=650
        element(4, 40, 200, 560, 350),  # cid=4, l=40, r=560, b=200, t=350
    ]
    separator = SeparatorElement(
        cid=-1,
        page_no=1,
        page_size=page_size,
        orientation="horizontal",
        l=40,
        r=560,
        b=400,
        t=400,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    result = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements), page_separators=[separator]
    )

    assert [element.cid for element in result] == [0, 1, 3, 2, 4]


def test_horizontal_separator_finishes_sidecar_before_next_band() -> None:
    page_size = Size(width=600, height=800)

    def element(
        cid: int,
        left: float,
        bottom: float,
        right: float,
        top: float,
        *,
        label: DocItemLabel = DocItemLabel.TEXT,
    ) -> PageElement:
        return PageElement(
            cid=cid,
            text=str(cid),
            page_no=1,
            page_size=page_size,
            label=label,
            l=left,
            r=right,
            b=bottom,
            t=top,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    elements = [
        element(0, 40, 620, 420, 680),
        element(1, 40, 580, 180, 610),
        element(2, 40, 540, 260, 570),
        element(3, 40, 200, 560, 350),
        element(30, 480, 620, 560, 680, label=DocItemLabel.PICTURE),
    ]
    separators = [
        SeparatorElement(
            cid=-1,
            page_no=1,
            page_size=page_size,
            orientation="horizontal",
            l=40,
            r=560,
            b=700,
            t=700,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
        SeparatorElement(
            cid=-2,
            page_no=1,
            page_size=page_size,
            orientation="horizontal",
            l=40,
            r=560,
            b=500,
            t=500,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
    ]

    result = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements), page_separators=separators
    )

    assert [element.cid for element in result] == [0, 1, 2, 30, 3]


def test_horizontal_separator_is_not_a_dilation_anchor() -> None:
    from docling.models.postprocessing.reading_order_rb import (
        _ReadingOrderPredictorState,
    )

    page_size = Size(width=600, height=800)
    element = PageElement(
        cid=0,
        text="text",
        page_no=1,
        page_size=page_size,
        label=DocItemLabel.TEXT,
        l=100,
        r=200,
        b=300,
        t=350,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    separator = SeparatorElement(
        cid=-1,
        page_no=1,
        page_size=page_size,
        orientation="horizontal",
        l=90,
        r=210,
        b=250,
        t=250,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    state = _ReadingOrderPredictorState(
        up_map={0: [], 1: [0]},
        dn_map={0: [1], 1: []},
    )

    result = ReadingOrderPredictor()._do_horizontal_dilation(
        [element, separator],
        copy.deepcopy([element, separator]),
        state,
        vertical_separators=None,
    )

    assert result[0].l == element.l
    assert result[0].r == element.r


def test_horizontal_dilation_preserves_columns_at_graph_branches() -> None:
    page_size = Size(width=600, height=800)

    def element(
        cid: int,
        left: float,
        bottom: float,
        right: float,
        top: float,
        *,
        label: DocItemLabel = DocItemLabel.TEXT,
    ) -> PageElement:
        return PageElement(
            cid=cid,
            text=str(cid),
            page_no=1,
            page_size=page_size,
            label=label,
            l=left,
            r=right,
            b=bottom,
            t=top,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    elements = [
        element(0, 40, 700, 360, 740, label=DocItemLabel.SECTION_HEADER),
        element(1, 40, 620, 180, 680),
        element(2, 40, 520, 180, 600),
        element(3, 220, 620, 360, 680),
        element(4, 220, 520, 360, 600),
        element(30, 40, 400, 360, 500, label=DocItemLabel.PICTURE),
    ]

    result = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements)
    )

    assert [element.cid for element in result] == [0, 1, 2, 3, 4, 30]


def test_graphics_continue_left_to_right_only_across_an_empty_gap() -> None:
    from docling.models.postprocessing.reading_order_rb import (
        _ReadingOrderPredictorState,
    )

    page_size = Size(width=600, height=800)

    def element(cid: int, left: float, right: float) -> PageElement:
        return PageElement(
            cid=cid,
            text=str(cid),
            page_no=1,
            page_size=page_size,
            label=DocItemLabel.PICTURE,
            l=left,
            r=right,
            b=200,
            t=300,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    left = element(0, 40, 120)
    right = element(1, 300, 380)
    predictor = ReadingOrderPredictor()

    empty_gap_state = _ReadingOrderPredictorState()
    predictor._init_l2r_map([left, right], empty_gap_state, vertical_separators=None)
    assert empty_gap_state.l2r_map == {0: 1}

    occupied_gap_state = _ReadingOrderPredictorState()
    predictor._init_l2r_map(
        [left, element(10, 160, 240), right],
        occupied_gap_state,
        vertical_separators=None,
    )
    assert occupied_gap_state.l2r_map == {}


def test_build_page_separators_merges_visible_rules_and_rejects_text_crossing() -> None:
    page_size = Size(width=600, height=800)
    elements = [
        _box(0, DocItemLabel.TEXT, l=40, r=560, b=500, t=650).model_copy(
            update={"text": "upper band"}
        ),
        _box(1, DocItemLabel.TEXT, l=40, r=560, b=150, t=300).model_copy(
            update={"text": "lower band"}
        ),
    ]
    horizontal_line = BoundingBox(
        l=40,
        r=560,
        t=400,
        b=400,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    horizontal_filled_rule = BoundingBox(
        l=40,
        r=560,
        t=399.8,
        b=400.2,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    crossing_vertical = BoundingBox(
        l=300,
        r=300,
        t=100,
        b=700,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    separators = build_page_separators(
        page_no=1,
        page_size=page_size,
        page_elements=elements,
        shape_lines=[horizontal_line, crossing_vertical],
        shape_bounding_boxes=[horizontal_filled_rule],
    )

    assert len(separators) == 1
    assert separators[0].orientation == "horizontal"
    assert separators[0].l == pytest.approx(40)
    assert separators[0].r == pytest.approx(560)


def test_vertical_separator_clamps_horizontal_dilation() -> None:
    element = _box(0, DocItemLabel.TEXT, l=100, r=200, b=200, t=300)
    separator = SeparatorElement(
        cid=-1,
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        orientation="vertical",
        l=250,
        r=250,
        b=100,
        t=400,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    x0, x1 = ReadingOrderPredictor._clamp_dilation_to_vertical_separators(
        element,
        x0=80,
        x1=300,
        vertical_separators=[separator],
    )

    assert x0 == 80
    assert x1 == 250


def _text(cid: int, b: float, t: float, l: float, r: float) -> PageElement:  # noqa: E741
    return _box(cid, DocItemLabel.TEXT, b, t, l, r).model_copy(update={"text": "text"})


def _hline(l: float, r: float, y: float) -> BoundingBox:  # noqa: E741
    return BoundingBox(l=l, r=r, b=y, t=y, coord_origin=CoordOrigin.BOTTOMLEFT)


def _vline(x: float, b: float, t: float) -> BoundingBox:
    return BoundingBox(l=x, r=x, b=b, t=t, coord_origin=CoordOrigin.BOTTOMLEFT)


def _page_separators(
    elements: list[PageElement], lines: list[BoundingBox]
) -> list[SeparatorElement]:
    return build_page_separators(
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        page_elements=elements,
        shape_lines=lines,
        shape_bounding_boxes=None,
    )


def test_build_page_separators_joins_fragmented_rule() -> None:
    elements = [_text(0, 500, 600, 40, 560), _text(1, 200, 300, 40, 560)]
    # One rule drawn as three collinear pieces with sub-point gaps.
    lines = [_hline(381, 560, 400), _hline(40, 200, 400), _hline(200.5, 380, 400)]

    separators = _page_separators(elements, lines)

    assert [(s.l, s.r, s.b) for s in separators] == [(40, 560, 400)]


def test_build_page_separators_joins_short_horizontal_fragments() -> None:
    elements = [_text(0, 500, 600, 40, 560), _text(1, 200, 300, 40, 560)]
    lines = [_hline(40 + i * 40, 79.5 + i * 40, 400) for i in range(13)]
    lines.extend([_hline(100, 130, 390), _hline(40, 70, 380), _hline(72, 102, 380)])

    separators = _page_separators(elements, lines)

    assert [(s.l, s.r, s.b) for s in separators] == [(40, 559.5, 400)]


def test_build_page_separators_joins_short_vertical_fragments() -> None:
    elements = [_text(0, 220, 580, 40, 250), _text(1, 220, 580, 350, 560)]
    lines = [_vline(300, 200 + i * 30, 229.5 + i * 30) for i in range(13)]
    lines.extend([_vline(310, 300, 330), _vline(320, 200, 230), _vline(320, 232, 262)])

    separators = _page_separators(elements, lines)

    assert [(s.l, s.b, s.t) for s in separators] == [(300, 200, 589.5)]


def test_build_page_separators_merges_until_no_rules_overlap() -> None:
    # One greedy pass merged (60..280) into (40..100) and (250..360 @ 400.5)
    # into (300..360 @ 399.5), leaving two overlapping rules at y=400. Each
    # was then strictly above the other, and the reading order never ended.
    elements = [_text(0, 500, 600, 40, 360), _text(1, 200, 300, 40, 360)]
    lines = [
        _hline(300, 360, 399.5),
        _hline(40, 100, 400),
        _hline(60, 280, 400),
        _hline(250, 360, 400.5),
    ]

    separators = _page_separators(elements, lines)

    assert [(s.l, s.r, s.b) for s in separators] == [(40, 360, 400)]


def test_build_page_separators_rejects_rule_through_graphic() -> None:
    # 2206.01062 p9: a chart rule overshoots the predicted picture box, so it
    # is not inside it, yet it runs through it and must not split the page.
    elements = [
        _box(0, DocItemLabel.PICTURE, 343, 707, 54, 549),
        _text(1, 720, 750, 54, 549),
        _text(2, 100, 320, 54, 549),
    ]
    through = _hline(53.0, 575.4, 530.9)
    below = _hline(53.0, 575.4, 331.0)

    separators = _page_separators(elements, [through, below])

    assert [s.b for s in separators] == [331.0]


def test_build_page_separators_rejects_graphic_frame_and_one_sided_rules() -> None:
    elements = [
        _text(0, 600, 700, 100, 500),
        _box(1, DocItemLabel.PICTURE, 300, 500, 100, 500),
    ]
    between = _hline(100, 500, 550)
    # A rule strictly inside a graphic already crosses it; this check is what
    # drops the graphic's own frame.
    frame = _hline(100, 500, 500)
    # Nothing below it to separate from.
    trailing = _hline(100, 500, 200)

    separators = _page_separators(elements, [between, frame, trailing])

    assert [s.b for s in separators] == [550]


def test_vertical_separator_blocks_left_to_right_link() -> None:
    elements = [
        _text(0, 600, 700, 40, 280),
        _text(1, 600, 700, 320, 560),
        _text(2, 400, 580, 40, 280),
    ]
    separator = SeparatorElement(
        cid=-1,
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        orientation="vertical",
        l=300,
        r=300,
        b=380,
        t=720,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )

    baseline = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements)
    )
    separated = ReadingOrderPredictor().predict_reading_order(
        page_elements=copy.deepcopy(elements), page_separators=[separator]
    )

    # 0 -> 1 is a same-row continuation until the rule divides the columns.
    assert [element.cid for element in baseline] == [0, 1, 2]
    assert [element.cid for element in separated] == [0, 2, 1]


def test_consecutive_text_overlapping_slightly_stays_in_sequence() -> None:
    # 1 continues 0 in the left column but its top pokes 1pt into 0, so 0 is
    # not strictly above it; without the link, 0 flows into the right column
    # (2) first.
    elements = [
        _text(0, 600, 700, 40, 560),
        _text(1, 200, 601, 40, 280),
        _text(2, 500, 590, 320, 560),
    ]
    assert not elements[0].is_strictly_above(elements[1])

    result = ReadingOrderPredictor().predict_reading_order(page_elements=elements)

    assert [element.cid for element in result] == [0, 1, 2]


def test_upward_walk_stops_on_up_map_cycle() -> None:
    from docling.models.postprocessing.reading_order_rb import (
        _ReadingOrderPredictorState,
    )

    # Each node above the other, as two overlapping rules at one height were.
    state = _ReadingOrderPredictorState(up_map={0: [1], 1: [0]})

    top = ReadingOrderPredictor()._depth_first_search_upwards(0, [False, False], state)

    assert top == 1


def _thin_rects(count: int, seed: int = 0) -> list[BoundingBox]:
    """Many tiny thin rectangles, like seats on a venue plan.

    Each is thin enough to become a separator candidate but far too short
    to ever be accepted, mirroring docling-project/docling#4632.
    """
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        w = rng.uniform(8, 30)
        h = rng.uniform(1.0, 3.0)
        x = rng.uniform(0, _DUMMY_PAGE_SIZE.width - w)
        y = rng.uniform(150, 650)
        boxes.append(
            BoundingBox(
                l=x,
                r=x + w,
                b=y,
                t=y + h,
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )
        )
    return boxes


def test_build_page_separators_skips_degenerate_candidate_counts() -> None:
    # https://github.com/docling-project/docling/issues/4632 — a page whose
    # vector geometry yields tens of thousands of tiny candidates sent the
    # quadratic candidate merge into an unbounded run. Above the candidate
    # cap the page is skipped quickly instead.
    elements = [_text(0, 500, 600, 40, 560), _text(1, 200, 300, 40, 560)]
    boxes = _thin_rects(6000)

    start = time.monotonic()
    separators = build_page_separators(
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        page_elements=elements,
        shape_lines=None,
        shape_bounding_boxes=boxes,
    )
    elapsed = time.monotonic() - start

    assert separators == []
    assert elapsed < 10


def test_build_page_separators_aborts_on_exhausted_time_budget() -> None:
    # A budget that is already exhausted must skip separator extraction
    # immediately instead of spending more time on an optional refinement.
    elements = [_text(0, 500, 600, 40, 560), _text(1, 200, 300, 40, 560)]
    rule = _hline(40, 560, 400)

    for time_limit_s in (0.0, -5.0):
        separators = build_page_separators(
            page_no=0,
            page_size=_DUMMY_PAGE_SIZE,
            page_elements=elements,
            shape_lines=[rule],
            shape_bounding_boxes=None,
            time_limit_s=time_limit_s,
        )
        assert separators == []


def test_build_page_separators_merges_fragments_below_candidate_cap() -> None:
    # Short fragments must still merge into long rules when the candidate
    # count stays below the degenerate-geometry cap.
    elements = [_text(0, 700, 760, 40, 560), _text(1, 40, 100, 40, 560)]
    lines = []
    for row in range(5):
        y = 200 + row * 100
        x = 40.0
        while x < 560:
            lines.append(_hline(x, x + 6, y))
            x += 6.1

    separators = build_page_separators(
        page_no=0,
        page_size=_DUMMY_PAGE_SIZE,
        page_elements=elements,
        shape_lines=lines,
        shape_bounding_boxes=None,
    )

    assert len(separators) == 5
    assert all(s.orientation == "horizontal" for s in separators)
