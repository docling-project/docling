# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
"""Unit tests for the AcroForm widget->label order-preserving binding.

Pure geometry over synthetic bboxes -- no conversion, no fixtures. This is where
the minimization property lives: `_match_labels` binds each widget to the nearest
unconsumed label at or after the last binding in reading order, so the tests that
matter exercise the no-crossing guard and the 1:1 exhaustion, not distance alone.

Coordinates are top-left origin (y grows downward). A synthetic line-height of 10
gives cap = 20 (two lines) and a row band of 15 throughout.
"""

from docling_core.types.doc import BoundingBox, DocItemLabel

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.form_field_model import (
    _gap,
    _match_labels,
    _precedes,
)

LINE = 10.0
CAP = 2 * LINE
ROW_BAND = 1.5 * LINE


def _bbox(left: float, t: float, r: float, b: float) -> BoundingBox:
    return BoundingBox(l=left, t=t, r=r, b=b)


def _label(id_: int, left: float, t: float, r: float, b: float) -> Cluster:
    return Cluster(id=id_, label=DocItemLabel.TEXT, bbox=_bbox(left, t, r, b))


def test_gap_is_zero_on_overlap_and_sums_axis_distances():
    w = _bbox(100, 50, 140, 60)
    assert _gap(w, _bbox(100, 50, 140, 60)) == 0.0  # identical rects
    assert _gap(w, _bbox(10, 50, 90, 60)) == 10.0  # 10 to the left, same row
    assert _gap(w, _bbox(100, 30, 140, 45)) == 5.0  # 5 above, aligned column
    assert _gap(w, _bbox(10, 30, 90, 45)) == 10 + 5  # diagonal: both axes count


def test_label_left_above_and_below_all_bind():
    # One widget per case; the gap alone picks the direction, no per-direction rule.
    left = _match_labels(
        [(0, _bbox(100, 10, 140, 20))], [_label(1, 10, 10, 90, 20)], CAP, ROW_BAND
    )
    above = _match_labels(
        [(0, _bbox(100, 50, 140, 60))], [_label(1, 100, 30, 140, 45)], CAP, ROW_BAND
    )
    below = _match_labels(
        [(0, _bbox(100, 50, 140, 60))], [_label(1, 100, 70, 140, 85)], CAP, ROW_BAND
    )
    assert left[0].id == above[0].id == below[0].id == 1


def test_aligned_label_beats_diagonal_distractor():
    w = _bbox(100, 50, 140, 60)
    aligned = _label(1, 10, 50, 90, 60)  # gap 10, same row
    diagonal = _label(2, 10, 10, 90, 20)  # gap 10 + 30, corner-to-corner
    bound = _match_labels([(0, w)], [aligned, diagonal], CAP, ROW_BAND)
    assert bound[0].id == 1


def test_second_widget_does_not_steal_the_first_widgets_label():
    # Two rows in reading order; L1 is within cap of BOTH widgets, but it is
    # consumed by w1, so w2 must fall through to its own row's label, not steal L1.
    w1, w2 = _bbox(100, 10, 140, 20), _bbox(100, 30, 140, 40)
    l1, l2 = _label(1, 10, 10, 90, 20), _label(2, 10, 30, 90, 40)
    bound = _match_labels([(0, w1), (1, w2)], [l1, l2], CAP, ROW_BAND)
    assert bound[0].id == 1
    assert bound[1].id == 2


def test_crossing_guard_refuses_to_bind_a_label_before_the_frontier():
    # Messy page: widget index order (w1 then w2) disagrees with geometry (w1 is
    # physically below w2). w1 binds its own row-2 label, moving the frontier down;
    # w2's row-1 label now precedes the frontier, so the guard keeps w2 keyless
    # rather than create a crossing. Phase-1 policy: no crossings over recall.
    w1, w2 = _bbox(100, 60, 140, 70), _bbox(100, 10, 140, 20)
    l_row2, l_row1 = _label(1, 10, 60, 90, 70), _label(2, 10, 10, 90, 20)
    bound = _match_labels([(0, w1), (1, w2)], [l_row2, l_row1], CAP, ROW_BAND)
    assert bound[0].id == 1
    assert 1 not in bound  # w2 (index 1) stays unbound -- no crossing


def test_field_with_no_label_within_cap_stays_keyless():
    w = _bbox(100, 50, 140, 60)
    far = _label(1, 10, 500, 90, 510)  # far below, gap >> cap
    assert _match_labels([(0, w)], [far], CAP, ROW_BAND) == {}


def test_shared_header_binds_one_widget_the_rest_skip():
    # A column header over two fields is 1:1 in Phase 1: w1 takes it, w2 finds it
    # used. Matrix/column association is a later phase.
    header = _label(1, 10, 10, 200, 20)
    w1, w2 = _bbox(10, 30, 50, 40), _bbox(100, 30, 140, 40)
    bound = _match_labels([(0, w1), (1, w2)], [header], CAP, ROW_BAND)
    assert bound == {0: header} or (bound[0].id == 1 and 1 not in bound)


def test_precedes_orders_by_row_then_left():
    upper = _bbox(200, 10, 280, 20)
    lower_left = _bbox(10, 60, 90, 70)
    same_row_left = _bbox(10, 12, 90, 22)  # within row band of `upper`, further left
    assert _precedes(upper, lower_left, ROW_BAND)  # earlier row precedes
    assert not _precedes(lower_left, upper, ROW_BAND)
    assert _precedes(same_row_left, upper, ROW_BAND)  # same row, to the left
    assert not _precedes(upper, same_row_left, ROW_BAND)
