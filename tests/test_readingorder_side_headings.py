"""Reading-order handling of left-margin "side-heading" layouts.

Regression coverage for docling-project/docling#3643: narrow section headers in
a left column must be interleaved with the paragraph block on their right
instead of being emitted as one block of headers followed by one block of
paragraphs. Ordinary single- and multi-column layouts must stay untouched.
"""

from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel
from docling_ibm_models.reading_order.reading_order_rb import PageElement

from docling.models.stages.reading_order.readingorder_model import ReadingOrderModel

_PAGE = Size(width=600, height=800)


def _el(
    cid: int,
    label: DocItemLabel,
    left: float,
    right: float,
    bottom: float,
    top: float,
):
    # Bottom-left origin (as produced by predict_reading_order): larger y = higher.
    return PageElement(
        cid=cid,
        ref=RefItem(cref=f"#/0/{cid}"),
        text=str(cid),
        page_no=0,
        page_size=_PAGE,
        label=label,
        l=left,
        r=right,
        b=bottom,
        t=top,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


_H = DocItemLabel.SECTION_HEADER
_T = DocItemLabel.TEXT
_PF = DocItemLabel.PAGE_FOOTER


def test_side_headings_are_interleaved_with_their_paragraph():
    # Narrow headers in a left column, paragraphs in a wider right column. The
    # predictor reads the left column first, so the input is all headers then
    # all paragraphs.
    h1 = _el(0, _H, 50, 150, 700, 750)
    p1 = _el(1, _T, 200, 550, 640, 750)
    h2 = _el(2, _H, 50, 150, 500, 550)
    p2 = _el(3, _T, 200, 550, 400, 550)
    h3 = _el(4, _H, 50, 150, 250, 300)
    p3 = _el(5, _T, 200, 550, 150, 300)
    predicted = [h1, h2, h3, p1, p2, p3]

    result = ReadingOrderModel._reorder_side_headings(predicted)

    assert [e.cid for e in result] == [0, 1, 2, 3, 4, 5]


def test_single_column_layout_is_unchanged():
    # Headers span the same column as the text below them -> normal headers.
    h1 = _el(0, _H, 50, 550, 700, 750)
    p1 = _el(1, _T, 50, 550, 540, 690)
    h2 = _el(2, _H, 50, 550, 480, 530)
    p2 = _el(3, _T, 50, 550, 300, 470)
    predicted = [h1, p1, h2, p2]

    result = ReadingOrderModel._reorder_side_headings(predicted)

    assert [e.cid for e in result] == [0, 1, 2, 3]


def test_real_world_full_width_line_does_not_block_interleaving():
    """Regression for TechnicalSupplement.pdf page 1 (reported on PR #3648).

    A full-width trademark line spans the whole page, so a naive "header is
    disjoint from all body" guard misclassifies every left-margin header and the
    page title ends up 4th. The geometry below is taken verbatim from that page
    (bounding boxes only, no text), in the order the rule-based predictor emits
    it: the left-column headers first, then the right column.
    """
    # cid: (label, left, right, bottom, top); bottom-left origin.
    geom = {
        0: (_H, 195.2, 470.0, 479.2, 511.6),  # page title (right column)
        1: (_T, 195.7, 460.3, 466.5, 477.4),  # subtitle
        2: (_H, 73.1, 183.1, 437.1, 447.8),  # "Package Contents"
        3: (_T, 195.7, 512.9, 409.2, 447.5),
        4: (_H, 55.8, 183.2, 392.4, 403.1),  # "Required Equipment"
        5: (_T, 195.9, 513.4, 373.1, 401.6),
        6: (_T, 196.1, 513.0, 343.1, 371.4),
        7: (_T, 196.2, 513.5, 283.7, 341.5),
        8: (_T, 196.4, 513.4, 253.7, 282.0),
        9: (_H, 55.3, 183.9, 237.0, 247.7),  # "Installation Concepts"
        10: (_T, 196.7, 513.8, 198.4, 246.1),
        11: (_T, 196.9, 514.0, 139.3, 196.9),
        13: (_T, 49.1, 512.5, 41.3, 60.8),  # full-width trademark line
        12: (_PF, 197.9, 361.5, 62.0, 70.2),
        14: (_PF, 279.4, 281.4, 14.6, 21.9),
    }
    els = {cid: _el(cid, *args) for cid, args in geom.items()}
    predictor_order = [2, 4, 9, 0, 1, 3, 5, 6, 7, 8, 10, 11, 13, 12, 14]
    predicted = [els[cid] for cid in predictor_order]

    result = [e.cid for e in ReadingOrderModel._reorder_side_headings(predicted)]

    # Title first, then each side-heading immediately before its paragraph(s).
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 14]


def test_true_two_column_layout_is_unchanged():
    # Each header heads its own column (horizontal overlap with that column's
    # body), so nothing should be re-interleaved across columns.
    lh = _el(0, _H, 50, 290, 700, 750)
    lp = _el(1, _T, 50, 290, 400, 690)
    rh = _el(2, _H, 310, 550, 700, 750)
    rp = _el(3, _T, 310, 550, 400, 690)
    predicted = [lh, lp, rh, rp]

    result = ReadingOrderModel._reorder_side_headings(predicted)

    assert [e.cid for e in result] == [0, 1, 2, 3]
