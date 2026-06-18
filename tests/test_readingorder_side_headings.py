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
