"""Tests for recovering mis-detected page headers/footers (issue #3693).

The layout model labels regions as PAGE_HEADER/PAGE_FOOTER from per-page position only, so a
non-repeating heading at the top of a page is tagged as a header, moved to the FURNITURE layer
and dropped from the default export. The recovery pass demotes such non-repeating detections
back into the body while leaving genuine running headers/footers as furniture.
"""

from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel

from docling.datamodel.pipeline_options import HeaderFooterOptions
from docling.models.stages.header_footer.header_footer_model import (
    HeaderFooterModel,
    select_header_footer_reclassifications,
)


def _el(label: DocItemLabel, page_no: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(label=label, page_no=page_no, text=text)


def _select(elements, page_count, **opts) -> dict[int, DocItemLabel]:
    options = HeaderFooterOptions(enabled=True, **opts)
    return select_header_footer_reclassifications(elements, page_count, options)


def test_sparse_nonrepeating_headers_are_recovered_genuine_footers_kept():
    # Mirrors the affected PDF (#3693): no real running header, just a few scattered
    # mis-detected top headings across an 8-page doc, plus a genuine footer on every page.
    elements = [
        _el(DocItemLabel.PAGE_HEADER, 1, "Compensation"),  # one-off heading
        _el(DocItemLabel.PAGE_HEADER, 4, "Scheduling"),  # one-off heading
        *[
            _el(DocItemLabel.PAGE_FOOTER, p, f"Republic Airlines  -  {p}")
            for p in range(1, 9)
        ],
    ]
    result = _select(elements, page_count=8)

    # Both bogus headers recovered as section headers; every genuine footer left untouched.
    assert result == {0: DocItemLabel.SECTION_HEADER, 1: DocItemLabel.SECTION_HEADER}


def test_running_header_and_footer_repeating_text_are_kept():
    # Mirrors the "works well" PDF: identical running header + footer on every page.
    elements = []
    for p in range(1, 7):
        elements.append(
            _el(DocItemLabel.PAGE_HEADER, p, "Republic Airways Collective Agreement")
        )
        elements.append(_el(DocItemLabel.PAGE_FOOTER, p, f"Page {p} of 6"))
    assert _select(elements, page_count=6) == {}


def test_section_varying_header_is_kept_via_coverage():
    # Header text changes per page but it is present on every page -> trusted by coverage,
    # even though no single normalized signature repeats.
    elements = [
        _el(DocItemLabel.PAGE_HEADER, p, f"Chapter {chr(ord('A') + p)} Overview")
        for p in range(6)
    ]
    assert _select(elements, page_count=6) == {}


def test_footers_recover_as_plain_text_not_section_header():
    elements = [
        _el(DocItemLabel.PAGE_FOOTER, 2, "A note that only appears once"),
        *[
            _el(DocItemLabel.PAGE_HEADER, p, "Genuine Running Header")
            for p in range(1, 7)
        ],
    ]
    # Footer side coverage is 1/6 and the text does not repeat -> recovered as TEXT.
    assert _select(elements, page_count=6) == {0: DocItemLabel.TEXT}


def test_promote_headers_to_section_disabled_recovers_as_text():
    elements = [
        _el(DocItemLabel.PAGE_HEADER, 1, "Only Once"),
        *[_el(DocItemLabel.PAGE_FOOTER, p, "running footer") for p in range(1, 6)],
    ]
    assert _select(elements, page_count=5, promote_headers_to_section=False) == {
        0: DocItemLabel.TEXT
    }


def test_short_documents_are_left_unchanged():
    elements = [_el(DocItemLabel.PAGE_HEADER, 1, "Title On Page One")]
    # Below min_pages (default 3): repetition/coverage are not observable.
    assert _select(elements, page_count=2) == {}


def test_disabled_is_a_noop():
    elements = [_el(DocItemLabel.PAGE_HEADER, p, f"unique {p}") for p in range(1, 9)]
    options = HeaderFooterOptions(enabled=False)
    assert select_header_footer_reclassifications(elements, 8, options) == {}


def test_blank_detections_are_not_recovered():
    elements = [
        _el(DocItemLabel.PAGE_HEADER, 1, "   "),
        _el(DocItemLabel.PAGE_HEADER, 5, ""),
        *[_el(DocItemLabel.PAGE_FOOTER, p, "footer") for p in range(1, 9)],
    ]
    # Empty/blank headers carry no content to recover and stay as furniture.
    assert _select(elements, page_count=8) == {}


def test_model_relabels_elements_and_moves_them_out_of_headers_bucket():
    header = _el(DocItemLabel.PAGE_HEADER, 1, "Compensation")
    header.cluster = SimpleNamespace(label=DocItemLabel.PAGE_HEADER)
    footers = []
    for p in range(1, 9):
        f = _el(DocItemLabel.PAGE_FOOTER, p, "running footer")
        f.cluster = SimpleNamespace(label=DocItemLabel.PAGE_FOOTER)
        footers.append(f)

    assembled = SimpleNamespace(
        elements=[header, *footers],
        headers=[header],  # page_assemble files headers/footers in the headers bucket
        body=list(footers),
    )
    conv_res = SimpleNamespace(assembled=assembled, pages=[object()] * 8)

    model = HeaderFooterModel(HeaderFooterOptions(enabled=True))
    model(conv_res)

    # Element and its backing cluster are relabeled to a body heading...
    assert header.label == DocItemLabel.SECTION_HEADER
    assert header.cluster.label == DocItemLabel.SECTION_HEADER
    # ...and it is moved out of the furniture headers bucket into the body bucket.
    assert header not in assembled.headers
    assert header in assembled.body
    # Genuine footers are untouched.
    assert all(f.label == DocItemLabel.PAGE_FOOTER for f in footers)
