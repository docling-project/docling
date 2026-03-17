"""Unit tests for PageAssembleModel."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfHyperlink,
    SegmentedPdfPage,
)
from pydantic import AnyUrl

from docling.datamodel.base_models import Page
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)


@pytest.fixture
def model() -> PageAssembleModel:
    return PageAssembleModel(options=PageAssembleOptions())


class TestSanitizeTextLigatures:
    """Tests for Unicode ligature expansion in sanitize_text()."""

    def test_fi_ligature_no_space(self, model):
        """U+FB01 ﬁ → fi (no spurious space)."""
        assert model.sanitize_text(["\ufb01eld"]) == "field"

    def test_fl_ligature_no_space(self, model):
        """U+FB02 ﬂ → fl (no spurious space)."""
        assert model.sanitize_text(["\ufb02ow"]) == "flow"

    def test_fi_ligature_with_spurious_space(self, model):
        """U+FB01 ﬁ followed by spurious space before word char → fi (space absorbed)."""
        assert model.sanitize_text(["\ufb01 eld"]) == "field"

    def test_fl_ligature_with_spurious_space(self, model):
        """U+FB02 ﬂ followed by spurious space before word char → fl (space absorbed)."""
        assert model.sanitize_text(["\ufb02 ow"]) == "flow"

    def test_ff_ligature(self, model):
        """U+FB00 ﬀ → ff."""
        assert model.sanitize_text(["\ufb00"]) == "ff"

    def test_fi_ligature(self, model):
        """U+FB01 ﬁ → fi."""
        assert model.sanitize_text(["\ufb01"]) == "fi"

    def test_fl_ligature(self, model):
        """U+FB02 ﬂ → fl."""
        assert model.sanitize_text(["\ufb02"]) == "fl"

    def test_ffi_ligature(self, model):
        """U+FB03 ﬃ → ffi."""
        assert model.sanitize_text(["\ufb03"]) == "ffi"

    def test_ffl_ligature(self, model):
        """U+FB04 ﬄ → ffl."""
        assert model.sanitize_text(["\ufb04"]) == "ffl"

    def test_long_st_ligature(self, model):
        """U+FB05 ﬅ → st."""
        assert model.sanitize_text(["\ufb05"]) == "st"

    def test_st_ligature(self, model):
        """U+FB06 ﬆ → st."""
        assert model.sanitize_text(["\ufb06"]) == "st"

    def test_ligature_space_at_word_boundary_preserved(self, model):
        """Space after ligature at word boundary (not before word char) is preserved."""
        assert model.sanitize_text(["\ufb01eld of view"]) == "field of view"

    def test_multiple_ligatures_in_text(self, model):
        """Multiple ligatures in a single text block are all expanded."""
        # "ﬁeld" + space + "ﬂow" → "field flow"
        assert model.sanitize_text(["\ufb01eld \ufb02ow"]) == "field flow"

    def test_ligature_with_spurious_space_in_multiline(self, model):
        """Ligature with spurious space works correctly across multi-line input."""
        assert model.sanitize_text(["\ufb01 eld", "of view"]) == "field of view"


def _make_page(hyperlinks: list[PdfHyperlink], page_height: float = 100.0) -> Page:
    """Create a Page with mocked parsed_page carrying the given hyperlinks."""
    page = Page(page_no=0, size=Size(width=200, height=page_height))
    pp = MagicMock(spec=SegmentedPdfPage)
    pp.hyperlinks = hyperlinks
    page.parsed_page = pp
    return page


def _make_hyperlink(
    left: float,
    bottom: float,
    right: float,
    top: float,
    uri: str | None = "https://example.com",
) -> PdfHyperlink:
    """Create a PdfHyperlink with a BOTTOMLEFT-origin rect."""
    return PdfHyperlink(
        index=0,
        rect=BoundingRectangle(
            r_x0=left,
            r_y0=bottom,
            r_x1=right,
            r_y1=bottom,
            r_x2=right,
            r_y2=top,
            r_x3=left,
            r_y3=top,
        ),
        uri=uri,
    )


class TestMatchHyperlink:
    """Tests for _match_hyperlink() spatial matching."""

    def test_no_hyperlinks(self):
        page = _make_page([])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        assert PageAssembleModel._match_hyperlink(bbox, page) is None

    def test_no_parsed_page(self):
        page = Page(page_no=0, size=Size(width=200, height=100))
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        assert PageAssembleModel._match_hyperlink(bbox, page) is None

    def test_single_hyperlink_full_overlap(self):
        """Hyperlink rect fully covers the cluster → match."""
        # Cluster at TOPLEFT (10, 10)-(90, 20) = BOTTOMLEFT (10, 80)-(90, 90)
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90)
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is not None
        assert str(result) == "https://example.com/"

    def test_below_threshold_returns_none(self):
        """Hyperlink covers <50% of cluster → no match."""
        # Cluster is 80 wide, hyperlink only covers 30 of it
        hl = _make_hyperlink(left=10, bottom=80, right=40, top=90)
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is None

    def test_internal_link_skipped(self):
        """Hyperlink with uri=None (internal PDF link) is skipped."""
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90, uri=None)
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        assert PageAssembleModel._match_hyperlink(bbox, page) is None

    def test_best_uri_wins(self):
        """When two URIs overlap the cluster, the one with higher coverage wins."""
        hl_small = _make_hyperlink(
            left=10, bottom=80, right=50, top=90, uri="https://small.com"
        )
        hl_large = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="https://large.com"
        )
        page = _make_page([hl_small, hl_large])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is not None
        assert str(result) == "https://large.com/"

    def test_multi_rect_same_uri_aggregated(self):
        """Multiple rects for the same URI aggregate coverage above threshold."""
        # Each rect covers ~35% of the cluster, but together they cover ~70%
        hl1 = _make_hyperlink(
            left=10, bottom=80, right=38, top=90, uri="https://wrapped.com"
        )
        hl2 = _make_hyperlink(
            left=38, bottom=80, right=66, top=90, uri="https://wrapped.com"
        )
        page = _make_page([hl1, hl2])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is not None
        assert str(result) == "https://wrapped.com/"

    def test_invalid_url_falls_back_to_path(self):
        """URI that fails AnyUrl validation falls back to Path."""
        hl = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="not a valid url"
        )
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is not None
        assert isinstance(result, Path)

    def test_no_page_size_returns_none(self):
        page = Page(page_no=0, size=None)
        pp = MagicMock(spec=SegmentedPdfPage)
        pp.hyperlinks = [_make_hyperlink(left=10, bottom=80, right=90, top=90)]
        page.parsed_page = pp
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        assert PageAssembleModel._match_hyperlink(bbox, page) is None


class TestMatchHyperlinkTracking:
    """Tests that _match_hyperlink correctly records matched indices."""

    def test_matched_indices_recorded_on_match(self):
        """Successful match records the hyperlink index."""
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90, uri="https://a.com")
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()
        result = PageAssembleModel._match_hyperlink(bbox, page, matched)
        assert result is not None
        assert 0 in matched

    def test_below_threshold_overlap_not_consumed(self):
        """Below-threshold hyperlink is NOT consumed so it becomes a fallback REFERENCE.

        If an inline link or misaligned PDF rect doesn't meet the 50% threshold,
        it should remain unconsumed and be picked up by _collect_unmatched_hyperlinks
        rather than being silently dropped.
        """
        hl = _make_hyperlink(left=10, bottom=80, right=40, top=90, uri="https://a.com")
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()
        result = PageAssembleModel._match_hyperlink(bbox, page, matched)
        assert result is None  # below threshold → no match
        assert len(matched) == 0  # not consumed → becomes fallback REFERENCE

    def test_zero_overlap_not_consumed(self):
        """Hyperlink with zero overlap is not consumed."""
        # Cluster at TOPLEFT (10,10)-(90,20); hyperlink far below at BOTTOMLEFT (10,5)-(90,15)
        hl = _make_hyperlink(left=10, bottom=5, right=90, top=15, uri="https://a.com")
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()
        PageAssembleModel._match_hyperlink(bbox, page, matched)
        assert len(matched) == 0

    def test_multi_rect_same_uri_overlapping_indices_recorded(self):
        """Multiple overlapping rects for same URI → all overlapping indices recorded."""
        hl1 = _make_hyperlink(left=10, bottom=80, right=50, top=90, uri="https://a.com")
        hl2 = _make_hyperlink(left=50, bottom=80, right=90, top=90, uri="https://a.com")
        page = _make_page([hl1, hl2])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()
        PageAssembleModel._match_hyperlink(bbox, page, matched)
        assert matched == {0, 1}

    def test_same_uri_distant_rect_not_consumed(self):
        """Same URI at two locations: only the rect overlapping the cluster is consumed."""
        # Cluster is at TOPLEFT (10,10)-(90,20) → BOTTOMLEFT (10,80)-(90,90)
        hl_near = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="https://a.com"
        )
        # Distant rect (e.g. footer) — no overlap with the cluster at all
        hl_far = _make_hyperlink(
            left=10, bottom=5, right=90, top=15, uri="https://a.com"
        )
        page = _make_page([hl_near, hl_far])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()
        result = PageAssembleModel._match_hyperlink(bbox, page, matched)
        assert result is not None
        # Only the overlapping rect (index 0) should be consumed
        assert matched == {0}

    def test_same_uri_small_clip_not_consumed(self):
        """A nearby same-URI sliver should not suppress a second fallback link."""
        hl_primary = _make_hyperlink(
            left=10,
            bottom=80,
            right=70,
            top=90,
            uri="https://a.com",
        )
        hl_small_clip = _make_hyperlink(
            left=86,
            bottom=80,
            right=100,
            top=90,
            uri="https://a.com",
        )
        page = _make_page([hl_primary, hl_small_clip])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        matched = set()

        result = PageAssembleModel._match_hyperlink(bbox, page, matched)

        assert result is not None
        assert matched == {0}

    def test_none_matched_indices_is_safe(self):
        """Passing matched_indices=None (default) does not error."""
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90)
        page = _make_page([hl])
        bbox = BoundingBox(l=10, t=10, r=90, b=20)
        result = PageAssembleModel._match_hyperlink(bbox, page)
        assert result is not None


class TestCollectUnmatchedHyperlinks:
    """Tests for _collect_unmatched_hyperlinks."""

    def test_unmatched_hyperlink_creates_reference_element(self):
        """An unmatched hyperlink becomes a REFERENCE TextElement."""
        hl = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="https://orphan.com"
        )
        page = _make_page([hl])
        # Index 0 is NOT in matched set → it's unmatched.
        text_bboxes = [BoundingBox(l=10, t=10, r=90, b=20)]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, set(), 100, text_bboxes
        )
        assert len(elements) == 1
        el = elements[0]
        assert el.label == DocItemLabel.REFERENCE
        assert el.text == "https://orphan.com/"
        assert el.hyperlink is not None
        assert str(el.hyperlink) == "https://orphan.com/"
        assert el.cluster.label == DocItemLabel.REFERENCE
        assert el.id == 100

    def test_matched_hyperlinks_excluded(self):
        """Hyperlinks in matched_indices are not collected."""
        hl = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="https://matched.com"
        )
        page = _make_page([hl])
        text_bboxes = [BoundingBox(l=10, t=10, r=90, b=20)]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, {0}, 100, text_bboxes
        )
        assert len(elements) == 0

    def test_repeated_uri_preserved_as_separate_elements(self):
        """Multiple rects for same unmatched URI → one element per rect."""
        hl1 = _make_hyperlink(
            left=10, bottom=80, right=50, top=90, uri="https://dup.com"
        )
        hl2 = _make_hyperlink(
            left=50, bottom=70, right=90, top=80, uri="https://dup.com"
        )
        page = _make_page([hl1, hl2])
        text_bboxes = [
            BoundingBox(l=10, t=10, r=50, b=20),
            BoundingBox(l=50, t=20, r=90, b=30),
        ]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, set(), 100, text_bboxes
        )
        assert len(elements) == 2
        # Each element gets its own bbox matching its hyperlink rect.
        assert elements[0].cluster.bbox.l == 10
        assert elements[0].cluster.bbox.r == 50
        assert elements[1].cluster.bbox.l == 50
        assert elements[1].cluster.bbox.r == 90
        # Distinct cluster IDs.
        assert elements[0].id == 100
        assert elements[1].id == 101

    def test_all_matched_no_elements(self):
        """When all hyperlinks are matched, no synthetic elements are created."""
        hl1 = _make_hyperlink(left=10, bottom=80, right=90, top=90, uri="https://a.com")
        hl2 = _make_hyperlink(left=10, bottom=60, right=90, top=70, uri="https://b.com")
        page = _make_page([hl1, hl2])
        text_bboxes = [
            BoundingBox(l=10, t=10, r=90, b=20),
            BoundingBox(l=10, t=30, r=90, b=40),
        ]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, {0, 1}, 100, text_bboxes
        )
        assert len(elements) == 0

    def test_none_uri_hyperlinks_skipped(self):
        """Hyperlinks with uri=None are never collected."""
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90, uri=None)
        page = _make_page([hl])
        text_bboxes = [BoundingBox(l=10, t=10, r=90, b=20)]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, set(), 100, text_bboxes
        )
        assert len(elements) == 0

    def test_no_hyperlinks_returns_empty(self):
        page = _make_page([])
        elements = PageAssembleModel._collect_unmatched_hyperlinks(page, set(), 100, [])
        assert len(elements) == 0

    def test_invalid_uri_falls_back_to_path(self):
        """Invalid URIs fall back to Path, same as _match_hyperlink."""
        hl = _make_hyperlink(left=10, bottom=80, right=90, top=90, uri="not a url")
        page = _make_page([hl])
        text_bboxes = [BoundingBox(l=10, t=10, r=90, b=20)]
        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page, set(), 100, text_bboxes
        )
        assert len(elements) == 1
        assert isinstance(elements[0].hyperlink, Path)

    def test_non_text_hyperlink_is_not_materialized(self):
        """Hyperlinks that do not overlap text clusters should not become REFERENCE text."""
        hl = _make_hyperlink(
            left=10, bottom=80, right=90, top=90, uri="https://image-link.com"
        )
        page = _make_page([hl])

        elements = PageAssembleModel._collect_unmatched_hyperlinks(
            page,
            set(),
            100,
            [BoundingBox(l=110, t=10, r=190, b=20)],
        )

        assert elements == []
