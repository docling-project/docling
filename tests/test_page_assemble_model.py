"""Unit tests for PageAssembleModel."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import BoundingBox, Size
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

    def test_ij_capital_ligature(self, model):
        """U+0132 Ĳ → IJ (Dutch capital ligature)."""
        assert model.sanitize_text(["\u0132ssel"]) == "IJssel"

    def test_ij_small_ligature(self, model):
        """U+0133 ĳ → ij (Dutch small ligature)."""
        assert model.sanitize_text(["be\u0133"]) == "beij"

    def test_private_use_glyph_stripped(self, model):
        """U+F0A0 private-use glyph is discarded (emitted by some PDF fonts)."""
        assert model.sanitize_text(["hello\uf0a0world"]) == "helloworld"

    def test_private_use_glyph_with_spurious_space_stripped(self, model):
        """U+F0A0 followed by a real word-boundary space preserves the space.

        Unlike true ligatures (which are always intra-word), U+F0A0 maps to "".
        When it sits between two actual words the trailing space is a genuine word
        separator and must be re-emitted so the words remain distinct.
        """
        assert model.sanitize_text(["hello\uf0a0 world"]) == "hello world"

    def test_pua_glyph_at_string_start(self, model):
        """U+F0A0 at start of string is discarded, rest preserved."""
        assert model.sanitize_text(["\uf0a0word"]) == "word"

    def test_pua_glyph_at_string_end(self, model):
        """U+F0A0 at end of string is discarded."""
        assert model.sanitize_text(["word\uf0a0"]) == "word"

    def test_pua_glyph_alone(self, model):
        """U+F0A0 in isolation produces empty string."""
        assert model.sanitize_text(["\uf0a0"]) == ""

    def test_pua_glyph_preserves_word_boundary_space(self, model):
        """U+F0A0 between words preserves the separating space."""
        assert model.sanitize_text(["hello\uf0a0 world"]) == "hello world"

    def test_pua_glyph_no_space_merges(self, model):
        """U+F0A0 with no following space still merges adjacent chars."""
        assert model.sanitize_text(["hello\uf0a0world"]) == "helloworld"

    def test_ij_capital_standalone(self, model):
        """U+0132 as standalone token preserves trailing space."""
        # "IJ is een rivier" — IJ appears as a standalone word
        assert model.sanitize_text(["\u0132 is"]) == "IJ is"

    def test_regex_matches_new_codepoints(self, model):
        """Verify the regex actually matches U+0132, U+0133, U+F0A0."""
        import re

        from docling.models.stages.page_assemble.page_assemble_model import _LIGATURE_RE

        assert _LIGATURE_RE.search("\u0132") is not None, "U+0132 not matched by regex"
        assert _LIGATURE_RE.search("\u0133") is not None, "U+0133 not matched by regex"
        assert _LIGATURE_RE.search("\uf0a0") is not None, "U+F0A0 not matched by regex"


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
