"""Unit tests for PageAssembleModel.sanitize_text(), focusing on ligature normalization."""

import pytest

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
