# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for character-level normalization of extracted text."""

import pytest

from docling.utils.text_normalization import (
    _SYMBOL_ENCODING,
    _SYMBOL_PUA_MAP,
    normalize_extracted_text,
)

# The eight code points list_marker_processor treats as list markers. A
# Wingdings-encoded run uses them for bullets, so the Symbol map must not
# claim them or bullet detection would change.
LIST_MARKER_CODE_POINTS = [
    "\uf0b7",
    "\uf0a7",
    "\uf06e",
    "\uf076",
    "\uf0d8",
    "\uf0e0",
    "\uf0fc",
    "\uf0a8",
]


class TestSymbolPrivateUseGlyphs:
    """A PDF whose /ToUnicode map targets the Private-Use Area (issue #5097)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("\uf077", "\u03c9"),  # omega
            ("\uf066", "\u03c6"),  # phi
            ("\uf070", "\u03c0"),  # pi
            ("\uf072", "\u03c1"),  # rho
            ("\uf073", "\u03c3"),  # sigma
            ("\uf074", "\u03c4"),  # tau
            ("\uf061", "\u03b1"),  # alpha
            ("\uf062", "\u03b2"),  # beta
            ("\uf057", "\u03a9"),  # Omega
            ("\uf044", "\u0394"),  # Delta
            ("\uf0de", "\u21d2"),  # rightwards double arrow
            ("\uf0ae", "\u2192"),  # rightwards arrow
            ("\uf0d6", "\u221a"),  # square root
            ("\uf0e5", "\u2211"),  # n-ary summation
            # Symbol leaves ASCII punctuation and digits where ASCII has them,
            # but U+F03D is not "=", so these need folding just as much.
            ("\uf03d", "="),
            ("\uf02b", "+"),
            ("\uf028", "("),
            ("\uf029", ")"),
            ("\uf032", "2"),
            ("\uf02f", "/"),
            ("\uf020", " "),
        ],
    )
    def test_single_glyph_is_folded(self, raw: str, expected: str) -> None:
        assert normalize_extracted_text(raw) == expected

    def test_relaxation_period_line_from_issue_5097(self) -> None:
        """Equals signs also arrive as private-use code points."""
        raw = "Relaxation period, 2 \uf074 \uf03d \uf072 m ne"
        assert (
            normalize_extracted_text(raw) == "Relaxation period, 2 \u03c4 = \u03c1 m ne"
        )

    def test_formula_line_from_issue_5097(self) -> None:
        """The reported line, as the PDF text layer carries it."""
        raw = "SHM is given by x = Asin(\uf077t + \uf066) ...(I)"
        assert (
            normalize_extracted_text(raw)
            == "SHM is given by x = Asin(\u03c9t + \u03c6) ...(I)"
        )

    def test_charge_density_line_from_issue_5097(self) -> None:
        raw = "surface charge densities of the spheres (\uf0731 /\uf0732 ) is"
        assert (
            normalize_extracted_text(raw)
            == "surface charge densities of the spheres (\u03c31 /\u03c32 ) is"
        )

    def test_navigation_arrow_from_test_corpus(self) -> None:
        """U+F0AE in redp5110_sampled: breadcrumbs that render as nothing."""
        raw = "click Schemas \uf0ae HR_SCHEMA\uf0ae Tables"
        assert (
            normalize_extracted_text(raw)
            == "click Schemas \u2192 HR_SCHEMA\u2192 Tables"
        )

    def test_folding_is_idempotent(self) -> None:
        once = normalize_extracted_text("x = Asin(\uf077t + \uf066)")
        assert normalize_extracted_text(once) == once


class TestSymbolMapBoundaries:
    """What the Symbol map must not touch."""

    @pytest.mark.parametrize("code_point", LIST_MARKER_CODE_POINTS)
    def test_list_marker_code_points_are_not_claimed(self, code_point: str) -> None:
        assert code_point not in _SYMBOL_PUA_MAP
        assert normalize_extracted_text(f"{code_point} item") == f"{code_point} item"

    @pytest.mark.parametrize(
        "code_point",
        [
            "\uf0e6",  # parenlefttp  -- bracket builder, no Unicode equivalent
            "\uf0e7",  # parenleftex
            "\uf0e8",  # parenleftbt
            "\uf0f6",  # parenrighttp
            "\uf0f7",  # parenrightex
            "\uf0f8",  # parenrightbt
        ],
    )
    def test_drawing_fragments_are_left_alone(self, code_point: str) -> None:
        assert code_point not in _SYMBOL_PUA_MAP
        assert normalize_extracted_text(code_point) == code_point

    def test_private_use_discard_glyph_still_discarded(self) -> None:
        """U+F0A0 keeps the _LIGATURE_MAP behaviour added in #3254."""
        assert "\uf0a0" not in _SYMBOL_PUA_MAP
        assert normalize_extracted_text("hello\uf0a0 world") == "hello world"

    def test_map_covers_only_the_symbol_private_use_block(self) -> None:
        assert all(0xF020 <= ord(c) <= 0xF0FF for c in _SYMBOL_PUA_MAP)

    def test_map_targets_are_never_private_use(self) -> None:
        """Folding a PUA glyph to another PUA code point would be pointless."""
        for char in _SYMBOL_ENCODING.values():
            assert not any(0xE000 <= ord(c) <= 0xF8FF for c in char)


class TestUnaffectedText:
    """Normalization must not disturb text that was already correct."""

    @pytest.mark.parametrize(
        "text",
        [
            "Plain ASCII stays exactly as it is.",
            "Accented Latin: Ore\u0161i\u010d, na\u00efve, Z\u00fcrich.",
            "Real Greek already decoded: \u03c9t + \u03c6 = \u03c0.",
            "Math already decoded: 2\u03c0 / \u03c9 \u21d2 n = 1.",
            "CJK unaffected: \u6587\u5b57\u5217\u3002",
        ],
    )
    def test_text_is_unchanged(self, text: str) -> None:
        assert normalize_extracted_text(text) == text


class TestExistingNormalizationStillApplies:
    """Regression cover for behaviour that predates the Symbol map."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("\ufb01eld", "field"),  # fi ligature
            ("\ufb01 eld", "field"),  # with the parser's spurious space
            ("\u0132 is", "IJ is"),  # Dutch IJ keeps the word boundary
            ("\u2018quoted\u2019", "'quoted'"),  # typographic quotes
            ("\u201cquoted\u201d", '"quoted"'),  # typographic double quotes
            ("a\u2044b", "a/b"),  # fraction slash
        ],
    )
    def test_ligatures_and_typography(self, raw: str, expected: str) -> None:
        assert normalize_extracted_text(raw) == expected

    def test_arabic_presentation_forms_are_folded(self) -> None:
        # U+FEDF/U+FEE4 are shaped forms of lam and meem.
        assert normalize_extracted_text("\ufedf\ufee4") == "\u0644\u0645"
