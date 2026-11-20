# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for character-level normalization of extracted text."""

from pathlib import Path

import pytest
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfCellRenderingMode,
    PdfTextCell,
    TextCell,
)

from docling.utils.text_normalization import (
    _SYMBOL_ENCODING,
    _SYMBOL_PUA_MAP,
    _font_family,
    cell_text_lines,
    normalize_extracted_text,
)

SYMBOL_FONT = "/AAAAAA+SymbolMT"
WINGDINGS_FONT = "/AAAAAA+Wingdings-Regular"
# docling-parse falls back to the resource key when a line mixes fonts, which
# is what the Symbol-in-Helvetica lines of a real document report.
UNKNOWN_FONT = "/F1"


def _rect() -> BoundingRectangle:
    return BoundingRectangle(
        r_x0=0,
        r_y0=0,
        r_x1=10,
        r_y1=0,
        r_x2=10,
        r_y2=10,
        r_x3=0,
        r_y3=10,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def fold(text: str, font_name: str = UNKNOWN_FONT) -> str:
    """Run one cell of the given font through the per-cell fold."""
    cell = PdfTextCell(
        rect=_rect(),
        text=text,
        orig=text,
        rendering_mode=PdfCellRenderingMode.FILL_TEXT,
        widget=False,
        font_key="/F1",
        font_name=font_name,
    )
    return cell_text_lines([cell])[0]


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
        ],
    )
    def test_single_glyph_is_folded(self, raw: str, expected: str) -> None:
        # Wrapped in running text on purpose. When the font is not known, a
        # glyph in leading position is left alone; see TestUnknownFont below.
        assert fold(f"a{raw}b") == f"a{expected}b"

    def test_relaxation_period_line_from_issue_5097(self) -> None:
        """Equals signs also arrive as private-use code points."""
        raw = "Relaxation period, 2 \uf074 \uf03d \uf072 m ne"
        assert fold(raw) == "Relaxation period, 2 \u03c4 = \u03c1 m ne"

    def test_formula_line_from_issue_5097(self) -> None:
        """The reported line, as the PDF text layer carries it."""
        raw = "SHM is given by x = Asin(\uf077t + \uf066) ...(I)"
        assert fold(raw) == "SHM is given by x = Asin(\u03c9t + \u03c6) ...(I)"

    def test_charge_density_line_from_issue_5097(self) -> None:
        raw = "surface charge densities of the spheres (\uf0731 /\uf0732 ) is"
        assert (
            fold(raw)
            == "surface charge densities of the spheres (\u03c31 /\u03c32 ) is"
        )

    def test_navigation_arrow_from_test_corpus(self) -> None:
        """U+F0AE in redp5110_sampled: breadcrumbs that render as nothing."""
        raw = "click Schemas \uf0ae HR_SCHEMA\uf0ae Tables"
        assert fold(raw) == "click Schemas \u2192 HR_SCHEMA\u2192 Tables"

    def test_folding_is_idempotent(self) -> None:
        once = fold("x = Asin(\uf077t + \uf066)")
        assert fold(once) == once


class TestSymbolMapBoundaries:
    """What the Symbol map must not touch."""

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
        assert fold(f"a{code_point}b") == f"a{code_point}b"

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


# Bytes where Symbol reads a Greek letter and Wingdings draws a bullet. Both
# fonts name the glyph for byte 0x6C "uniF06C" and both expose a symbol cmap
# over F020-F0FF, so the code point alone cannot say which font drew it.
# The Wingdings shapes below were read off a rendering of wingding.ttf at
# these code points, not from a published table.
AMBIGUOUS_CODE_POINTS = [
    "\uf06c",  # Symbol lambda  / Wingdings filled circle
    "\uf06d",  # Symbol mu      / Wingdings half-filled circle
    "\uf06f",  # Symbol omicron / Wingdings hollow square
    "\uf070",  # Symbol pi      / Wingdings bold hollow square
    "\uf071",  # Symbol theta   / Wingdings shadowed square
    "\uf072",  # Symbol rho     / Wingdings outlined shadowed square
    "\uf073",  # Symbol sigma   / Wingdings small filled diamond
    "\uf074",  # Symbol tau     / Wingdings filled diamond
    "\uf075",  # Symbol upsilon / Wingdings large filled diamond
    "\uf077",  # Symbol omega   / Wingdings small filled diamond
]


class TestPictorialFontIsNeverFolded:
    """A Wingdings glyph is a picture. It has no Greek letter behind it."""

    @pytest.mark.parametrize("code_point", AMBIGUOUS_CODE_POINTS)
    def test_bullet_is_untouched(self, code_point: str) -> None:
        raw = f"{code_point} Spring, mass set and metre rule"
        assert fold(raw, WINGDINGS_FONT) == raw

    @pytest.mark.parametrize("code_point", AMBIGUOUS_CODE_POINTS)
    def test_glyph_in_running_text_is_also_untouched(self, code_point: str) -> None:
        """The font settles it, so position does not matter here."""
        raw = f"press {code_point} to continue"
        assert fold(raw, WINGDINGS_FONT) == raw

    @pytest.mark.parametrize(
        "font_name",
        ["/Wingdings", "/Wingdings-Regular", "/Webdings", "/ZapfDingbats", "/Marlett"],
    )
    def test_other_pictorial_families_are_refused(self, font_name: str) -> None:
        assert fold("\uf06c item", font_name) == "\uf06c item"


class TestSymbolFontFoldsEverywhere:
    """When the font says Symbol, every glyph in the cell is a Symbol glyph."""

    @pytest.mark.parametrize("code_point", AMBIGUOUS_CODE_POINTS)
    def test_leading_glyph_is_folded_too(self, code_point: str) -> None:
        expected = _SYMBOL_PUA_MAP[code_point]
        assert fold(f"{code_point} = 2", SYMBOL_FONT) == f"{expected} = 2"

    def test_glyph_alone_in_a_cell_is_folded(self) -> None:
        """A table header holding only sigma is recovered."""
        assert fold("\uf073", SYMBOL_FONT) == "\u03c3"


class TestUnknownFontKeepsMarkers:
    """Without a font, a leading glyph may be a bullet, so it is kept.

    docling-parse reports a bare resource key for a line that mixes fonts,
    which is what a Symbol-in-Helvetica line looks like. The fold still runs
    there, but a leading glyph is left exactly as it was before this fold
    existed, so a bullet can never get worse, only stay unfixed.
    """

    @pytest.mark.parametrize("code_point", AMBIGUOUS_CODE_POINTS)
    def test_leading_glyph_is_kept(self, code_point: str) -> None:
        raw = f"{code_point} Spring, mass set and metre rule"
        assert fold(raw) == raw

    @pytest.mark.parametrize("code_point", AMBIGUOUS_CODE_POINTS)
    def test_running_text_is_still_folded(self, code_point: str) -> None:
        expected = _SYMBOL_PUA_MAP[code_point]
        assert fold(f"value {code_point} = 2") == f"value {expected} = 2"

    def test_every_bullet_in_a_multi_line_cell_is_kept(self) -> None:
        """A page backend can put several list items in one TextCell.

        list_marker_processor compiles its patterns with re.DOTALL for this
        reason, so protecting only the first line would leave the rest broken.
        """
        raw = "\uf06c Spring, mass set\n   \uf075 Nested item\n\uf06f Last item"
        assert fold(raw) == raw

    def test_a_bullet_line_and_a_science_line_in_one_cell(self) -> None:
        raw = "\uf06c item\nWavelength \uf06c = 2\uf070"
        assert fold(raw) == "\uf06c item\nWavelength \u03bb = 2\u03c0"


class TestFontFamily:
    """Reducing a PDF font name to a comparable family key."""

    @pytest.mark.parametrize(
        ("font_name", "expected"),
        [
            ("/AAAAAA+Wingdings-Regular", "wingdings"),
            ("/AAAAAA+SymbolMT", "symbolmt"),
            ("/Symbol", "symbol"),
            ("/Helvetica", "helvetica"),
            ("Times-Roman", "times"),
            # Bare resource keys name no family.
            ("/F1", ""),
            ("/C2", ""),
            ("", ""),
        ],
    )
    def test_family_is_extracted(self, font_name: str, expected: str) -> None:
        assert _font_family(font_name) == expected


class TestPlainTextCellHasNoFont:
    """OCR and pypdfium2 cells are plain TextCells, so markers stay protected."""

    @staticmethod
    def _plain(text: str) -> str:
        cell = TextCell(rect=_rect(), text=text, orig=text, from_ocr=False)
        return cell_text_lines([cell])[0]

    def test_leading_glyph_is_kept(self) -> None:
        assert self._plain("\uf06c item") == "\uf06c item"

    def test_running_text_is_folded(self) -> None:
        assert self._plain("2\uf070 r") == "2\u03c0 r"


class TestNormalizeExtractedTextIsFontIndependent:
    """The Symbol fold moved to cell_text_lines, where the font is known."""

    def test_symbol_glyphs_are_not_folded_without_a_cell(self) -> None:
        raw = "Wavelength \uf06c = 2\uf070"
        assert normalize_extracted_text(raw) == raw


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
        assert fold(text) == text


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


class TestMixedSymbolAndWingdingsPdf:
    """The fold over a real PDF that holds both fonts on one page.

    ``tests/data/pdf/symbol_wingdings_mixed.pdf`` was produced with reportlab
    from the Windows ``symbol.ttf`` and ``wingding.ttf`` TrueType fonts. Its
    two ``/ToUnicode`` CMaps are reportlab's own output, not hand-written: the
    Wingdings map sends byte 0x01 to U+F06C and the Symbol map sends byte 0x06
    to that same U+F06C. This is the collision the font check exists for,
    reproduced by an ordinary PDF producer rather than by a crafted file.
    """

    FIXTURE = Path("./tests/data/pdf/symbol_wingdings_mixed.pdf")

    @staticmethod
    def _cells():
        from docling_core.types.doc.page import TextCellUnit

        from docling.backend.docling_parse_backend import (
            ThreadedDoclingParseDocumentBackend,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument

        path = TestMixedSymbolAndWingdingsPdf.FIXTURE
        in_doc = InputDocument(
            path_or_stream=path,
            format=InputFormat.PDF,
            backend=ThreadedDoclingParseDocumentBackend,
        )
        backend = ThreadedDoclingParseDocumentBackend(in_doc, path)
        try:
            page = next(iter(backend.iter_pages()))
            return list(page.get_segmented_page().iterate_cells(TextCellUnit.LINE))
        finally:
            backend.unload()

    def test_the_fixture_really_mixes_the_two_fonts(self) -> None:
        """Guards the premise. Without this the test below proves nothing."""
        fonts = {
            _font_family(cell.font_name)
            for cell in self._cells()
            if isinstance(cell, PdfTextCell)
            and any(0xE000 <= ord(c) <= 0xF8FF for c in cell.text)
        }
        assert "wingdings" in fonts
        # The Symbol runs sit inside Helvetica lines, so docling-parse reports
        # the resource key for them. That is the unknown-font path.
        assert "" in fonts

    def test_science_is_recovered_and_bullets_are_kept(self) -> None:
        cells = self._cells()
        folded = cell_text_lines(cells)
        joined = "\n".join(folded)

        # Greek and the operators come back, including the bytes that a
        # byte-level exclusion would have had to sacrifice.
        assert "Wavelength \u03bb = 2\u03c0 / k" in joined
        assert "x = Asin(\u03c9t + \u03c6) ...(I)" in joined
        assert "2\u03c4 = \u03c1 m ne" in joined
        assert "(\u03c31 /\u03c32 )" in joined
        assert "\u03b8 \u2264 30\u00b0" in joined
        assert "\u2211E = \u221a(2m) \u21d2" in joined
        assert "Schemas \u2192 HR_SCHEMA \u2192 Tables" in joined

        # The Wingdings bullets stay exactly as the text layer had them,
        # refused by the font check rather than by their position.
        assert [c for c in joined if 0xE000 <= ord(c) <= 0xF8FF] == [
            "\uf06c",
            "\uf06c",
            "\uf075",
            "\uf06f",
        ]

    def test_plain_text_is_untouched(self) -> None:
        cells = self._cells()
        folded = cell_text_lines(cells)
        control = [t for t in folded if "Helvetica" in t]
        assert control, "control line missing from the fixture"
        for line in control:
            assert normalize_extracted_text(line) == line
