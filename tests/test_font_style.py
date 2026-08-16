"""Tests for reading font weight and slant out of PDF font names."""

import pytest
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.common.formatting import Formatting
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfCellRenderingMode,
    PdfTextCell,
    TextCell,
)

from docling.utils.font_style import (
    formatting_from_cells,
    parse_font_style,
    tally_cell_styles,
    weight_class,
)

# Names harvested from the PDFs under tests/data/pdf/sources/, plus the foundry conventions they
# represent. The expectations pin how each convention is read, since there is no standard for
# encoding weight and slant in a font name.
_REAL_FONT_NAMES = [
    # (font name, weight, italic)
    ("/AAAAAC+Verdana-Bold", 700, False),
    ("/NKDKGK+HelveticaNeueLTPro-Bd", 700, False),  # abbreviation as a whole part
    ("/HelveticaNeue-BoldCond", 700, False),  # camel-cased width suffix
    ("/NKDKHL+LubalinGraphStd-Demi", 600, False),  # demi alone means semibold
    ("/AAAAAV+FoundrySterling-Medium", 500, False),  # must not round up to bold
    ("/NKDKFH+HelveticaNeueLTPro-Md", 500, False),
    ("/WACECQ+Times-Roman", 400, False),  # "Roman" is upright regular, not italic
    (
        "/AAAAAR+Avenir-Book",
        400,
        False,
    ),  # "Book" is a weight, unlike the family "Bookman"
    ("/JRMZCQ+MyriadPro-Regular", 400, False),
    ("/KIDKQO+Times-Italic", 400, True),
    ("/BLKGOW+Helvetica-Oblique", 400, True),
    (
        "Arial-BoldItalicMT",
        700,
        True,
    ),  # weight and slant in one part, plus a foundry tag
]

# Names whose styling cannot be read. They must resolve to regular and upright so the heading
# ranking falls back to font size instead of inventing a level.
_UNREADABLE_FONT_NAMES = [
    "/F1",  # resource key: docling-parse reports it when the font dict has no name
    "null",  # docling-parse's placeholder when no name is found at all
    "/AAAAAJ+ArialMT",  # MT is a foundry tag, not a style
    "/NKDKLM+HelveticaNeueLTCom",  # LT/Com are foundry tags
    "/AAAAAE+Verdana",  # bare family name
    "",
]


@pytest.mark.parametrize(("font_name", "weight", "italic"), _REAL_FONT_NAMES)
def test_real_font_names(font_name: str, weight: int, italic: bool):
    style = parse_font_style(font_name)

    assert (style.weight, style.italic) == (weight, italic)
    assert style.known


@pytest.mark.parametrize("font_name", _UNREADABLE_FONT_NAMES)
def test_unreadable_font_names_resolve_to_regular(font_name: str):
    style = parse_font_style(font_name)

    assert (style.weight, style.italic, style.known) == (400, False, False)


@pytest.mark.parametrize(
    "font_name", ["/RWPIRK+LinLibertineTB", "/TKQZJF+LinLibertineTI"]
)
def test_glued_single_letter_suffixes_are_not_read(font_name: str):
    # Linux Libertine encodes bold/italic as a T{B,I} suffix glued to the family name. Reading
    # single letters without a separator would also turn foundry tags into styles, so this
    # regular-and-upright result is a deliberate miss: a wrong bold silently rewrites a heading
    # level, while a miss only falls back to font size.
    assert not parse_font_style(font_name).known


@pytest.mark.parametrize(
    ("font_name", "weight"),
    [
        ("Foo-SemiBold", 600),
        ("Foo-ExtraBold", 800),
        ("Foo-UltraLight", 200),
        ("Foo-Black", 900),
    ],
)
def test_camel_cased_modifiers_combine_with_the_weight(font_name: str, weight: int):
    # Camel-case splitting separates "Semi" from "Bold"; the pair must not be read as plain bold.
    assert parse_font_style(font_name).weight == weight


def test_weight_classes_group_neighbouring_weights():
    # Bold and heavier share a class, medium and semibold share one, everything lighter is regular.
    assert [weight_class(w) for w in (100, 300, 400)] == [0, 0, 0]
    assert [weight_class(w) for w in (500, 600)] == [1, 1]
    assert [weight_class(w) for w in (700, 800, 900)] == [2, 2, 2]


def _rect() -> BoundingRectangle:
    return BoundingRectangle(
        r_x0=0.0,
        r_y0=0.0,
        r_x1=10.0,
        r_y1=0.0,
        r_x2=10.0,
        r_y2=5.0,
        r_x3=0.0,
        r_y3=5.0,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _pdf_cell(
    text: str,
    font_name: str,
    rendering_mode: PdfCellRenderingMode = PdfCellRenderingMode.FILL_TEXT,
) -> PdfTextCell:
    return PdfTextCell(
        rect=_rect(),
        text=text,
        orig=text,
        font_key="F0",
        font_name=font_name,
        widget=False,
        rendering_mode=rendering_mode,
    )


def _ocr_cell(text: str) -> TextCell:
    return TextCell(rect=_rect(), text=text, orig=text, from_ocr=True)


_BOLD = Formatting(bold=True)
_ITALIC = Formatting(italic=True)

_FORMATTING_CASES = [
    # (case id, cells, expected formatting)
    ("uniform bold", [_pdf_cell("Lorem", "Times-Bold")] * 2, _BOLD),
    ("uniform italic", [_pdf_cell("Lorem", "Times-Italic")], _ITALIC),
    (
        "bold italic",
        [_pdf_cell("Lorem", "Arial-BoldItalicMT")],
        Formatting(bold=True, italic=True),
    ),
    # Semibold is emphasis in the heading ranking but not bold text.
    ("semibold is not bold", [_pdf_cell("Lorem", "LubalinGraphStd-Demi")], None),
    # A readable, plain run stays None rather than an all-false Formatting: the field is
    # serialized with exclude_none, so all-false would bloat every text item.
    ("readable and plain", [_pdf_cell("Lorem", "Times-Roman")] * 2, None),
    (
        # The real caption of tests/data/pdf/sources/amt_handbook_sample.pdf.
        "mixed bold label and italic caption",
        [
            _pdf_cell("Figure 7-26.", "Helvetica-Bold"),
            _pdf_cell("Self-locking nuts.", "Times-Italic"),
        ],
        None,
    ),
    (
        "dominant bold outweighs a short unreadable run",
        [_pdf_cell("L" * 90, "Times-Bold"), _pdf_cell("L" * 5, "ArialMT")],
        _BOLD,
    ),
    (
        "short bold run does not carry an unreadable item",
        [_pdf_cell("L" * 5, "Times-Bold"), _pdf_cell("L" * 90, "ArialMT")],
        None,
    ),
    ("ocr cells carry no font", [_ocr_cell("Lorem")] * 2, None),
    (
        "one bold cell among ocr cells",
        [_pdf_cell("Lorem", "Times-Bold")] + [_ocr_cell("Lorem")] * 3,
        None,
    ),
    ("no cells", [], None),
    (
        "whitespace only",
        [_pdf_cell(" ", "Times-Bold"), _pdf_cell("\n", "Times-Bold")],
        None,
    ),
]


@pytest.mark.parametrize(
    ("cells", "expected"),
    [(cells, expected) for _, cells, expected in _FORMATTING_CASES],
    ids=[case_id for case_id, _, _ in _FORMATTING_CASES],
)
def test_formatting_from_cells(cells: list[TextCell], expected: Formatting | None):
    assert formatting_from_cells(cells) == expected


@pytest.mark.parametrize(
    ("font_name", "rendering_mode", "use_rendering_mode", "expected"),
    [
        # A stroked outline is how a PDF fakes bold when no bold font is embedded, so it is read
        # as bold -- but only where the font name itself says nothing.
        ("ArialMT", PdfCellRenderingMode.FILL_THEN_STROKE, True, _BOLD),
        ("ArialMT", PdfCellRenderingMode.STROKE_TEXT, True, _BOLD),
        ("ArialMT", PdfCellRenderingMode.FILL_THEN_STROKE, False, None),
        ("Times-Roman", PdfCellRenderingMode.STROKE_TEXT, True, None),
        ("ArialMT", PdfCellRenderingMode.FILL_TEXT, True, None),
        # Most PDFs never issue a Tr operator, and docling-parse reports UNKNOWN for those.
        ("ArialMT", PdfCellRenderingMode.UNKNOWN, True, None),
    ],
)
def test_stroked_text_is_read_as_bold_only_when_the_name_is_silent(
    font_name: str,
    rendering_mode: PdfCellRenderingMode,
    use_rendering_mode: bool,
    expected: Formatting | None,
):
    cells = [_pdf_cell("Lorem", font_name, rendering_mode)]

    assert (
        formatting_from_cells(cells, use_rendering_mode=use_rendering_mode) == expected
    )


def test_dominant_weight_class_breaks_ties_towards_the_heavier_class():
    # The heading ranking depends on this: emphasis is what makes a heading stand out.
    tally = tally_cell_styles(
        [_pdf_cell("Lorem", "Times-Bold"), _pdf_cell("Lorem", "Times-Roman")]
    )

    assert tally.dominant_weight_class() == 2
