"""Tests for reading font weight and slant out of PDF font names."""

import pytest

from docling.utils.font_style import parse_font_style, weight_class

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
