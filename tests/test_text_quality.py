# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detection of PDF text layers that are not the text drawn on the page."""

import pytest

from docling.utils.text_quality import is_broken_text_layer

_REPORT = (
    "For all new facilities we design and construct in North America, we are "
    "targeting LEED certification. This process is in progress for the following "
    "facilities:\n• Dallas (DFW4)\n• Raleigh-Durham (DUR2)\n• Chicago (CHI6)\n"
    "INTRODUCTION | CORPORATE GOVERNANCE | ENVIRONMENTAL IMPACT\n"
)

# Ordinary text layers that must keep their text, image or not.
GOOD_LAYERS = {
    "bullets_and_pipes": (_REPORT * 2, True),
    "form_lines_and_checkboxes": (
        "Indicate by check mark whether the registrant is a shell company. "
        "Yes ☐ No ☒\nSecurities registered pursuant to Section 12(g) of the Act: "
        "None\n______________________________________\n"
        "0.800% Notes Due 2040 KO40B New York Stock Exchange\n" * 3,
        True,
    ),
    "units_and_maths": (
        "General Description Product Summary\nRDS(ON) (at VGS=10V) < 8.7mΩ\n"
        "Avalanche energy L=0.05mH ± 5%\nTJ, TSTG -55 to 150 °C\nt ≤ 10s "
        "Steady-State\n• Latest advanced trench technology\n" * 3,
        True,
    ),
    "square_markers": (
        "TED KOENIG ■■\nChairman & CEO\nJoined 2004, 41* years of experience in "
        "private credit and leveraged finance across lower middle market deals.\n" * 3,
        True,
    ),
    "icon_font_glyphs": (
        " Hermes Teil von Brabant Konzession  Süd-Ost Brabant "
        "Konzession - Hermes Teil von Arriva Personenvervoer Nederland.\n" * 3,
        True,
    ),
    "asterisk_heavy_report": (
        "Speeches/Conference Papers (150) -- Historical Materials (060)\n"
        "EDRS PRICE MF01/PC01 Plus Postage.\n"
        "***********************************************************************\n"
        "Reproductions supplied by EDRS are the best that can be made\n" * 2,
        True,
    ),
    "dot_leader_table": (
        "Net revenues . . . . . . . . . . . . . . $ 12,345 $ 11,234\n"
        "Operating income . . . . . . . . . . . . . 2,345 — 1,987\n" * 6,
        False,
    ),
    "ukrainian": (
        "Цей документ містить загальну інформацію про страховий продукт. "
        "Повна інформація надається перед укладенням договору.\n" * 4,
        True,
    ),
    "japanese": ("この文書には保険商品に関する一般情報が含まれています。" * 8, True),
    "thai_vowel_signs": (
        "กิจกรรม ปัจฉิมนิเทศ และ กิจกรรม พัฒนาบุคลิกภาพ ของนักเรียนทุกระดับชั้น\n" * 4,
        True,
    ),
    "empty_scan": ("   ", True),
}

# Broken text layers: replaced by OCR when the threshold is set.
BROKEN_LAYERS = {
    "font_encoding_as_control_chars": (
        "&>66*;B \x1b8?.;7*7,. \x182;.,=8;< \x17869.7<*=287#;898<*5< \x01\x13\x12\n"
        * 4,
        False,
    ),
    "control_char_digits": (
        "360 \x0e\x1081, 3\x01317814\x12\x015\x11\x01,578\x015\x0e2\x10,87\n" * 4,
        False,
    ),
    "legacy_font_read_as_latin": (
        "4. H¥$VrH$m`©H«$_mMr {Xem 4. H¥$VrH$m`©H«$_mMr {Xem n§MdmfI© "
        "Am{U Ë`mMm ¶moOZm H$m`©H«$_\n" * 3,
        False,
    ),
    "legacy_font_read_as_accented_latin": (
        "ĊáäåĊĜĈđĂĜûïćĒçĂĒùĒĎđčăđþđĊëùĒ čĜąēïđĉïĉĘĈĘïðĂíïð ĉĘĈĒďĘĆđćĞ\n" * 4,
        False,
    ),
    "private_use_font": ("t o k " * 12, False),
    "shifted_font_with_controls": (
        "&RGHRI(WKLFV $SSURYDOVDQG:DLYHUV :KHQQHFHVVDU\\\x0fDVSHFWVRIWKH\x11\x10\n" * 4,
        True,
    ),
    "punctuation_soup_scan": (
        "..-= ,\": —*- s:. 0 £'S § \"s c.' 3 .»•?•§ ill ^^ ^. C/2 ^ ^ < E- ^\n" * 4,
        True,
    ),
    "poor_hidden_ocr_layer": (
        "tlir Alf SHOUSHA, Pasha, .P..e j.0nal Director, preseh:.ed an apology "
        "tor· ·tile·:···, due. tdt-urgsn~ ·national affa::rs, of the Chairman, "
        "R.I.Dr.N&~b ~. Pl-1th&, the FQ'ptian Mini.star of Health.\n" * 4,
        True,
    ),
    "scan_file_name": ("scan005.TIF", True),
    "image_file_name_with_spaces": ("court document 2.tiff", True),
    "docket_stamp_over_scan": (
        "4-\nI-\nE\n0\nE\n0\nC\n0\nz\nU,\nI-\n0\n[.\nI-.\nLx~\n18\n",
        True,
    ),
}


@pytest.mark.parametrize("name", sorted(GOOD_LAYERS))
def test_ordinary_text_layer_is_kept(name: str) -> None:
    text, has_bitmap = GOOD_LAYERS[name]
    assert is_broken_text_layer(text, has_bitmap=has_bitmap) is False


@pytest.mark.parametrize("name", sorted(BROKEN_LAYERS))
def test_broken_text_layer_is_detected(name: str) -> None:
    text, has_bitmap = BROKEN_LAYERS[name]
    assert is_broken_text_layer(text, has_bitmap=has_bitmap) is True


def test_unusual_symbols_only_count_on_scans() -> None:
    # The same poor-OCR-looking text is left alone on a page without an image.
    text, _ = BROKEN_LAYERS["poor_hidden_ocr_layer"]
    assert is_broken_text_layer(text, has_bitmap=False) is False
