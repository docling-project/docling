# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detection of PDF text layers that do not match the text drawn on the page."""

import pytest

from docling.utils.text_quality import is_broken_text_layer


def _words(first: int, last: int, count: int = 40) -> str:
    """Words made of the letters in a Unicode range, as a stand-in for a script."""
    letters = [chr(cp) for cp in range(first, last + 1) if chr(cp).isalpha()]
    return " ".join(
        "".join(letters[(i + j) % len(letters)] for j in range(5)) for i in range(count)
    )


_PROSE = (
    "The quarterly report shows that revenue increased across all regions, "
    "driven by strong demand for cloud services and improved margins. "
) * 3


def _shift(text: str, offset: int) -> str:
    """Letters moved by a constant, as a broken ToUnicode map does."""
    return "".join(chr(ord(ch) + offset) if ch.isalpha() else ch for ch in text)


# Ordinary text layers: never broken.
ORDINARY = {
    "latex": r"The loss $L = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ with "
    r"\alpha \in [0,1] and \beta^{2} \leq \gamma_{k}. " * 3,
    "shell": "export PATH=$HOME/bin:$PATH; cd ${USER}/work && ls ~/docs/*.md "
    "| grep -v '#draft' > ~/out/${DATE}.log; echo $? " * 3,
    "python": "def f(x):\n    return {k: v**2 for k, v in x.items() if v > 0}\n"
    "print(f({'a': 1, 'b': -2}))\n" * 3,
    "json": '{"id": 42, "tags": ["a", "b"], "meta": {"ok": true}, "path": "/v/a.log"}\n'
    * 3,
    "markdown_table": "| Metric | Value |\n|---|---|\n| F1 | 0.91 |\n**Note:** see "
    "`config.yaml` and [docs](https://x.y/z).\n" * 3,
    "dot_leaders": "Net revenues . . . . . . $ 12,345 $ 11,234 (1)\n"
    "Operating income . . . . 2,345 (1,987)\n" * 5,
    "caption": "Q3 2024: $12,345 (+5.2%) vs 11,734 in the prior year",
    "units": "RDS(ON) < 8.7 m\N{OHM SIGN} at VGS=10V; TJ = \N{MINUS SIGN}55 to 150 "
    "\N{DEGREE SIGN}C; t \N{LESS-THAN OR EQUAL TO} 10 \N{MICRO SIGN}s. " * 3,
    "bullets_and_checkboxes": "\N{BULLET} Dallas (DFW4)\n\N{BLACK SMALL SQUARE} "
    "Chicago\n\N{BALLOT BOX} No \N{BALLOT BOX WITH X} Yes\n______________\n" * 4,
    "icon_font_glyphs": "".join(chr(0xF002) + " Brabant Konzession, Teil von Arriva. ")
    * 4,
    "accented_european": "L\N{RIGHT SINGLE QUOTATION MARK}\N{LATIN CAPITAL LETTER E WITH ACUTE}cole "
    "doctorale accueille aujourd\N{RIGHT SINGLE QUOTATION MARK}hui des \N{LATIN SMALL LETTER E WITH ACUTE}tudiants "
    "de la r\N{LATIN SMALL LETTER E WITH ACUTE}gion. " * 4,
    "greek": _words(0x03B1, 0x03C9),
    "cyrillic": _words(0x0430, 0x044F),
    "hebrew": _words(0x05D0, 0x05EA),
    "arabic": _words(0x0627, 0x064A),
    "thai": _words(0x0E01, 0x0E2E),
    "devanagari": _words(0x0915, 0x0939),
    "cjk": _words(0x4E00, 0x4E40),
    # Long enough for the letter-frequency check.
    "english_prose": _PROSE,
    "czech": "Vl\N{LATIN SMALL LETTER A WITH ACUTE}da schv\N{LATIN SMALL LETTER A WITH ACUTE}lila "
    "n\N{LATIN SMALL LETTER A WITH ACUTE}vrh z\N{LATIN SMALL LETTER A WITH ACUTE}kona, kter\N{LATIN SMALL LETTER Y WITH ACUTE} "
    "zjednodu\N{LATIN SMALL LETTER S WITH CARON}uje stavebn\N{LATIN SMALL LETTER I WITH ACUTE} "
    "\N{LATIN SMALL LETTER R WITH CARON}\N{LATIN SMALL LETTER I WITH ACUTE}zen\N{LATIN SMALL LETTER I WITH ACUTE} "
    "a podporuje v\N{LATIN SMALL LETTER Y WITH ACUTE}stavbu nov\N{LATIN SMALL LETTER Y WITH ACUTE}ch byt\N{LATIN SMALL LETTER U WITH RING ABOVE} "
    "ve velk\N{LATIN SMALL LETTER Y WITH ACUTE}ch m\N{LATIN SMALL LETTER E WITH CARON}stech. "
    * 3,
    "welsh": "Mae'r llywodraeth wedi cyhoeddi cynllun newydd i gefnogi busnesau bach "
    "yng nghefn gwlad Cymru dros y blynyddoedd nesaf. " * 3,
    "vietnamese": "\N{LATIN CAPITAL LETTER D WITH STROKE}\N{LATIN SMALL LETTER U WITH HORN AND GRAVE}\N{LATIN SMALL LETTER O WITH HORN}ng "
    "\N{LATIN SMALL LETTER D WITH STROKE}\N{LATIN SMALL LETTER E WITH CIRCUMFLEX AND GRAVE}n t\N{LATIN SMALL LETTER U WITH HORN}\N{LATIN SMALL LETTER O WITH HORN}ng "
    "lai c\N{LATIN SMALL LETTER U WITH HOOK ABOVE}a nh\N{LATIN SMALL LETTER U WITH HORN AND TILDE}ng ng\N{LATIN SMALL LETTER U WITH HORN}\N{LATIN SMALL LETTER O WITH HORN AND GRAVE}i "
    "d\N{LATIN SMALL LETTER A WITH CIRCUMFLEX}n Vi\N{LATIN SMALL LETTER E WITH CIRCUMFLEX AND DOT BELOW}t Nam "
    "r\N{LATIN SMALL LETTER A WITH CIRCUMFLEX AND GRAVE}t d\N{LATIN SMALL LETTER A WITH GRAVE}i. "
    * 4,
    "c_code": "for (int i = 0; i < n; i++) { ptr->buf[i] = xmm_mul(cfg.k, src[i]); "
    "if (!chk(ptr)) return -EINVAL; } " * 4,
    "part_numbers": "STM32F407VGT6 LQFP100 GPIO PB12 SPI2_NSS PB13 SPI2_SCK TIM1_CH1N "
    "HCLK PCLK1 PCLK2 RCC_CFGR SYSCLK PLLM PLLN\n" * 4,
    "base64": "TWFueSBoYW5kcyBtYWtlIGxpZ2h0IHdvcmsuIFRoZSBxdWljayBicm93biBmb3gganVtcHM=\n"
    * 5,
    "dna": "ACGTTGCAAGCTTGCATGCCTGCAGGTCGACTCTAGAGGATCCCCGGGTACCGAGCTCGAATTC\n" * 5,
    # Icon fonts repeat a few private-use code points, also in a row.
    "icon_rating_and_footer": _PROSE
    + "\n"
    + chr(0xF0AB) * 4
    + chr(0xF0AA)
    + "\n"
    + " ".join(chr(0xF09A + i) for i in range(5)),
    # One U+FFFD from a math font, on a full page.
    "single_replacement_character": _PROSE + "\nx \N{REPLACEMENT CHARACTER} y\n",
    # pdfium can mark one end-of-line hyphen with a control character.
    "hyphen_control": _PROSE.replace("regions", "re\x02\ngions", 1),
}

# Text layers from fonts whose glyphs were not mapped to real characters.
BROKEN = {
    "glyphs_as_control_codes": "Net assets 2,635 2,849\n"
    + "".join(chr(0x10 + i % 15) for i in range(40))
    + "\nTotal 3,004\n",
    "replacement_characters": "Annual report \N{REPLACEMENT CHARACTER}"
    "\N{REPLACEMENT CHARACTER} 2024 " * 6,
    "glyphs_as_private_use": " ".join(
        "".join(chr(0xF021 + (i + j) % 60) for j in range(4)) for i in range(30)
    ),
    "glyphs_as_accented_latin": _words(0x0100, 0x017F),
    "glyphs_as_shifted_letters": _shift(_PROSE, 3),
    "glyphs_as_letters_and_digits": _shift(_PROSE, -11),
    # One cell in a broken font, on an otherwise good page.
    "one_private_use_cell": _PROSE * 2
    + "\n"
    + "".join(chr(0xF041 + i) for i in range(6))
    + "\n",
    "one_control_code_cell": _PROSE * 2
    + "\nTotal "
    + "".join(chr(0x10 + i) for i in range(6))
    + "\n",
    # Below the share the control check needs, but in runs or in several cells.
    "replacement_run_on_short_page": "Annual report of the regional transport "
    "authority for 2024: \N{REPLACEMENT CHARACTER}\N{REPLACEMENT CHARACTER} "
    "summary, outlook, notes",
    "replacement_characters_in_many_cells": _PROSE
    + "\n"
    + "\n".join("Item " + "\N{REPLACEMENT CHARACTER}" * n for n in (4, 3, 3)),
}


@pytest.mark.parametrize("name", sorted(ORDINARY))
def test_ordinary_text_layer_is_kept(name: str) -> None:
    assert not is_broken_text_layer(ORDINARY[name])


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_unmapped_font_is_detected(name: str) -> None:
    assert is_broken_text_layer(BROKEN[name])


def test_short_or_empty_text_is_left_alone() -> None:
    assert not is_broken_text_layer("")
    assert not is_broken_text_layer("\x01\x02\x03")  # too little text to judge
