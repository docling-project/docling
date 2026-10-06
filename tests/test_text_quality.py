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
