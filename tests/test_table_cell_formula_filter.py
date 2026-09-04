# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Pre-filter for table-cell formula enrichment.

The cases marked "TR 38.901" are real cells from 3GPP TR 38.901 Table 7.4.1-1, including the
two that shaped the design: an assignment that reads like a bare number after extraction, and
a cell beginning with Private Use Area font glyphs.
"""

import pytest

from docling.models.stages.code_formula.table_cell_formula_vlm_model import (
    cell_text_may_contain_formula,
)


@pytest.mark.parametrize(
    "text",
    [
        # TR 38.901: sigma_SF = 4, extracted in visual order. It looks like it starts with a
        # bare number, which is exactly why the numeric veto has to be anchored.
        "4 SF = σ",  # noqa: RUF001 - the confusable glyph IS the fixture
        # TR 38.901: Symbol-font pieces of the large piecewise brace in the pathloss
        # definition. PUA codepoints carry no Unicode meaning, so this cell is unrecoverable
        # from its text and MUST reach the model.
        " 0.5",
        "a^2 + 8 = 12",
        "$\\sigma_{SF}=4$",
        "32.4 + 20log10(fc)",
        "d_{3D} ≤ 20 m",
        "\\frac{a}{b}",
        "P = ∑ x_i",
        "PL = 28.0 + 22log10(d3D) + 20log10(fc)",
        "h_{BS} ≥ 10 m",
    ],
)
def test_candidate_cells_reach_the_model(text: str) -> None:
    assert cell_text_may_contain_formula(text) is True


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "30",
        "-3.5",
        "1.5e3",
        "1–5",  # noqa: RUF001 - an en-dash range is exactly what must be vetoed
        "20 dB",
        "2 GHz",
        "50 %",
        "30°",
        "dB",
        "NLOS",
        "Scenario",
        "Applicable for all 3GPP scenarios",
        "See Table 7.4.1-1",
        "Optional",
    ],
)
def test_plain_cells_are_not_sent_to_the_model(text: str | None) -> None:
    assert cell_text_may_contain_formula(text) is False
