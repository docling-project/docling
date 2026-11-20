# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.utils.text_normalization import normalize_extracted_text


def test_bullet_point_preserved_and_not_collided_with_middle_dot():
    # Real drug monograph dosage/warning table cell:
    cell_text = "• Withhold dose until recovery • Permanently discontinue if severe"
    normalized = normalize_extracted_text(cell_text)
    assert normalized == cell_text
    assert "•" in normalized
    assert "·" not in normalized


def test_unit_middle_dots_preserved():
    unit_text = "AUC (h·ng/mL), Cmax (µg·hr/mL)"
    normalized = normalize_extracted_text(unit_text)
    assert normalized == unit_text


def test_typographic_and_ligature_normalization():
    # Quotes and fraction slash continue to fold
    input_text = "“Special” ‘quote’ 1⁄2 ﬁle ﬆep"  # noqa: RUF001
    expected = "\"Special\" 'quote' 1/2 file step"
    assert normalize_extracted_text(input_text) == expected
