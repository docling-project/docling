# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for heading numbers built from the ``w:lvlText`` of a numbering level.

Kept separate from ``test_backend_msword.py`` so that file stays under the
repository's per-file line limit.
"""

from docling_core.types.doc import DoclingDocument, SectionHeaderItem
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

from .test_backend_msword_lists import _convert_docx

_NUM_ID = 91


def _numbered_headings_docx(
    levels: list[tuple[str, str | None]],
    headings: list[tuple[str, int]],
    starts: dict[int, int] | None = None,
):
    """Build a document whose heading styles use one numbering definition.

    ``levels`` gives the ``(w:numFmt, w:lvlText)`` of each numbering level, from
    level 0. A ``w:lvlText`` of ``None`` leaves out the element. The style ``Heading N`` uses level ``N - 1``, as in a Word document
    with numbered headings. ``starts`` maps a level to its ``w:start``, which is
    1 by default. ``headings`` are ``(text, heading level)`` pairs. A heading
    level of 0 gives a paragraph without a heading style.
    """
    starts = starts or {}
    document = Document()
    numbering = document.part.numbering_part.element
    level_xml = "".join(
        f'<w:lvl w:ilvl="{ilvl}"><w:start w:val="{starts.get(ilvl, 1)}"/>'
        f'<w:numFmt w:val="{num_fmt}"/>'
        + ("" if lvl_text is None else f'<w:lvlText w:val="{lvl_text}"/>')
        + "</w:lvl>"
        for ilvl, (num_fmt, lvl_text) in enumerate(levels)
    )
    # All w:abstractNum elements come before the first w:num.
    numbering.find(qn("w:num")).addprevious(
        parse_xml(
            f'<w:abstractNum {nsdecls("w")} w:abstractNumId="90">'
            f"{level_xml}</w:abstractNum>"
        )
    )
    numbering.append(
        parse_xml(
            f'<w:num {nsdecls("w")} w:numId="{_NUM_ID}">'
            '<w:abstractNumId w:val="90"/></w:num>'
        )
    )
    for ilvl in range(len(levels)):
        style = document.styles[f"Heading {ilvl + 1}"]
        num_pr = style.element.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = ilvl
        num_pr.get_or_add_numId().val = _NUM_ID
    for text, heading_level in headings:
        if heading_level:
            document.add_heading(text, level=heading_level)
        else:
            document.add_paragraph(text)
    return document


def _heading_texts(doc: DoclingDocument) -> list[str]:
    return [
        item.text
        for item, _ in doc.iterate_items()
        if isinstance(item, SectionHeaderItem)
    ]


def test_heading_number_keeps_the_text_of_the_level_template(tmp_path):
    """Text around %N in the template is part of the heading number.

    The template of level 1 is ``1.1.%2``, so its headings have the numbers
    ``1.1.1``, ``1.1.2`` and ``1.1.3``. No heading uses level 0.
    """
    document = _numbered_headings_docx(
        [
            ("decimal", "%1"),
            ("decimal", "1.1.%2"),
            ("decimal", "1.1.%2.%3"),
        ],
        [
            ("Purpose", 2),
            ("Scope", 2),
            ("Vessels", 3),
            ("Offices", 3),
            ("Language", 2),
            ("Records", 3),
        ],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "1.1.1 Purpose",
        "1.1.2 Scope",
        "1.1.2.1 Vessels",
        "1.1.2.2 Offices",
        "1.1.3 Language",
        "1.1.3.1 Records",
    ]
    assert doc.export_to_markdown() == (
        "### 1.1.1 Purpose\n"
        "\n"
        "### 1.1.2 Scope\n"
        "\n"
        "#### 1.1.2.1 Vessels\n"
        "\n"
        "#### 1.1.2.2 Offices\n"
        "\n"
        "### 1.1.3 Language\n"
        "\n"
        "#### 1.1.3.1 Records"
    )


def test_heading_number_reads_one_digit_after_the_percent_sign(tmp_path):
    """A placeholder is % and one digit, and a level of format none shows no number.

    ``%113.1.%2`` is ``%1``, the text ``13.1.`` and ``%2``. Level 0 has the
    number format ``none``, so ``%1`` gives no text.
    """
    document = _numbered_headings_docx(
        [
            ("none", "13.1"),
            ("decimal", "%113.1.%2"),
            ("decimal", "%113.1.%2.%3"),
        ],
        [("Objectives", 2), ("Governance", 2), ("Leadership", 3), ("Policy", 3)],
        starts={0: 20},
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "13.1.1 Objectives",
        "13.1.2 Governance",
        "13.1.2.1 Leadership",
        "13.1.2.2 Policy",
    ]


def test_heading_number_keeps_a_word_in_the_level_template(tmp_path):
    """A word in the w:lvlText of a level is part of the heading number."""
    document = _numbered_headings_docx(
        [("decimal", "Chapter %1:"), ("decimal", "%1.%2")],
        [("Safety", 1), ("Drills", 2), ("Navigation", 1), ("Passage plan", 2)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "Chapter 1: Safety",
        "1.1 Drills",
        "Chapter 2: Navigation",
        "2.1 Passage plan",
    ]


def test_heading_number_names_only_the_levels_in_the_template(tmp_path):
    """A template without %1 shows no counter of level 0."""
    document = _numbered_headings_docx(
        [("decimal", "%1"), ("decimal", "%2")],
        [("Deck", 1), ("Mooring", 2), ("Anchoring", 2), ("Engine", 1), ("Fuel", 2)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "1 Deck",
        "1 Mooring",
        "2 Anchoring",
        "2 Engine",
        "1 Fuel",
    ]


def test_heading_number_uses_the_number_format_of_each_level(tmp_path):
    """Each %N in the template of a heading uses the w:numFmt of its level."""
    document = _numbered_headings_docx(
        [("upperLetter", "%1"), ("decimal", "%1.%2"), ("lowerRoman", "%1.%2.%3")],
        [
            ("Annex", 1),
            ("Forms", 2),
            ("Checklists", 2),
            ("Bridge", 3),
            ("Engine room", 3),
            ("Second annex", 1),
            ("Contacts", 2),
        ],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "A Annex",
        "A.1 Forms",
        "A.2 Checklists",
        "A.2.i Bridge",
        "A.2.ii Engine room",
        "B Second annex",
        "B.1 Contacts",
    ]


def test_heading_number_shows_the_start_value_of_a_level_without_a_heading(tmp_path):
    """A %N for a level without a heading shows the start value of that level."""
    document = _numbered_headings_docx(
        [("decimal", "Part %1"), ("decimal", "Part %1, section %2")],
        [("Scope", 2), ("Terms", 2)],
        starts={0: 4},
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == ["Part 4, section 1 Scope", "Part 4, section 2 Terms"]


def test_heading_number_starts_a_skipped_level_again_after_its_parent(tmp_path):
    """A level without a heading under the current parent shows its start value.

    ``Wires`` (level 3) comes directly after ``Engine`` (level 1). Its ``%2``
    shows 1, not the 3 of ``Lines``, which is under ``Deck``.
    """
    document = _numbered_headings_docx(
        [("decimal", "%1"), ("decimal", "%2"), ("decimal", "%2.%3")],
        [
            ("Deck", 1),
            ("Mooring", 2),
            ("Anchoring", 2),
            ("Lines", 2),
            ("Ropes", 3),
            ("Engine", 1),
            ("Wires", 3),
        ],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "1 Deck",
        "1 Mooring",
        "2 Anchoring",
        "3 Lines",
        "3.1 Ropes",
        "2 Engine",
        "1.1 Wires",
    ]


def test_heading_template_without_a_placeholder_is_the_heading_number(tmp_path):
    """A template without %N is shown as it is, and an empty one shows no number."""
    document = _numbered_headings_docx(
        [("decimal", "Annex"), ("decimal", "")],
        [("Deck", 1), ("Mooring", 2), ("Engine", 1)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == ["Annex Deck", "Mooring", "Annex Engine"]


def test_heading_level_without_a_template_shows_no_number(tmp_path):
    """A level without w:lvlText shows no number."""
    document = _numbered_headings_docx(
        [("decimal", None)],
        [("Deck", 1), ("Engine", 1)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == ["Deck", "Engine"]


def test_plain_decimal_template_gives_the_counters_joined_with_dots(tmp_path):
    """A template that only joins decimal counters gives them joined with dots.

    For ``%1.`` and ``%1.%2.`` the heading numbers are ``1`` and ``1.1``,
    without the last dot of the template.
    """
    document = _numbered_headings_docx(
        [("decimal", "%1."), ("decimal", "%1.%2.")],
        [("Deck", 1), ("Mooring", 2), ("Anchoring", 2), ("Engine", 1), ("Fuel", 2)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == [
        "1 Deck",
        "1.1 Mooring",
        "1.2 Anchoring",
        "2 Engine",
        "2.1 Fuel",
    ]


def test_heading_without_a_number_stays_between_template_numbers(tmp_path):
    """A heading of a level with the format none gets no number and is not counted."""
    document = _numbered_headings_docx(
        [("decimal", "%1"), ("decimal", "1.1.%2"), ("none", "")],
        [("Purpose", 2), ("Note", 3), ("Body text.", 0), ("Scope", 2)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _heading_texts(doc) == ["1.1.1 Purpose", "Note", "1.1.2 Scope"]
