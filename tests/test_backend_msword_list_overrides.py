# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the ``w:lvlOverride`` of Word numbering instances and the start values
of list levels.

Kept separate from ``test_backend_msword_lists.py`` so that file stays under the
repository's per-file line limit.
"""

import pytest
from docling_core.types.doc import DoclingDocument, ListItem
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

from .test_backend_msword_lists import _convert_docx

_DECIMAL_LEVEL = (
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>'
    '<w:lvlText w:val="%1."/></w:lvl>'
)


def _numbering_instances_docx(
    level_overrides: dict[int, str | None], paragraphs: list[tuple[str, int | None]]
):
    """Build a document whose numbering instances share one abstract definition.

    The abstract numbering definition has one level: decimal, ``%1.``, start 1.
    ``level_overrides`` maps the numId of each numbering instance to the content
    of its ``w:lvlOverride`` for level 0, or to ``None`` for no override.
    ``paragraphs`` are ``(text, numId)`` pairs at level 0. A numId of ``None``
    gives a paragraph without numbering.
    """
    document = Document()
    numbering = document.part.numbering_part.element
    # All w:abstractNum elements come before the first w:num.
    numbering.find(qn("w:num")).addprevious(
        parse_xml(
            f'<w:abstractNum {nsdecls("w")} w:abstractNumId="90">'
            f"{_DECIMAL_LEVEL}</w:abstractNum>"
        )
    )
    for num_id, level_override in level_overrides.items():
        num = parse_xml(
            f'<w:num {nsdecls("w")} w:numId="{num_id}">'
            '<w:abstractNumId w:val="90"/></w:num>'
        )
        if level_override is not None:
            num.append(
                parse_xml(
                    f'<w:lvlOverride {nsdecls("w")} w:ilvl="0">'
                    f"{level_override}</w:lvlOverride>"
                )
            )
        numbering.append(num)
    for text, num_id in paragraphs:
        paragraph = document.add_paragraph(text)
        if num_id is not None:
            num_pr = paragraph._p.get_or_add_pPr().get_or_add_numPr()
            num_pr.get_or_add_ilvl().val = 0
            num_pr.get_or_add_numId().val = num_id
    return document


def _list_markers(doc: DoclingDocument) -> list[tuple[str, str]]:
    return [
        (item.text, item.marker)
        for item, _ in doc.iterate_items()
        if isinstance(item, ListItem)
    ]


_LETTERED_LEVEL = (
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/>'
    '<w:lvlText w:val="%1)"/></w:lvl>'
)


def test_start_override_sets_the_start_of_a_numbering_instance(tmp_path):
    """A w:startOverride sets the number at which a numbering instance starts.

    Instances 91 and 92 use the same abstract numbering definition. Instance
    92 starts level 0 at 5 (ECMA-376 Part 1, 17.9.27). The value applies when
    the instance starts. After other content, the instance continues its count.
    """
    document = _numbering_instances_docx(
        {91: None, 92: '<w:startOverride w:val="5"/>'},
        [
            ("alpha", 91),
            ("beta", 91),
            ("The list starts again at 5:", None),
            ("epsilon", 92),
            ("zeta", 92),
            ("A note between two items.", None),
            ("eta", 92),
        ],
    )

    doc = _convert_docx(document, tmp_path)

    assert _list_markers(doc) == [
        ("alpha", "1."),
        ("beta", "2."),
        ("epsilon", "5."),
        ("zeta", "6."),
        ("eta", "7."),
    ]
    assert doc.export_to_markdown() == (
        "1. alpha\n"
        "2. beta\n"
        "\n"
        "The list starts again at 5:\n"
        "\n"
        "5. epsilon\n"
        "6. zeta\n"
        "\n"
        "A note between two items.\n"
        "\n"
        "7. eta"
    )


def test_level_override_replaces_the_abstract_level(tmp_path):
    """A w:lvl in a w:lvlOverride replaces the level of the abstract definition.

    Instances 93 and 94 replace the decimal ``%1.`` level with a lettered
    ``%1)`` level (ECMA-376 Part 1, 17.9.9). Instance 94 also has a
    w:startOverride, which has priority over the w:start of its w:lvl.
    """
    document = _numbering_instances_docx(
        {93: _LETTERED_LEVEL, 94: f'<w:startOverride w:val="3"/>{_LETTERED_LEVEL}'},
        [
            ("apple", 93),
            ("banana", 93),
            ("More fruit:", None),
            ("cherry", 94),
            ("date", 94),
        ],
    )

    doc = _convert_docx(document, tmp_path)

    assert _list_markers(doc) == [
        ("apple", "a)"),
        ("banana", "b)"),
        ("cherry", "c)"),
        ("date", "d)"),
    ]
    assert doc.export_to_markdown() == (
        "- a) apple\n- b) banana\n\nMore fruit:\n\n- c) cherry\n- d) date"
    )


@pytest.mark.parametrize(
    "start_override",
    [
        '<w:startOverride w:val=""/>',
        '<w:startOverride w:val="five"/>',
        '<w:startOverride w:val="-1"/>',
        '<w:startOverride w:val="32768"/>',
        '<w:startOverride w:val="1000000000000000000000000000000"/>',
        "<w:startOverride/>",
    ],
)
def test_invalid_start_override_keeps_the_start_of_the_level(tmp_path, start_override):
    """A w:startOverride without a usable value is ignored.

    The level's own w:start applies. A large value would otherwise build very long
    letter or Roman markers, or fail the conversion.
    """
    document = _numbering_instances_docx(
        {95: f"{start_override}{_LETTERED_LEVEL}"},
        [("apple", 95), ("banana", 95)],
    )

    doc = _convert_docx(document, tmp_path)

    assert _list_markers(doc) == [("apple", "a)"), ("banana", "b)")]


_SUB_LIST = [
    ("one", 0),
    ("one-a", 1),
    ("one-b", 1),
    ("two", 0),
    ("two-a", 1),
    ("two-b", 1),
]


def _sub_level(start: int, lvl_restart: str = "") -> str:
    """Return a decimal ``%1.%2.`` level 1 that starts at ``start``.

    ``lvl_restart`` is an optional ``w:lvlRestart`` element for the level.
    """
    return (
        f'<w:lvl w:ilvl="1"><w:start w:val="{start}"/><w:numFmt w:val="decimal"/>'
        f'{lvl_restart}<w:lvlText w:val="%1.%2."/></w:lvl>'
    )


def _sub_list_docx(sub_level: str, level_override: str | None):
    """Build the paragraphs of ``_SUB_LIST`` as a list with two levels.

    Level 0 of the abstract numbering definition is decimal, ``%1.``, start 1,
    and ``sub_level`` is level 1. ``level_override`` is the content of the
    ``w:lvlOverride`` of numbering instance 91 for level 1, or ``None`` for no
    override. ``_SUB_LIST`` gives the text and ``w:ilvl`` of each paragraph.
    """
    document = Document()
    numbering = document.part.numbering_part.element
    # All w:abstractNum elements come before the first w:num.
    numbering.find(qn("w:num")).addprevious(
        parse_xml(
            f'<w:abstractNum {nsdecls("w")} w:abstractNumId="90">'
            f"{_DECIMAL_LEVEL}{sub_level}</w:abstractNum>"
        )
    )
    num = parse_xml(
        f'<w:num {nsdecls("w")} w:numId="91"><w:abstractNumId w:val="90"/></w:num>'
    )
    if level_override is not None:
        num.append(
            parse_xml(
                f'<w:lvlOverride {nsdecls("w")} w:ilvl="1">'
                f"{level_override}</w:lvlOverride>"
            )
        )
    numbering.append(num)
    for text, ilvl in _SUB_LIST:
        num_pr = document.add_paragraph(text)._p.get_or_add_pPr().get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = ilvl
        num_pr.get_or_add_numId().val = 91
    return document


_RESTARTS_AT_3 = ["1.", "1.3.", "1.4.", "2.", "2.3.", "2.4."]


@pytest.mark.parametrize(
    ("sub_level", "level_override", "markers"),
    [
        pytest.param(_sub_level(3), None, _RESTARTS_AT_3, id="start"),
        pytest.param(
            _sub_level(1),
            '<w:startOverride w:val="3"/>',
            _RESTARTS_AT_3,
            id="start-override",
        ),
        pytest.param(
            _sub_level(3, '<w:lvlRestart w:val="1"/>'),
            None,
            _RESTARTS_AT_3,
            id="start-with-lvl-restart-ignored",
        ),
        pytest.param(
            _sub_level(0),
            None,
            ["1.", "1.0.", "1.1.", "2.", "2.0.", "2.1."],
            id="start-0",
        ),
        pytest.param(
            _sub_level(1),
            None,
            ["1.", "1.1.", "1.2.", "2.", "2.1.", "2.2."],
            id="start-1",
        ),
    ],
)
def test_restarted_level_starts_at_its_start_value(
    tmp_path, sub_level, level_override, markers
):
    """A level that restarts after a higher level advances starts at its start value.

    The start value is the w:startOverride of the numbering instance for the
    level, else the w:start of the level. It applies when the level starts and
    whenever it restarts (ECMA-376 Part 1, 17.9.26 and 17.9.27). The backend
    does not read w:lvlRestart. In the ``start-with-lvl-restart-ignored`` case,
    w:lvlRestart names level 0 (the value is one-based), which is also the
    default for level 1, so the level restarts as it does without it.
    """
    doc = _convert_docx(_sub_list_docx(sub_level, level_override), tmp_path)

    assert _list_markers(doc) == [
        (text, marker) for (text, _), marker in zip(_SUB_LIST, markers, strict=True)
    ]


def test_markdown_shows_restarted_level_at_its_start_value(tmp_path):
    """The Markdown export shows level 1 at its w:startOverride after each restart."""
    doc = _convert_docx(
        _sub_list_docx(_sub_level(1), '<w:startOverride w:val="3"/>'), tmp_path
    )

    assert doc.export_to_markdown() == (
        "1. one\n"
        "    - 1.3. one-a\n"
        "    - 1.4. one-b\n"
        "2. two\n"
        "    - 2.3. two-a\n"
        "    - 2.4. two-b"
    )
