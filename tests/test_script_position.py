# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for finding superscript and subscript text from PDF character geometry."""

import re

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.document import InlineGroup, ListItem, Script, TextItem

from docling.datamodel.pipeline_options import ScriptPositionOptions
from docling.models.stages.script_position.script_position_model import (
    ScriptPositionModel,
    _Page,
)
from docling.utils.pdf_char_geometry import PdfChar, PdfPageChars
from docling.utils.script_position import script_positions, script_runs

B, SUP, SUB = Script.BASELINE, Script.SUPER, Script.SUB


def _line(*segments, y: float = 700.0, x: float = 50.0) -> list[PdfChar]:
    """Lay out (text, size, rise) segments left to right on one baseline."""
    chars = []
    for text, size, rise in segments:
        for ch in text:
            width = 0.5 * size
            chars.append(
                PdfChar(
                    text=ch,
                    left=x,
                    bottom=y + rise - 0.2 * size,
                    right=x + width,
                    top=y + rise + 0.8 * size,
                    baseline=y + rise,
                    size=size,
                    horizontal=True,
                )
            )
            x += width
    return chars


def _runs(text: str, chars: list[PdfChar]) -> list[tuple[str, Script]]:
    runs = script_runs(text, chars)
    assert runs is not None
    return [(r.text, r.script) for r in runs]


def _marked(chars: list[PdfChar]) -> str:
    """Render the positions found as <sup>/<sub> markup, ignoring spaces."""
    out = ""
    for char, script in zip(chars, script_positions(chars)):
        if char.text.isspace():
            continue
        out += (
            char.text
            if script == B
            else f"<{script.value}>{char.text}</{script.value}>"
        )
    return out.replace("</super><super>", "").replace("</sub><sub>", "")


@pytest.mark.parametrize(
    "segments, expected",
    [
        pytest.param(
            [("E = mc", 12, 0), ("2", 7, 4.5)], "E=mc<super>2</super>", id="exponent"
        ),
        pytest.param(
            [("H", 12, 0), ("2", 7, -2.5), ("O", 12, 0)], "H<sub>2</sub>O", id="formula"
        ),
        pytest.param(
            # Word lowers a subscript by under a tenth of the font size.
            [("CO", 12, 0), ("2", 7, -1.1), (" gas", 12, 0)],
            "CO<sub>2</sub>gas",
            id="word-subscript",
        ),
        pytest.param(
            [("the 1", 12, 0), ("st", 7, 4.5), (" of May", 12, 0)],
            "the1<super>st</super>ofMay",
            id="ordinal",
        ),
        pytest.param(
            [("as shown", 12, 0), ("1,2\u20134", 7, 4.5), (". Next", 12, 0)],
            "asshown<super>1,2\u20134</super>.Next",
            id="citation",
        ),
        pytest.param(
            [("3", 6, 3), ("This footnote", 9, 0)],
            "<super>3</super>Thisfootnote",
            id="footnote-marker-first",
        ),
        pytest.param(
            [("Author", 12, 0), ("*", 7, 4.5), (" and", 12, 0)],
            "Author<super>*</super>and",
            id="footnote-mark",
        ),
        pytest.param(
            # The last line of a paragraph can hold fewer letters than its citation.
            [("al.", 12, 0), ("[12]", 7, 4.5)],
            "al.<super>[12]</super>",
            id="citation-on-short-line",
        ),
    ],
)
def test_marks_scripts(segments, expected):
    assert _marked(_line(*segments)) == expected


@pytest.mark.parametrize(
    "segments",
    [
        # Drawn high but full size: not a script.
        pytest.param([("grew 12% at 25\u00b0C", 12, 0)], id="percent-degree"),
        # Smaller but on the baseline.
        pytest.param(
            [("S", 12, 0), ("MALL ", 9, 0), ("C", 12, 0), ("APS", 9, 0)],
            id="small-caps",
        ),
        # Shifted but full size.
        pytest.param(
            [("baseline ", 12, 0), ("jitter", 12, 1.5), (" here", 12, 0)],
            id="wavy-baseline",
        ),
        # A symbol font drawn small and raised.
        pytest.param(
            [("Concepts", 12, 0), ("\u2192", 7, 2), ("Docs", 12, 0)], id="arrow"
        ),
        # The raised A of the LaTeX logo sits inside the word.
        pytest.param([("L", 12, 0), ("A", 8, 3), ("TeX", 12, 0)], id="latex-logo"),
        # A drop cap is set larger and lower than the line it starts.
        pytest.param([("T", 36, -24), ("his line goes on", 12, 0)], id="drop-cap"),
        # A large symbol does not set the size of the text around it.
        pytest.param(
            [("sum ", 10, 0), ("\u2211", 18, -3), (" of x", 10, 0)], id="large-symbol"
        ),
    ],
)
def test_leaves_other_text_alone(segments):
    chars = _line(*segments)
    assert all(script == B for script in script_positions(chars))


def test_small_text_away_from_body_is_not_a_script():
    chars = _line(("Report 2024", 12, 0)) + _line(("page 3", 7, 4.5), x=400.0)
    assert all(script == B for script in script_positions(chars))


def test_measures_each_line_against_its_own_baseline():
    chars = _line(("x", 12, 0), ("2", 7, 4.5), y=700.0) + _line(
        ("H", 12, 0), ("2", 7, -2.5), ("O", 12, 0), y=680.0
    )
    assert _marked(chars) == "x<super>2</super>H<sub>2</sub>O"


@pytest.mark.parametrize("subscript", ["i", "ijk"])
def test_stacked_sub_and_superscript(subscript):
    chars = _line(("x", 12, 0), (subscript, 7, -2.5))
    # The superscript is drawn back over the start of the subscript, however wide.
    chars += _line(("2", 7, 4.5), x=chars[1].left)
    assert script_positions(chars) == [B, *[SUB] * len(subscript), SUP]


def test_footnote_marker_starting_a_tightly_set_line():
    # Lines 9 points apart: the raised marker is less than a line below the line above.
    chars = _line(("the line above it", 9, 0), y=700.0)
    chars += _line(("3", 6, 3), ("Note text", 9, 0), y=691.0)
    assert _marked(chars) == "thelineaboveit<super>3</super>Notetext"


def test_runs_have_no_whitespace_at_their_edges():
    # As the Word backend's runs: the space between two runs is left to the export.
    chars = _line(("E = mc", 12, 0), ("2", 7, 4.5), (" in 1905.", 12, 0))
    assert _runs(" E = mc 2 in 1905. ", chars) == [
        ("E = mc", B),
        ("2", SUP),
        ("in 1905.", B),
    ]


def test_runs_match_through_ligatures_and_dashes():
    chars = _line(("\ufb01rst", 12, 0), ("1\u20132", 7, 4.5))
    assert _runs("first1-2", chars) == [("first", B), ("1-2", SUP)]


def test_runs_point_to_their_characters():
    chars = _line(("H", 12, 0), ("2", 7, -2.5), ("O", 12, 0))
    runs = script_runs("H2O", chars)
    assert runs is not None
    assert [r.chars for r in runs] == [(0,), (1,), (2,)]


def test_no_runs_without_scripts_or_when_text_does_not_match():
    plain = _line(("plain text", 12, 0))
    assert script_runs("plain text", plain) is None
    chars = _line(("E = mc", 12, 0), ("2", 7, 4.5))
    assert script_runs("something else entirely", chars) is None


def test_runs_ignore_characters_of_a_neighbouring_line():
    # The item's box also catches a raised character of the line below it.
    chars = _line(("E = mc", 12, 0), ("2", 7, 4.5), y=700.0)
    chars += _line(("x", 12, 0), ("3", 7, 4.5), y=686.0)
    assert _runs("E = mc 2", chars) == [("E = mc", B), ("2", SUP)]


def _box(chars: list[PdfChar]) -> BoundingBox:
    return BoundingBox(
        l=min(c.left for c in chars),
        b=min(c.bottom for c in chars),
        r=max(c.right for c in chars),
        t=max(c.top for c in chars),
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


def _item_runs(
    document: DoclingDocument, group: InlineGroup
) -> list[tuple[str, Script]]:
    runs = []
    for ref in group.children:
        item = ref.resolve(document)
        assert isinstance(item, TextItem)
        runs.append((item.text, item.formatting.script if item.formatting else B))
    return runs


def test_stage_rewrites_paragraphs_and_list_items():
    paragraph = _line(("E = mc", 12, 0), ("2", 7, 4.5), (" here", 12, 0), y=700.0)
    list_item = _line(("H", 12, 0), ("2", 7, -2.5), ("O", 12, 0), y=650.0)
    document = DoclingDocument(name="test")
    document.add_page(page_no=1, size=Size(width=612, height=792))
    document.add_text(
        label=DocItemLabel.TEXT,
        text="E = mc 2 here",
        prov=ProvenanceItem(page_no=1, bbox=_box(paragraph), charspan=(0, 13)),
    )
    document.add_list_item(
        text="H 2 O",
        parent=document.add_list_group(),
        prov=ProvenanceItem(page_no=1, bbox=_box(list_item), charspan=(0, 5)),
    )
    page = _Page(
        PdfPageChars(
            chars=paragraph + list_item,
            rotation=0,
            display_size=Size(width=612, height=792),
            origin=(0.0, 0.0),
        )
    )

    model = ScriptPositionModel(ScriptPositionOptions(enabled=True))
    model.apply_scripts(document, model.find_scripts(document, {1: page}.get))

    groups = [g for g in document.groups if isinstance(g, InlineGroup)]
    assert [_item_runs(document, g) for g in groups] == [
        [("E = mc", B), ("2", SUP), ("here", B)],
        [("H", B), ("2", SUB), ("O", B)],
    ]
    # Each item keeps its place and holds its runs; the list item keeps its marker.
    paragraph_item, list_item_item = document.texts[0], document.texts[1]
    assert isinstance(list_item_item, ListItem)
    assert paragraph_item.text == "" and list_item_item.text == ""
    assert groups[0].parent == paragraph_item.get_ref()
    assert groups[1].parent == list_item_item.get_ref()
    # A run's box is the box of its own characters.
    superscript = groups[0].children[1].resolve(document)
    assert isinstance(superscript, TextItem)
    assert superscript.prov[0].bbox == _box([paragraph[6]])


def _stage(document: DoclingDocument, pages: dict[int, list[PdfChar]]) -> None:
    lookup = {
        page_no: _Page(
            PdfPageChars(
                chars=chars,
                rotation=0,
                display_size=Size(width=612, height=792),
                origin=(0.0, 0.0),
            )
        )
        for page_no, chars in pages.items()
    }
    model = ScriptPositionModel(ScriptPositionOptions(enabled=True))
    model.apply_scripts(document, model.find_scripts(document, lookup.get))


def test_paragraphs_stay_separate_blocks_in_html():
    """Two paragraphs in a row with scripts must not run together in a browser."""
    first = _line(("E = mc", 12, 0), ("2", 7, 4.5), y=700.0)
    second = _line(("H", 12, 0), ("2", 7, -2.5), ("O", 12, 0), y=650.0)
    document = DoclingDocument(name="test")
    document.add_page(page_no=1, size=Size(width=612, height=792))
    for text, chars in (("E = mc 2", first), ("H 2 O", second)):
        document.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(page_no=1, bbox=_box(chars), charspan=(0, len(text))),
        )
    _stage(document, {1: first + second})

    html = document.export_to_html()
    assert len(re.findall(r"<p>\s*<span class='inline-group'>", html)) == 2


def test_run_crossing_a_page_break_keeps_both_pages():
    """The text after a script can continue on the next page."""
    page_1 = _line(("Water is H", 12, 0), ("2", 7, -2.5), ("O and", 12, 0), y=60.0)
    page_2 = _line(("more text", 12, 0), y=740.0)
    document = DoclingDocument(name="test")
    for page_no in (1, 2):
        document.add_page(page_no=page_no, size=Size(width=612, height=792))
    text = "Water is H 2 O and more text"
    item = document.add_text(
        label=DocItemLabel.TEXT,
        text=text,
        prov=ProvenanceItem(page_no=1, bbox=_box(page_1), charspan=(0, 18)),
    )
    item.prov.append(
        ProvenanceItem(page_no=2, bbox=_box(page_2), charspan=(18, len(text)))
    )
    _stage(document, {1: page_1, 2: page_2})

    last = next(t for t in document.texts if t.text.startswith("O"))
    assert last.text == "O and more text"
    assert [p.page_no for p in last.prov] == [1, 2]
    assert [last.text[slice(*p.charspan)] for p in last.prov] == ["O and ", "more text"]
    assert last.prov[1].bbox == _box(page_2)
