# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for dots.ocr / dots.mocr JSON layout parser."""

import json
import sys
from pathlib import Path

import pytest
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    InlineGroup,
    RichTableCell,
    Script,
    Size,
)

from docling.utils.dots_utils import _clean_json, parse_dots_json


@pytest.fixture
def page_size() -> Size:
    """Standard page size for tests: 500x700 points."""
    return Size(width=500.0, height=700.0)


@pytest.mark.parametrize(
    "category",
    [
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "Caption",
        "Footnote",
        "Page-header",
        "Page-footer",
    ],
)
def test_markdown_emphasis_roundtrip(category: str, page_size: Size, tmp_path: Path):
    text = "Mostly **one** important *factor* with __bold__ and _italic_ and ***both***"
    if category == "Section-header":
        text = "### " + text
    data = [{"bbox": [10, 20, 300, 50], "category": category, "text": text}]
    doc = parse_dots_json(json.dumps(data), page_size, page_no=1)
    assert any(isinstance(group, InlineGroup) for group in doc.groups)
    owner = next(item for item in doc.texts if item.prov)
    assert owner.prov[0].bbox.l == 10
    if category == "Section-header":
        assert owner.level == 2
    if category in {"Title", "Section-header", "List-item"}:
        assert owner.orig == text

    archive = tmp_path / "emphasis.dclx"
    doc.save_as_doclang_archive(archive)
    imported = DoclingDocument.load_from_doclang_archive(archive)
    # Core exports styled heading children but currently flattens them on import.
    documents = [doc] if category in {"Title", "Section-header"} else [doc, imported]
    for mapped in documents:
        styled = {
            item.text: (item.formatting.bold, item.formatting.italic)
            for item in mapped.texts
            if item.formatting
        }
        assert styled == {
            "one": (True, False),
            "factor": (False, True),
            "bold": (True, False),
            "italic": (False, True),
            "both": (True, True),
        }
        assert all(
            "*" not in item.text and "_" not in item.text for item in mapped.texts
        )
        assert "<bold>one</bold>" in mapped.export_to_doclang()


def test_markdown_emphasis_preserves_math_code_and_html(page_size: Size):
    text = (
        r"**Bold $x_1 * y_2 < 2$** and *italic* H<sub>2</sub>O "
        r"`**code**` <code>_literal_</code> and \*escaped\* snake_case"
    )
    data = [
        {"bbox": [10, 20, 300, 50], "category": "Text", "text": text},
        {"bbox": [10, 60, 300, 90], "category": "Formula", "text": r"x_1 * y_2 * z_3"},
        {
            "bbox": [10, 100, 300, 130],
            "category": "Text",
            "text": "```python\n'**literal**'\n```",
        },
    ]
    doc = parse_dots_json(json.dumps(data), page_size, page_no=1)
    assert any(
        item.text == "Bold $x_1 * y_2 < 2$" and item.formatting.bold
        for item in doc.texts
    )
    assert any(
        item.text == "2" and item.formatting.script == Script.SUB for item in doc.texts
    )
    assert any("*escaped* snake_case" in item.text for item in doc.texts)
    assert [item.text for item in doc.texts if isinstance(item, CodeItem)] == [
        "**code**",
        "_literal_",
        "'**literal**'",
    ]
    assert (
        next(item.text for item in doc.texts if item.label == DocItemLabel.FORMULA)
        == r"x_1 * y_2 * z_3"
    )


@pytest.mark.parametrize(
    "text",
    [
        "snake_case and a_b_c",
        r"\*literal\*",
        "unmatched **bold",
        "$x_1 * y_2 * z_3$",
        "______",
    ],
)
def test_literal_markers_stay_literal(text: str, page_size: Size):
    data = [{"bbox": [10, 20, 300, 50], "category": "Text", "text": text}]
    doc = parse_dots_json(json.dumps(data), page_size, page_no=1)
    assert len(doc.texts) == 1
    assert doc.texts[0].text == text
    assert doc.texts[0].formatting is None


def test_missing_markdown_dependency_has_install_hint(page_size: Size, monkeypatch):
    monkeypatch.setitem(sys.modules, "marko", None)
    data = [{"bbox": [10, 20, 300, 50], "category": "Text", "text": "**bold**"}]
    with pytest.raises(ImportError, match="format-markdown"):
        parse_dots_json(json.dumps(data), page_size, page_no=1)


class TestParseSingleTextElement:
    def test_text_and_bbox(self, page_size: Size):
        data = [{"bbox": [10, 20, 300, 50], "category": "Text", "text": "Hello world"}]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        assert item.label == DocItemLabel.TEXT
        assert "Hello world" in item.text

        # No model_image_size => scale 1:1, bbox unchanged
        prov = item.prov[0]
        assert abs(prov.bbox.l - 10.0) < 0.01
        assert abs(prov.bbox.t - 20.0) < 0.01
        assert abs(prov.bbox.r - 300.0) < 0.01
        assert abs(prov.bbox.b - 50.0) < 0.01


class TestParseTableElement:
    def test_table_html(self, page_size: Size):
        html = (
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        data = [{"bbox": [0, 0, 100, 100], "category": "Table", "text": html}]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        assert item.label == DocItemLabel.TABLE
        # Table should have parsed cells
        assert item.data.num_rows == 2
        assert item.data.num_cols == 2

    def test_rich_cells_and_caption(self, page_size: Size):
        html = (
            "<table><caption>Results <sup>1</sup></caption>"
            "<tr><td><b>Bold</b> <i>value</i><sup>1</sup><sub>2</sub></td></tr></table>"
        )
        data = [{"bbox": [0, 0, 100, 100], "category": "Table", "text": html}]

        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        table = doc.tables[0]
        assert isinstance(table.data.table_cells[0], RichTableCell)
        assert len(table.captions) == 1
        assert table.captions[0].resolve(doc).text == "Results 1"
        assert "<caption>" in doc.export_to_doclang()
        assert any(item.formatting and item.formatting.bold for item in doc.texts)
        assert any(item.formatting and item.formatting.italic for item in doc.texts)
        assert {Script.SUPER, Script.SUB} <= {
            item.formatting.script for item in doc.texts if item.formatting
        }
        assert table.prov
        assert all(not item.prov for item in doc.texts)


class TestParsePictureNoTextField:
    def test_picture_no_text(self, page_size: Size):
        data = [{"bbox": [50, 50, 200, 200], "category": "Picture"}]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        assert item.label == DocItemLabel.PICTURE


class TestParseMalformedJsonTruncated:
    def test_truncated_array(self, page_size: Size):
        # Valid first element, truncated second element
        raw = '[{"bbox": [0,0,100,100], "category": "Text", "text": "OK"}, {"bbox": [0,0,100,1'
        doc = parse_dots_json(raw, page_size, page_no=1)

        # Should recover at least the first valid element
        items = list(doc.iterate_items())
        # After cleanup the truncated element is dropped by json.loads
        # because _clean_json closes the array after the last }
        assert len(items) >= 1
        item, _ = items[0]
        assert "OK" in item.text

    def test_leading_garbage(self, page_size: Size):
        raw = 'some preamble text [{"bbox": [10,20,30,40], "category": "Text", "text": "hi"}]'
        doc = parse_dots_json(raw, page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1

    def test_no_json_structure(self, page_size: Size):
        raw = "completely invalid output with no brackets"
        doc = parse_dots_json(raw, page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 0


class TestParseEmptyJson:
    def test_empty_array(self, page_size: Size):
        doc = parse_dots_json("[]", page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 0

    def test_empty_string(self, page_size: Size):
        doc = parse_dots_json("", page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 0


class TestParseWithRescaling:
    def test_model_image_size_rescales_bbox(self, page_size: Size):
        # model input is 1000x1000; page is 500x700
        # scale_x = 500/1000 = 0.5, scale_y = 700/1000 = 0.7
        model_size = Size(width=1000.0, height=1000.0)
        data = [{"bbox": [100, 200, 400, 300], "category": "Text", "text": "Scaled"}]
        doc = parse_dots_json(
            json.dumps(data),
            page_size,
            page_no=1,
            model_image_size=model_size,
        )

        items = list(doc.iterate_items())
        assert len(items) == 1

        prov = items[0][0].prov[0]
        assert abs(prov.bbox.l - 50.0) < 0.01  # 100 * 0.5
        assert abs(prov.bbox.t - 140.0) < 0.01  # 200 * 0.7
        assert abs(prov.bbox.r - 200.0) < 0.01  # 400 * 0.5
        assert abs(prov.bbox.b - 210.0) < 0.01  # 300 * 0.7


class TestParseMultipleCategories:
    def test_four_categories(self, page_size: Size):
        data = [
            {"bbox": [0, 0, 100, 20], "category": "Title", "text": "Doc Title"},
            {"bbox": [0, 30, 100, 60], "category": "Section-header", "text": "Intro"},
            {"bbox": [0, 70, 100, 150], "category": "Text", "text": "Body text"},
            {"bbox": [0, 160, 100, 250], "category": "Picture"},
        ]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 4

        labels = [item.label for item, _ in items]
        assert DocItemLabel.TITLE in labels
        assert DocItemLabel.SECTION_HEADER in labels
        assert DocItemLabel.TEXT in labels
        assert DocItemLabel.PICTURE in labels

    def test_normalizes_headings_code_and_inline_html(self, page_size: Size):
        data = [
            {"bbox": [0, 0, 100, 20], "category": "Title", "text": "# Title"},
            {
                "bbox": [0, 30, 100, 50],
                "category": "Section-header",
                "text": "### Section",
            },
            {
                "bbox": [0, 60, 100, 100],
                "category": "Text",
                "text": "```python\nprint('ok')\n```",
            },
            {
                "bbox": [0, 110, 100, 140],
                "category": "Text",
                "text": "H<sub>2</sub>O",
            },
            {
                "bbox": [0, 150, 100, 180],
                "category": "List-item",
                "text": "Keep the inequality $0<a<2$ literal.",
            },
        ]

        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)

        assert (doc.texts[0].text, doc.texts[0].orig) == ("Title", "# Title")
        assert (doc.texts[1].text, doc.texts[1].orig, doc.texts[1].level) == (
            "Section",
            "### Section",
            2,
        )
        code = next(item for item in doc.texts if isinstance(item, CodeItem))
        assert code.text == "print('ok')"
        assert code.orig == "```python\nprint('ok')\n```"
        assert any(
            item.text == "2" and item.formatting.script == Script.SUB
            for item in doc.texts
        )
        assert any(
            item.text == "Keep the inequality $0<a<2$ literal." for item in doc.texts
        )


class TestCleanJson:
    def test_strips_leading_text(self):
        assert _clean_json('garbage [{"a":1}]') == '[{"a":1}]'

    def test_closes_truncated_array(self):
        result = _clean_json('[{"a":1}, {"b":2')
        assert result == '[{"a":1}]'

    def test_no_bracket(self):
        assert _clean_json("no json here") == "[]"

    def test_already_valid(self):
        assert _clean_json('[{"a":1}]') == '[{"a":1}]'


class TestParseFormulaAndFootnote:
    """Verify less common categories route correctly."""

    def test_formula(self, page_size: Size):
        data = [{"bbox": [10, 10, 200, 40], "category": "Formula", "text": r"E = mc^2"}]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1
        item, _ = items[0]
        assert item.label == DocItemLabel.FORMULA

    def test_footnote(self, page_size: Size):
        data = [
            {"bbox": [10, 600, 300, 620], "category": "Footnote", "text": "See ref 1."}
        ]
        doc = parse_dots_json(json.dumps(data), page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1
        item, _ = items[0]
        assert item.label == DocItemLabel.FOOTNOTE
