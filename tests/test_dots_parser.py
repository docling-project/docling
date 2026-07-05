"""Tests for dots.ocr / dots.mocr JSON layout parser."""

import json

import pytest
from docling_core.types.doc import DocItemLabel, Size

from docling.utils.dots_utils import _clean_json, parse_dots_json


@pytest.fixture
def page_size() -> Size:
    """Standard page size for tests: 500x700 points."""
    return Size(width=500.0, height=700.0)


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
